"""Forward-only decoder-layer block swap with CUDA-stream prefetch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import weakref

import torch
from torch import nn


_ALIGNMENT = 256
_HOST_PAGE_BYTES = 4096
_MIN_PIN_CHUNK_BYTES = 16 * 2**20


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _cuda_call_succeeded(result: Any) -> bool:
    if isinstance(result, tuple):
        result = result[0] if result else 0
    try:
        return int(result) == 0
    except (TypeError, ValueError):
        return result == 0


def _qualified_owner(layer: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = qualified_name.rpartition(".")
    return (layer.get_submodule(parent_name) if parent_name else layer), leaf


@dataclass
class _TensorRef:
    qualified_name: str
    owner: nn.Module
    leaf: str
    tensor: torch.Tensor
    parameter: nn.Parameter | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass
class _Placement:
    chunk_index: int
    chunk_offset: int
    slot_offset: int


@dataclass
class _HostChunk:
    tensor: torch.Tensor
    slot_offset: int
    owner: torch.Tensor | None = None


@dataclass
class _TensorBinding:
    owner: nn.Module
    leaf: str
    parameter: nn.Parameter | None
    host_view: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    slot_offset: int
    device_view: torch.Tensor | None = None

    def bind_host(self) -> None:
        if self.parameter is not None:
            self.parameter.data = self.host_view
        else:
            self.owner._buffers[self.leaf] = self.host_view

    def bind_device(self) -> None:
        if self.device_view is None:
            raise RuntimeError("Block-swap device views have been released")
        if self.parameter is not None:
            self.parameter.data = self.device_view
        else:
            self.owner._buffers[self.leaf] = self.device_view


@dataclass
class _LayerPack:
    chunks: tuple[_HostChunk, ...]
    bindings: tuple[_TensorBinding, ...]
    raw_bytes: int
    packed_bytes: int
    pin_method: str

    @property
    def pinned(self) -> bool:
        return self.pin_method in {"cudaHostRegister", "pin_memory"}


def _tensor_refs(layer: nn.Module) -> list[_TensorRef]:
    refs: list[_TensorRef] = []
    for qualified_name, parameter in layer.named_parameters():
        if parameter is None:
            continue
        owner, leaf = _qualified_owner(layer, qualified_name)
        refs.append(
            _TensorRef(
                qualified_name,
                owner,
                leaf,
                parameter.detach(),
                parameter,
                tuple(parameter.shape),
                parameter.dtype,
                parameter.numel() * parameter.element_size(),
            )
        )
    for qualified_name, buffer in layer.named_buffers():
        if buffer is None:
            continue
        owner, leaf = _qualified_owner(layer, qualified_name)
        refs.append(
            _TensorRef(
                qualified_name,
                owner,
                leaf,
                buffer.detach(),
                None,
                tuple(buffer.shape),
                buffer.dtype,
                buffer.numel() * buffer.element_size(),
            )
        )
    return refs


def _exact_layout(refs: Sequence[_TensorRef]) -> tuple[int, list[_Placement]]:
    cursor = 0
    placements: list[_Placement] = []
    for ref in refs:
        cursor = _align_up(cursor)
        placements.append(_Placement(0, cursor, cursor))
        cursor += ref.nbytes
    return _align_up(cursor), placements


def _remaining_packed_bytes(refs: Sequence[_TensorRef], start: int) -> int:
    cursor = 0
    for ref in refs[start:]:
        cursor = _align_up(cursor)
        cursor += ref.nbytes
    return _align_up(cursor)


def _chunked_layout(refs: Sequence[_TensorRef]) -> tuple[list[int], list[_Placement]]:
    """Pack ordered tensors into binary-decomposed, power-of-two pinned chunks."""

    if not refs:
        return [], []
    if not any(ref.nbytes for ref in refs):
        return [0], [_Placement(0, 0, 0) for _ in refs]

    capacities: list[int] = []
    placements: list[_Placement] = []
    ref_index = 0
    slot_base = 0
    while ref_index < len(refs):
        remaining = max(_MIN_PIN_CHUNK_BYTES, _remaining_packed_bytes(refs, ref_index))
        units = (remaining + _MIN_PIN_CHUNK_BYTES - 1) // _MIN_PIN_CHUNK_BYTES
        capacity = (1 << (units.bit_length() - 1)) * _MIN_PIN_CHUNK_BYTES
        first_nonempty = next((ref for ref in refs[ref_index:] if ref.nbytes), None)
        if first_nonempty is not None:
            needed = _next_power_of_two(max(_MIN_PIN_CHUNK_BYTES, _align_up(first_nonempty.nbytes)))
            capacity = max(capacity, needed)

        chunk_index = len(capacities)
        capacities.append(capacity)
        cursor = 0
        placed_before = ref_index
        while ref_index < len(refs):
            ref = refs[ref_index]
            offset = _align_up(cursor)
            if ref.nbytes and offset + ref.nbytes > capacity:
                break
            placements.append(_Placement(chunk_index, offset, slot_base + offset))
            cursor = offset + ref.nbytes
            ref_index += 1
        if ref_index == placed_before:
            raise RuntimeError("Could not place a block-swap tensor in pinned host chunks")
        slot_base += capacity
    return capacities, placements


def _typed_view(
    byte_buffer: torch.Tensor,
    byte_offset: int,
    nbytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return byte_buffer.narrow(0, int(byte_offset), int(nbytes)).view(dtype).view(shape)


class BlockSwapManager:
    """Keep leading layers resident and stream the rest through fixed CUDA slots."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        resident: int,
        slots: int,
        device: torch.device | str,
        pin: bool = True,
        pin_budget_bytes: int | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._removed = False
        self._root_ref: weakref.ReferenceType[nn.Module] | None = None
        self._hook_handles: list[Any] = []
        self._registered_ranges: list[tuple[int, int]] = []
        self._gpu_slots: list[torch.Tensor] = []
        self._copy_pairs: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = []
        self._ready_events: list[torch.cuda.Event] = []
        self._barrier_events: list[torch.cuda.Event] = []
        self._kickoff_event: torch.cuda.Event | None = None
        self._forward_done_event: torch.cuda.Event | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._layer_packs: list[_LayerPack] = []
        self._log = log
        self._pageable_logged = False
        self._pinning_stopped = False
        self._issued: set[int] = set()
        self._bound: set[int] = set()
        self._forward_active = False
        self._forward_kind: str | None = None
        self._lazy_expected_next: int | None = None
        self._forwards = 0
        self._layer_loads = 0
        self._bytes_h2d = 0

        self._layers = tuple(layers)
        if not all(isinstance(layer, nn.Module) for layer in self._layers):
            raise TypeError("layers must contain only torch.nn.Module instances")
        if len({id(layer) for layer in self._layers}) != len(self._layers):
            raise ValueError("layers must not contain the same module more than once")
        if isinstance(resident, bool) or not isinstance(resident, int):
            raise TypeError("resident must be an integer")
        if not 0 <= resident <= len(self._layers):
            raise ValueError(f"resident must be between 0 and {len(self._layers)}")
        if isinstance(slots, bool) or not isinstance(slots, int):
            raise TypeError("slots must be an integer")
        if resident < len(self._layers) and slots < 1:
            raise ValueError("slots must be at least 1 when layers are swapped")
        if resident == len(self._layers) and slots < 0:
            raise ValueError("slots must be non-negative")

        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("BlockSwapManager requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        self.device = target
        self.resident = resident
        self.swapped = len(self._layers) - resident
        self.slots = slots if self.swapped else 0
        self._pin_requested = bool(pin)
        self._pin_budget_bytes = None if pin_budget_bytes is None else max(0, int(pin_budget_bytes))
        self._pin_accounted_bytes = 0

        self._validate_layout()
        try:
            for layer in self._layers[self.resident :]:
                self._layer_packs.append(self._pack_layer(layer))

            self.layer_bytes = max((pack.raw_bytes for pack in self._layer_packs), default=0)
            self.slot_bytes = max((pack.packed_bytes for pack in self._layer_packs), default=0)
            self.pinned_bytes = sum(pack.packed_bytes for pack in self._layer_packs if pack.pinned)
            self.pageable_bytes = sum(pack.packed_bytes for pack in self._layer_packs if not pack.pinned)
            self._allocate_device_state()
        except Exception:
            self._release_after_failed_init()
            raise

    @classmethod
    def install(cls, root: nn.Module, layers: Sequence[nn.Module], **kwargs: Any) -> "BlockSwapManager":
        """Create a manager, attach hooks, and flag ``root`` as block-swapped."""

        if not isinstance(root, nn.Module):
            raise TypeError("root must be a torch.nn.Module")
        existing = getattr(root, "_vcap_block_swap_manager", None)
        if existing is not None:
            raise RuntimeError("A block-swap manager is already installed on this model")
        manager = cls(layers, **kwargs)
        try:
            manager._attach(root)
            manager._emit_ready()
            manager._log = None
            return manager
        except Exception:
            manager.remove()
            raise

    def _validate_layout(self) -> None:
        for index, layer in enumerate(self._layers[: self.resident]):
            for ref in _tensor_refs(layer):
                if ref.tensor.device != self.device:
                    raise ValueError(
                        f"Resident layer {index} tensor {ref.qualified_name!r} is on "
                        f"{ref.tensor.device}, expected {self.device}"
                    )

        swapped_signatures: list[tuple[tuple[str, tuple[int, ...], torch.dtype], ...]] = []
        for index, layer in enumerate(self._layers[self.resident :], start=self.resident):
            refs = _tensor_refs(layer)
            for ref in refs:
                if ref.tensor.device.type != "cpu":
                    raise ValueError(
                        f"Swapped layer {index} tensor {ref.qualified_name!r} is on "
                        f"{ref.tensor.device}, expected cpu"
                    )
                if ref.tensor.layout != torch.strided:
                    raise ValueError(
                        f"Swapped layer {index} tensor {ref.qualified_name!r} must use strided layout"
                    )
            swapped_signatures.append(tuple((ref.qualified_name, ref.shape, ref.dtype) for ref in refs))

        if len(swapped_signatures) < 2:
            return
        expected = swapped_signatures[0]
        for relative, actual in enumerate(swapped_signatures[1:], start=1):
            if actual == expected:
                continue
            mismatch = 0
            limit = min(len(expected), len(actual))
            while mismatch < limit and expected[mismatch] == actual[mismatch]:
                mismatch += 1
            wanted = expected[mismatch] if mismatch < len(expected) else None
            found = actual[mismatch] if mismatch < len(actual) else None
            name = (found or wanted or ("<unknown>", (), torch.uint8))[0]
            absolute = self.resident + relative
            raise ValueError(
                f"Block swap layer {absolute} signature mismatch at {name!r}: "
                f"expected {wanted!r}, got {found!r}"
            )

    def _pin_budget_allows(self, nbytes: int) -> bool:
        if not self._pin_requested or self._pinning_stopped:
            return False
        if self._pin_budget_bytes is None:
            return True
        if self._pin_accounted_bytes + int(nbytes) <= self._pin_budget_bytes:
            return True
        self._pinning_stopped = True
        self._note_pageable("Block swap pin budget reached; remaining layers use pageable host memory.")
        return False

    def _try_registered_buffer(self, nbytes: int) -> _HostChunk | None:
        if nbytes <= 0:
            return None
        try:
            owner = torch.empty(nbytes + _HOST_PAGE_BYTES, dtype=torch.uint8, device="cpu")
            offset = (-owner.data_ptr()) % _HOST_PAGE_BYTES
            buffer = owner.narrow(0, offset, nbytes)
            cudart = torch.cuda.cudart()
            result = cudart.cudaHostRegister(buffer.data_ptr(), nbytes, 0)
            if not _cuda_call_succeeded(result):
                return None
        except Exception:
            return None
        self._registered_ranges.append((buffer.data_ptr(), nbytes))
        return _HostChunk(buffer, 0, owner)

    def _try_pinned_chunks(
        self,
        capacities: Sequence[int],
    ) -> tuple[_HostChunk, ...] | None:
        chunks: list[_HostChunk] = []
        slot_offset = 0
        try:
            for capacity in capacities:
                tensor = torch.empty(int(capacity), dtype=torch.uint8, pin_memory=True)
                chunks.append(_HostChunk(tensor, slot_offset))
                slot_offset += int(capacity)
        except Exception:
            return None
        return tuple(chunks)

    def _pageable_pack(
        self,
        packed_bytes: int,
        placements: list[_Placement],
    ) -> tuple[tuple[_HostChunk, ...], list[_Placement], str]:
        owner = torch.empty(packed_bytes + _ALIGNMENT, dtype=torch.uint8, device="cpu")
        offset = (-owner.data_ptr()) % _ALIGNMENT
        buffer = owner.narrow(0, offset, packed_bytes)
        method = "disabled" if not self._pin_requested else "pageable"
        return (_HostChunk(buffer, 0, owner),), placements, method

    def _pack_layer(self, layer: nn.Module) -> _LayerPack:
        refs = _tensor_refs(layer)
        raw_bytes = sum(ref.nbytes for ref in refs)
        exact_bytes, exact_placements = _exact_layout(refs)
        chunks: tuple[_HostChunk, ...]
        placements: list[_Placement]
        pin_method: str

        if exact_bytes == 0:
            chunks, placements, pin_method = self._pageable_pack(exact_bytes, exact_placements)
        elif self._pin_budget_allows(exact_bytes):
            registered = self._try_registered_buffer(exact_bytes)
            if registered is not None:
                chunks = (registered,)
                placements = exact_placements
                pin_method = "cudaHostRegister"
                self._pin_accounted_bytes += exact_bytes
            else:
                capacities, chunked_placements = _chunked_layout(refs)
                chunked_bytes = sum(capacities)
                if self._pin_budget_allows(chunked_bytes):
                    pinned = self._try_pinned_chunks(capacities)
                else:
                    pinned = None
                if pinned is not None:
                    chunks = pinned
                    placements = chunked_placements
                    pin_method = "pin_memory"
                    self._pin_accounted_bytes += chunked_bytes
                else:
                    chunks, placements, pin_method = self._pageable_pack(exact_bytes, exact_placements)
                    if not self._pinning_stopped:
                        self._pinning_stopped = True
                        self._note_pageable(
                            "Block swap could not pin host memory; remaining layers use pageable buffers."
                        )
        else:
            chunks, placements, pin_method = self._pageable_pack(exact_bytes, exact_placements)

        bindings: list[_TensorBinding] = []
        with torch.no_grad():
            for ref, placement in zip(refs, placements, strict=True):
                host_view = _typed_view(
                    chunks[placement.chunk_index].tensor,
                    placement.chunk_offset,
                    ref.nbytes,
                    ref.dtype,
                    ref.shape,
                )
                host_view.copy_(ref.tensor)
                bindings.append(
                    _TensorBinding(
                        ref.owner,
                        ref.leaf,
                        ref.parameter,
                        host_view,
                        ref.shape,
                        ref.dtype,
                        ref.nbytes,
                        placement.slot_offset,
                    )
                )
            for binding in bindings:
                binding.bind_host()

        packed_bytes = max((chunk.slot_offset + chunk.tensor.numel() for chunk in chunks), default=0)
        return _LayerPack(tuple(chunks), tuple(bindings), raw_bytes, packed_bytes, pin_method)

    def _allocate_device_state(self) -> None:
        if not self.swapped:
            return
        with torch.cuda.device(self.device):
            self._gpu_slots = [
                torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device)
                for _ in range(self.slots)
            ]
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._ready_events = [torch.cuda.Event() for _ in range(self.swapped)]
            self._barrier_events = [torch.cuda.Event() for _ in range(self.swapped)]
            self._kickoff_event = torch.cuda.Event()
            self._forward_done_event = torch.cuda.Event()

            for relative, pack in enumerate(self._layer_packs):
                slot = self._gpu_slots[relative % self.slots]
                pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
                for chunk in pack.chunks:
                    destination = slot.narrow(0, chunk.slot_offset, chunk.tensor.numel())
                    pairs.append((destination, chunk.tensor))
                for binding in pack.bindings:
                    binding.device_view = _typed_view(
                        slot,
                        binding.slot_offset,
                        binding.nbytes,
                        binding.dtype,
                        binding.shape,
                    )
                self._copy_pairs.append(tuple(pairs))

            # CUDA events are lazy. Record each once during installation so hooks
            # never create a device-side event resource on their first forward.
            current = torch.cuda.current_stream(self.device)
            for event in (
                *self._ready_events,
                *self._barrier_events,
                self._kickoff_event,
                self._forward_done_event,
            ):
                event.record(current)

    def _release_after_failed_init(self) -> None:
        for pack in self._layer_packs:
            for binding in pack.bindings:
                binding.bind_host()
                binding.device_view = None
        self._copy_pairs.clear()
        self._gpu_slots.clear()
        self._ready_events.clear()
        self._barrier_events.clear()
        self._copy_stream = None
        self._kickoff_event = None
        self._forward_done_event = None
        self._unregister_host_ranges()

    def _attach(self, root: nn.Module) -> None:
        self._root_ref = weakref.ref(root)
        root._vcap_block_swap = True
        root._vcap_block_swap_manager = self
        self._hook_handles.append(root.register_forward_pre_hook(self._root_pre_hook, prepend=True))
        self._hook_handles.append(
            root.register_forward_hook(self._root_post_hook, prepend=False, always_call=True)
        )
        for relative, layer in enumerate(self._layers[self.resident :]):
            self._hook_handles.append(
                layer.register_forward_pre_hook(
                    lambda module, args, index=relative: self._layer_pre_hook(index, module, args),
                    prepend=True,
                )
            )
            self._hook_handles.append(
                layer.register_forward_hook(
                    lambda module, args, output, index=relative: self._layer_post_hook(
                        index, module, args, output
                    ),
                    always_call=True,
                )
            )

    def _safe_log(self, message: str) -> None:
        callback = self._log
        if callback is None:
            return
        try:
            callback(message)
        except Exception:
            pass

    def _note_pageable(self, message: str) -> None:
        if self._pageable_logged:
            return
        self._pageable_logged = True
        self._safe_log(message)

    @property
    def _pin_method(self) -> str:
        methods = {pack.pin_method for pack in self._layer_packs if pack.packed_bytes}
        if not methods:
            return "disabled" if not self._pin_requested else "pageable"
        return next(iter(methods)) if len(methods) == 1 else "mixed"

    def _emit_ready(self) -> None:
        pinned_layers = sum(pack.pinned for pack in self._layer_packs)
        slot_mib = self.slot_bytes / 2**20
        if pinned_layers == self.swapped:
            placement = (
                f"{self.swapped} layers pinned "
                f"({self.pinned_bytes / 2**30:.2f} GiB, {self._pin_method})"
            )
        elif pinned_layers:
            placement = (
                f"{pinned_layers}/{self.swapped} layers pinned "
                f"({self.pinned_bytes / 2**30:.2f} GiB, {self._pin_method}); "
                f"{self.pageable_bytes / 2**30:.2f} GiB pageable"
            )
        else:
            placement = f"{self.swapped} layers pageable ({self.pageable_bytes / 2**30:.2f} GiB)"
        self._safe_log(
            f"Block swap ready: {placement}, {self.slots} slots x {slot_mib:.0f} MiB on {self.device}"
        )

    def _issue_prefetch(self, relative: int) -> None:
        if relative in self._issued:
            return
        copy_stream = self._copy_stream
        if copy_stream is None:
            raise RuntimeError("Block-swap copy stream has been released")
        with torch.cuda.stream(copy_stream):
            for destination, source in self._copy_pairs[relative]:
                destination.copy_(source, non_blocking=True)
            self._ready_events[relative].record(copy_stream)
        self._issued.add(relative)
        self._bytes_h2d += self._layer_packs[relative].packed_bytes

    def _kickoff(self, kind: str) -> None:
        if not self.swapped:
            self._forward_active = True
            self._forward_kind = kind
            self._forwards += 1
            return
        copy_stream = self._copy_stream
        kickoff_event = self._kickoff_event
        if copy_stream is None or kickoff_event is None:
            raise RuntimeError("Block-swap manager has been removed")
        compute_stream = torch.cuda.current_stream(self.device)
        kickoff_event.record(compute_stream)
        copy_stream.wait_event(kickoff_event)
        if self._forward_done_event is not None:
            copy_stream.wait_event(self._forward_done_event)
        self._issued.clear()
        self._bound.clear()
        self._forward_active = True
        self._forward_kind = kind
        self._lazy_expected_next = None
        self._forwards += 1
        for relative in range(min(self.slots, self.swapped)):
            self._issue_prefetch(relative)

    def _root_pre_hook(self, _module: nn.Module, _args: tuple[Any, ...]) -> None:
        if not self._removed:
            self._kickoff("root")

    def _root_post_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if self._removed:
            return
        for relative in tuple(self._bound):
            self._bind_host(relative)
        if self.swapped and self._forward_done_event is not None:
            self._forward_done_event.record(torch.cuda.current_stream(self.device))
        self._forward_active = False
        self._forward_kind = None
        self._lazy_expected_next = None

    def _layer_pre_hook(
        self,
        relative: int,
        _module: nn.Module,
        _args: tuple[Any, ...],
    ) -> None:
        if self._removed:
            raise RuntimeError("Block-swap manager has been removed")
        lazy = False
        if not self._forward_active and (
            self._forward_kind == "lazy_pending" and self._lazy_expected_next == relative
        ):
            self._forward_active = True
            self._forward_kind = "lazy"
            self._lazy_expected_next = None
        elif not self._forward_active:
            lazy = True
            self._kickoff("lazy")

        # With K >= 2 the current layer is always already in flight. This is
        # also the specified direct-layer fallback when the root hook was skipped.
        if relative not in self._issued and (lazy or self.slots > 1):
            self._issue_prefetch(relative)

        compute_stream = torch.cuda.current_stream(self.device)
        barrier = self._barrier_events[relative]
        barrier.record(compute_stream)
        copy_stream = self._copy_stream
        if copy_stream is None:
            raise RuntimeError("Block-swap copy stream has been released")
        copy_stream.wait_event(barrier)

        # K=1 has no lookahead. Its current layer must be issued only after the
        # barrier that protects the slot used by the preceding layer.
        if relative not in self._issued:
            self._issue_prefetch(relative)
        lookahead = relative + self.slots - 1
        if lookahead < self.swapped and lookahead not in self._issued:
            self._issue_prefetch(lookahead)

        compute_stream.wait_event(self._ready_events[relative])
        self._bind_device(relative)

    def _layer_post_hook(
        self,
        relative: int,
        _module: nn.Module,
        _args: tuple[Any, ...],
        _output: Any,
    ) -> None:
        self._bind_host(relative)
        self._layer_loads += 1
        if self._forward_kind == "lazy":
            if self._forward_done_event is not None:
                self._forward_done_event.record(torch.cuda.current_stream(self.device))
            self._forward_active = False
            if relative + 1 < self.swapped:
                self._forward_kind = "lazy_pending"
                self._lazy_expected_next = relative + 1
            else:
                self._forward_kind = None
                self._lazy_expected_next = None

    def _bind_device(self, relative: int) -> None:
        for binding in self._layer_packs[relative].bindings:
            binding.bind_device()
        self._bound.add(relative)

    def _bind_host(self, relative: int) -> None:
        if not 0 <= relative < len(self._layer_packs):
            return
        for binding in self._layer_packs[relative].bindings:
            binding.bind_host()
        self._bound.discard(relative)

    def _unregister_host_ranges(self) -> None:
        if not self._registered_ranges:
            return
        try:
            cudart = torch.cuda.cudart()
        except Exception:
            self._registered_ranges.clear()
            return
        for pointer, _nbytes in self._registered_ranges:
            try:
                cudart.cudaHostUnregister(pointer)
            except Exception:
                pass
        self._registered_ranges.clear()

    def remove(self) -> None:
        """Remove all hooks, release slots, and leave swapped tensors on the CPU."""

        if getattr(self, "_removed", True):
            return
        self._removed = True
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()

        for relative in range(len(self._layer_packs)):
            self._bind_host(relative)
        self._bound.clear()
        self._issued.clear()
        self._forward_active = False
        self._forward_kind = None
        self._lazy_expected_next = None

        root = self._root_ref() if self._root_ref is not None else None
        if root is not None and getattr(root, "_vcap_block_swap_manager", None) is self:
            try:
                delattr(root, "_vcap_block_swap_manager")
            except AttributeError:
                pass
            try:
                delattr(root, "_vcap_block_swap")
            except AttributeError:
                pass
        self._root_ref = None

        for pack in self._layer_packs:
            for binding in pack.bindings:
                binding.device_view = None
        self._copy_pairs.clear()
        self._gpu_slots.clear()
        self._ready_events.clear()
        self._barrier_events.clear()
        self._copy_stream = None
        self._kickoff_event = None
        self._forward_done_event = None
        self._unregister_host_ranges()

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe description of the swap layout."""

        return {
            "resident_layers": int(self.resident),
            "swapped_layers": int(self.swapped),
            "slots": int(self.slots),
            "layer_bytes": int(self.layer_bytes),
            "layer_mib": round(self.layer_bytes / 2**20, 1),
            "slot_bytes": int(self.slot_bytes),
            "slot_mib": round(self.slot_bytes / 2**20, 1),
            "device": str(self.device),
            "pin_method": self._pin_method,
            "pinned_bytes": int(self.pinned_bytes),
            "pinned_gib": round(self.pinned_bytes / 2**30, 2),
            "pageable_bytes": int(self.pageable_bytes),
            "pageable_gib": round(self.pageable_bytes / 2**30, 2),
            "installed": not self._removed,
        }

    def stats(self) -> dict[str, Any]:
        """Return cumulative counters without synchronizing either CUDA stream."""

        return {
            "forwards": int(self._forwards),
            "layer_loads": int(self._layer_loads),
            "bytes_h2d": int(self._bytes_h2d),
            "h2d_gib": round(self._bytes_h2d / 2**30, 2),
            "pinned_bytes": int(self.pinned_bytes),
            "pageable_bytes": int(self.pageable_bytes),
        }

    def reset_stats(self) -> None:
        self._forwards = 0
        self._layer_loads = 0
        self._bytes_h2d = 0

    def __del__(self) -> None:
        try:
            self.remove()
        except Exception:
            pass


__all__ = ["BlockSwapManager"]
