"""Theme, product CSS, and browser bootstrap scripts for the Gradio shell.

The look of the app is the stock Gradio 6 ``Origin`` theme, used exactly as
shipped and shared with the SECourses IndexTTS app. The stylesheet below is
deliberately small: it covers only what a Gradio theme has no way to express.

1. the multi-hue action buttons -- a theme ships one primary, one secondary,
   and one stop button, and this app has several dozen coloured actions,
2. the markup this app renders itself -- input tiles, progress and VRAM
   meters, find/replace chips, status words -- which no component styles,
3. three small pieces of page furniture: the header rule, the preset strip's
   button alignment, and the inline confirmation bar.

Every colour outside the button gradients resolves to a Gradio theme variable,
so light and dark stay correct from a single rule instead of a duplicated
palette.
"""

from __future__ import annotations

import gradio as gr


# (deep, mid, bright) gradient stops per action-button hue. The twenty hues
# shared with IndexTTS use identical stops so both apps' buttons match.
_SEC_BTN_HUES: dict[str, tuple[str, str, str]] = {
    "emerald": ("#065f46", "#059669", "#34d399"),
    "green": ("#166534", "#16a34a", "#4ade80"),
    "lime": ("#3f6212", "#65a30d", "#a3e635"),
    "teal": ("#115e59", "#0d9488", "#2dd4bf"),
    "cyan": ("#155e75", "#0891b2", "#22d3ee"),
    "sky": ("#075985", "#0284c7", "#38bdf8"),
    "blue": ("#1e40af", "#2563eb", "#60a5fa"),
    "indigo": ("#3730a3", "#4f46e5", "#818cf8"),
    "violet": ("#5b21b6", "#7c3aed", "#a78bfa"),
    "purple": ("#6b21a8", "#9333ea", "#c084fc"),
    "fuchsia": ("#86198f", "#c026d3", "#e879f9"),
    "pink": ("#9d174d", "#db2777", "#f9a8d4"),
    "rose": ("#9f1239", "#e11d48", "#fda4af"),
    "red": ("#991b1b", "#dc2626", "#f87171"),
    "crimson": ("#4c0519", "#9f1239", "#fb7185"),
    "orange": ("#9a3412", "#ea580c", "#fb923c"),
    "amber": ("#92400e", "#d97706", "#fbbf24"),
    "bronze": ("#5c3a21", "#8b5a2b", "#d4a373"),
    "slate": ("#334155", "#475569", "#94a3b8"),
    "gray": ("#3f3f46", "#52525b", "#a1a1aa"),
    # Extra hues this app uses beyond the shared set.
    "yellow": ("#854d0e", "#ca8a04", "#facc15"),
    "aqua": ("#164e63", "#0e7490", "#67e8f9"),
    "mint": ("#064e3b", "#047857", "#6ee7b7"),
    "jade": ("#14532d", "#15803d", "#86efac"),
    "navy": ("#172554", "#1e3a8a", "#93c5fd"),
    "cobalt": ("#1e3a8a", "#1d4ed8", "#7dd3fc"),
    "steel": ("#1f2937", "#4b5563", "#cbd5e1"),
    "plum": ("#4a044e", "#a21caf", "#f0abfc"),
    "berry": ("#500724", "#be185d", "#f9a8d4"),
    "coral": ("#7f1d1d", "#dc4b3e", "#fca5a5"),
    "copper": ("#713f12", "#a16207", "#fde68a"),
    "olive": ("#365314", "#4d7c0f", "#bef264"),
    "maroon": ("#450a0a", "#b91c1c", "#fecaca"),
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _build_sec_btn_css() -> str:
    """Emit one custom-property line per hue.

    The gradient, border, shadow, and hover behaviour are declared once on
    ``button.vc-btn`` in the base stylesheet; a hue class only supplies the
    colours.
    """

    hues = dict(_SEC_BTN_HUES)
    hues["gold"] = _SEC_BTN_HUES["yellow"]
    rules: list[str] = []
    for name, (deep, mid, bright) in sorted(hues.items()):
        red, green, blue = _hex_to_rgb(mid)
        light_red, light_green, light_blue = _hex_to_rgb(bright)
        rules.append(
            f".vc-btn-{name}{{--vc-h1:{deep};--vc-h2:{mid};--vc-h3:{bright};"
            f"--vc-hue:{red} {green} {blue};--vc-hue-lt:{light_red} {light_green} {light_blue};}}"
        )
    return "\n".join(rules)


_BASE_CSS = r"""
/* Semantic status colours. Gradio themes have an error palette but no success
   or warning pair, so these three are defined here and nowhere else. */
:root { --vc-ok: #047857; --vc-warn: #b45309; --vc-err: #be123c; }
.dark { --vc-ok: #34d399; --vc-warn: #fbbf24; --vc-err: #f87171; }
.vc-ok { color: var(--vc-ok); font-weight: 700; }
.vc-warn { color: var(--vc-warn); font-weight: 700; }
.vc-err { color: var(--vc-err); font-weight: 700; }

/* ------------------------------------------------------- page furniture -- */
.vc-header { padding-bottom: var(--size-3); border-bottom: 1px solid var(--border-color-primary); }
.vc-header h1 { margin: 0 !important; line-height: 1.2; }
.vc-header p { margin: var(--size-1) 0 0 !important; color: var(--body-text-color-subdued); }
/* Line the preset buttons up with the fields they act on rather than with the
   whole labelled block. */
.vc-preset-bar { align-items: flex-end; gap: var(--size-2); }
.row.vc-preset-bar > button.vc-btn { align-self: flex-end !important; margin-bottom: 12px; }
.vc-help { color: var(--body-text-color-subdued); font-size: var(--text-sm); }
.vc-status { min-height: var(--size-6); }
.vc-mono textarea { font-family: var(--font-mono) !important; font-size: var(--text-sm) !important; line-height: 1.5 !important; }
.vc-confirm-bar {
  padding: var(--size-2) var(--size-3);
  border: 1px solid var(--error-border-color);
  border-radius: var(--block-radius);
  background: var(--error-background-fill);
}
.vc-confirm-bar > * { align-self: center; }
.vc-confirm-bar p { margin: 0 !important; color: var(--error-text-color); font-weight: 600; }

/* ------------------------------------------------- app-rendered widgets -- */
.vc-input-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: var(--size-2);
  max-height: 152px;
  overflow: auto;
}
.vc-input-tile {
  display: flex;
  gap: var(--size-2);
  align-items: center;
  min-width: 0;
  padding: var(--size-2) var(--size-3);
  border: 1px solid var(--block-border-color);
  border-radius: var(--radius-lg);
  background: var(--background-fill-secondary);
}
.vc-input-icon { flex: 0 0 auto; font-size: 17px; }
.vc-input-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-sm); }

.vc-progress, .vc-meter {
  padding: var(--size-3);
  border: 1px solid var(--block-border-color);
  border-radius: var(--radius-lg);
  background: var(--background-fill-secondary);
}
.vc-meter { display: grid; gap: var(--size-2); }
.vc-progress__labels, .vc-meter__label {
  display: flex;
  justify-content: space-between;
  gap: var(--size-4);
  margin-bottom: var(--size-1);
  color: var(--body-text-color-subdued);
  font-size: var(--text-sm);
}
.vc-progress__track, .vc-meter__track {
  height: 8px;
  overflow: hidden;
  border-radius: var(--radius-full);
  background: var(--border-color-primary);
}
.vc-progress__fill, .vc-meter__fill {
  display: block;
  height: 100%;
  border-radius: var(--radius-full);
  /* The theme's own stat gradient, the one Gradio uses for its Label bars. */
  background: var(--stat-background-fill);
  transition: width 250ms ease;
}
/* VRAM, then host RAM, then shared memory: accent, secondary, neutral tells
   the three bars apart at a glance without inventing a palette. */
.vc-meter__row:nth-child(2) .vc-meter__fill { background: linear-gradient(to right, var(--secondary-400), var(--secondary-600)); }
.vc-meter__row:nth-child(3) .vc-meter__fill { background: linear-gradient(to right, var(--neutral-400), var(--neutral-500)); }

.vcap-replace-chips { display: flex; flex-wrap: wrap; gap: var(--size-2); min-height: var(--size-7); align-items: center; }
.vcap-replace-chip {
  display: inline-flex;
  align-items: center;
  gap: var(--size-2);
  padding: var(--size-1) var(--size-2);
  border: 1px solid var(--border-color-accent-subdued);
  border-radius: var(--radius-lg);
  background: var(--color-accent-soft);
  font-size: var(--text-sm);
}
.vcap-replace-arrow { color: var(--color-accent); font-weight: 800; }

/* ------------------------------------------------------ action buttons --- */
/* Every button is the same height, weight and type size so rows of controls
   share a baseline; only the hue changes, and a hue class further down only
   supplies the colour stops (--vc-h1/2/3, --vc-hue, --vc-hue-lt). The rules
   are the ones the IndexTTS app uses, so both apps' buttons match. */
button.vc-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--size-2);
  min-height: 44px;
  padding: var(--size-2) var(--size-4) !important;
  border-width: 1px !important;
  border-style: solid !important;
  border-color: rgb(var(--vc-hue-lt) / .72) !important;
  border-radius: var(--radius-lg) !important;
  background: linear-gradient(135deg, var(--vc-h1) 0%, var(--vc-h2) 55%, var(--vc-h3) 100%) !important;
  box-shadow: 0 8px 20px rgb(var(--vc-hue) / .30), inset 0 1px 0 rgb(255 255 255 / .20) !important;
  color: #f8fafc !important;
  font-size: var(--text-md) !important;
  font-weight: 650 !important;
  line-height: 1.25 !important;
  text-align: center;
  text-shadow: 0 1px 2px rgb(2 6 23 / .45) !important;
  transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease;
}
button.vc-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  filter: brightness(1.05);
  border-color: rgb(var(--vc-hue-lt) / .98) !important;
  box-shadow: 0 12px 26px rgb(var(--vc-hue) / .44), inset 0 1px 0 rgb(255 255 255 / .28) !important;
}
button.vc-btn:active:not(:disabled) { transform: translateY(1px); filter: brightness(.96); }
button.vc-btn:focus-visible { outline: 2px solid #bae6fd; outline-offset: 2px; }
button.vc-btn:disabled { filter: grayscale(.45) opacity(.62); transform: none; box-shadow: none !important; }
/* Inside a row a button would otherwise stretch to the tallest neighbour, which
   is the one place the uniform height breaks down. */
.row > button.vc-btn { align-self: center; }
body:not(.dark) button.vc-btn {
  background: linear-gradient(135deg, var(--vc-h1) 0%, var(--vc-h2) 66%, var(--vc-h3) 100%) !important;
  border-color: var(--vc-h2) !important;
  box-shadow: 0 7px 17px rgb(var(--vc-hue) / .26), inset 0 1px 0 rgb(255 255 255 / .26) !important;
}
body:not(.dark) button.vc-btn:hover:not(:disabled) {
  box-shadow: 0 11px 24px rgb(var(--vc-hue) / .38), inset 0 1px 0 rgb(255 255 255 / .32) !important;
}
"""


def build_theme() -> gr.themes.Base:
    """Return the stock Gradio 6 ``Origin`` theme, used exactly as shipped.

    Every colour, radius, shadow and font in the interface comes from this
    theme; no design token is overridden, so the app looks the same as the
    other SECourses Gradio apps. Origin bundles its fonts (Source Sans Pro and
    IBM Plex Mono), so a page load makes no request to Google Fonts and the app
    renders identically without internet access.
    """

    return gr.themes.Origin()


def build_css() -> str:
    """Return all static application CSS including the action-button palette."""

    return _BASE_CSS + "\n" + _build_sec_btn_css() + "\n"


THEME_CHANGE_JS = r"""
(mode) => {
  const selected = ['dark', 'light', 'system'].includes(mode) ? mode : 'dark';
  localStorage.setItem('secourses_theme_mode', selected);
  if (typeof window.__secoursesApplyThemeMode === 'function') {
    window.__secoursesApplyThemeMode(selected);
  } else {
    const effective = selected === 'system'
      ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : selected;
    if (document.body) document.body.classList.toggle('dark', effective === 'dark');
    const url = new URL(window.location.href);
    url.searchParams.set('__theme', effective);
    window.history.replaceState({}, '', url.toString());
  }
  return [];
}
"""


# Runs entirely in the browser: Gradio's accordion header is a plain button, so
# opening or closing every section costs no server round-trip.
TOGGLE_ACCORDIONS_JS = r"""
() => {
  if (window.__vcapToggleAccordions) window.__vcapToggleAccordions();
  return [];
}
"""


# Header button: flips between explicit dark and light from whatever is on
# screen (System resolves to one of the two first), stores the choice, applies
# it in the browser, and hands the new mode back to the Global Settings radio
# so both controls always agree. No server round-trip, so it works mid-job.
TOGGLE_THEME_JS = r"""
() => {
  const mode = document.body.classList.contains('dark') ? 'light' : 'dark';
  localStorage.setItem('secourses_theme_mode', mode);
  if (typeof window.__secoursesApplyThemeMode === 'function') {
    window.__secoursesApplyThemeMode(mode);
  } else {
    document.documentElement.classList.toggle('dark', mode === 'dark');
    document.body.classList.toggle('dark', mode === 'dark');
  }
  return [mode];
}
"""


HOTKEYS_HEAD = r"""
<script>
(function () {
  if (window.__secoursesVcapInstalled) return;
  window.__secoursesVcapInstalled = true;

  const themeMedia = window.matchMedia('(prefers-color-scheme: dark)');
  function normalizeThemeMode(value) {
    return value === 'light' || value === 'system' || value === 'dark' ? value : 'dark';
  }
  function effectiveThemeMode(mode) {
    return mode === 'system' ? (themeMedia.matches ? 'dark' : 'light') : mode;
  }
  function applyThemeMode(mode) {
    const effective = effectiveThemeMode(normalizeThemeMode(mode));
    document.documentElement.classList.toggle('dark', effective === 'dark');
    if (document.body) document.body.classList.toggle('dark', effective === 'dark');
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== effective) {
      url.searchParams.set('__theme', effective);
      window.history.replaceState({}, '', url.toString());
    }
    return effective;
  }
  window.__secoursesApplyThemeMode = applyThemeMode;

  let completionAudioContext = null;
  function prepareCompletionSound() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return null;
    try {
      if (!completionAudioContext) completionAudioContext = new AudioContext();
    } catch (_error) {
      return null;
    }
    if (completionAudioContext.state === 'suspended') {
      completionAudioContext.resume().catch(function () {});
    }
    return completionAudioContext;
  }
  function playCompletionSound() {
    const context = prepareCompletionSound();
    if (!context) return;
    const now = context.currentTime + 0.02;
    [[523.25, 0], [659.25, 0.16]].forEach(function (tone) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const start = now + tone[1];
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(tone[0], start);
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.12, start + 0.025);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.22);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(start);
      oscillator.stop(start + 0.23);
    });
  }
  window.__vcapPrepareCompletionSound = prepareCompletionSound;
  const primeCompletionSound = function () { prepareCompletionSound(); };
  document.addEventListener('pointerdown', primeCompletionSound, {once: true, passive: true});
  document.addEventListener('keydown', primeCompletionSound, {once: true});
  window.__vcapNotifyJobDone = function (message, desktopEnabled, soundEnabled) {
    const text = String(message || 'Caption job finished');
    if (desktopEnabled && 'Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification('SECourses Video Captioner Pro', {body: text});
      } catch (_error) {}
    }
    if (soundEnabled) playCompletionSound();
  };

  function applyStoredTheme() {
    let stored = localStorage.getItem('secourses_theme_mode');
    if (stored !== 'dark' && stored !== 'light' && stored !== 'system') {
      stored = 'dark';
      localStorage.setItem('secourses_theme_mode', 'dark');
    }
    applyThemeMode(stored);
  }
  applyStoredTheme();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyStoredTheme, {once: true});
  } else {
    applyStoredTheme();
  }
  const onSystemThemeChange = function () {
    if (localStorage.getItem('secourses_theme_mode') === 'system') applyThemeMode('system');
  };
  if (typeof themeMedia.addEventListener === 'function') themeMedia.addEventListener('change', onSystemThemeChange);
  else if (typeof themeMedia.addListener === 'function') themeMedia.addListener(onSystemThemeChange);

  // Accordion headers of the tab the user is looking at. Hidden tabs are left
  // alone so the button always does what the visible page suggests.
  function visibleAccordionHeaders() {
    return Array.prototype.filter.call(
      document.querySelectorAll('.gr-accordion button.label-wrap'),
      function (header) { return header.offsetParent !== null; }
    );
  }

  window.__vcapToggleAccordions = function () {
    const first = visibleAccordionHeaders();
    if (!first.length) return;
    // Any section still closed means the user wants everything open.
    const shouldOpen = first.some(function (header) { return !header.classList.contains('open'); });
    const button = document.getElementById('vc_toggle_accordions');
    if (button) button.setAttribute('aria-expanded', String(shouldOpen));

    // Opening a section reveals the accordions nested inside it, and Svelte
    // only shows them on the next frame, so opening runs until nothing new
    // turns up. Closing needs one pass: a closed parent hides its children.
    let pass = 0;
    (function step() {
      let changed = false;
      visibleAccordionHeaders().forEach(function (header) {
        if (header.classList.contains('open') !== shouldOpen) {
          header.click();
          changed = true;
        }
      });
      pass += 1;
      if (shouldOpen && changed && pass < 6) setTimeout(step, 60);
    })();
  };

  function activeMainTab() {
    const root = document.getElementById('vc-main-tabs');
    if (!root) return null;
    const bar = root.querySelector(':scope > .tab-wrapper > .tab-container:not(.visually-hidden)') ||
      root.querySelector(':scope > .tab-wrapper .tab-container:not(.visually-hidden)');
    const selected = bar && bar.querySelector('[role="tab"][aria-selected="true"], button.selected');
    const label = (selected && selected.textContent ? selected.textContent : '').toLowerCase();
    if (label.includes('caption editor')) return 'editor';
    if (label.includes('transcribe')) return 'transcribe';
    if (label.includes('caption')) return 'caption';
    if (label.includes('processing pipeline')) return 'caption';
    const panels = root.querySelectorAll('[role="tabpanel"]');
    for (const panel of panels) {
      const style = window.getComputedStyle(panel);
      if (panel.getAttribute('aria-hidden') === 'true' || style.display === 'none' || style.visibility === 'hidden') continue;
      if (panel.querySelector('#hk_ed_save')) return 'editor';
      if (panel.querySelector('#hk_transcribe_start')) return 'transcribe';
      if (panel.querySelector('#hk_caption_start')) return 'caption';
    }
    return null;
  }

  function isDropdownSearch(target) {
    if (!target || typeof target.closest !== 'function') return false;
    const input = target.closest('input');
    if (!input) return false;
    return input.getAttribute('role') === 'combobox' ||
      input.getAttribute('aria-autocomplete') !== null ||
      Boolean(input.closest('.dropdown, [data-testid="dropdown"], [role="listbox"]'));
  }

  function isTextEntry(target) {
    if (!target) return false;
    const tag = (target.tagName || '').toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || Boolean(target.isContentEditable);
  }

  function clickHotkey(id, event) {
    const button = document.getElementById(id);
    if (!button || button.disabled || button.getAttribute('aria-disabled') === 'true') return false;
    event.preventDefault();
    button.click();
    return true;
  }

  function captionJobRunning() {
    const button = document.getElementById('vc_caption_cancel');
    return Boolean(button && !button.disabled && button.getAttribute('aria-disabled') !== 'true');
  }

  function transcribeJobRunning() {
    const button = document.getElementById('vc_transcribe_cancel');
    return Boolean(button && !button.disabled && button.getAttribute('aria-disabled') !== 'true');
  }

  document.addEventListener('keydown', function (event) {
    const target = event.target;
    const chatComposer = target && target.closest ? target.closest('#vc_chat_message textarea') : null;
    if (chatComposer && event.key === 'Enter' && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      const send = document.getElementById('vc_chat_send');
      if (send && !send.disabled) send.click();
      return;
    }
    if (isDropdownSearch(target)) return;
    const tab = activeMainTab();
    const plain = !event.ctrlKey && !event.metaKey && !event.altKey;
    const primary = (event.ctrlKey || event.metaKey) && !event.altKey;

    if (plain && !isTextEntry(target) && event.key === 'F4') {
      event.preventDefault();
      window.__vcapToggleAccordions();
      return;
    }
    if (tab === 'caption') {
      if (plain && event.key === 'F9' && !captionJobRunning()) clickHotkey('hk_caption_start', event);
      else if (plain && event.key === 'Escape' && captionJobRunning()) clickHotkey('hk_caption_cancel', event);
      return;
    }
    if (tab === 'transcribe') {
      if (plain && event.key === 'F9' && !transcribeJobRunning()) clickHotkey('hk_transcribe_start', event);
      else if (plain && event.key === 'Escape' && transcribeJobRunning()) clickHotkey('hk_transcribe_cancel', event);
      return;
    }
    if (tab === 'editor') {
      const key = event.key.toLowerCase();
      if (primary && key === 's') clickHotkey('hk_ed_save', event);
      else if (primary && event.key === 'Enter') clickHotkey('hk_ed_approve', event);
      else if (primary && event.key === 'Delete') clickHotkey('hk_ed_reject', event);
      else if (plain && !isTextEntry(target) && event.key === 'ArrowLeft') clickHotkey('hk_ed_prev', event);
      else if (plain && !isTextEntry(target) && event.key === 'ArrowRight') clickHotkey('hk_ed_next', event);
    }
  }, false);
})();
</script>
"""


__all__ = [
    "HOTKEYS_HEAD",
    "THEME_CHANGE_JS",
    "TOGGLE_ACCORDIONS_JS",
    "TOGGLE_THEME_JS",
    "build_css",
    "build_theme",
]
