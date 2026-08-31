"""Theme, product CSS, and browser bootstrap scripts for the Gradio shell."""

from __future__ import annotations

import gradio as gr


_SEC_BTN_HUES: dict[str, tuple[str, str, str]] = {
    "red": ("#991b1b", "#dc2626", "#f87171"),
    "crimson": ("#4c0519", "#be123c", "#fb7185"),
    "rose": ("#9f1239", "#e11d48", "#fda4af"),
    "pink": ("#9d174d", "#db2777", "#f9a8d4"),
    "fuchsia": ("#86198f", "#c026d3", "#e879f9"),
    "purple": ("#6b21a8", "#9333ea", "#c084fc"),
    "violet": ("#5b21b6", "#7c3aed", "#a78bfa"),
    "indigo": ("#3730a3", "#4f46e5", "#818cf8"),
    "blue": ("#1e40af", "#2563eb", "#60a5fa"),
    "sky": ("#075985", "#0284c7", "#38bdf8"),
    "cyan": ("#155e75", "#0891b2", "#22d3ee"),
    "teal": ("#115e59", "#0d9488", "#2dd4bf"),
    "emerald": ("#065f46", "#059669", "#34d399"),
    "green": ("#166534", "#16a34a", "#4ade80"),
    "lime": ("#3f6212", "#65a30d", "#a3e635"),
    "yellow": ("#854d0e", "#ca8a04", "#facc15"),
    "amber": ("#92400e", "#d97706", "#fbbf24"),
    "orange": ("#9a3412", "#ea580c", "#fb923c"),
    "bronze": ("#78350f", "#b45309", "#d6a05a"),
    "slate": ("#334155", "#475569", "#94a3b8"),
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _build_sec_btn_css() -> str:
    """Generate high-specificity dark and light rules for every button hue."""

    hues = dict(_SEC_BTN_HUES)
    hues["gold"] = _SEC_BTN_HUES["yellow"]
    rules: list[str] = []
    for name, (dark, mid, light) in hues.items():
        red, green, blue = _hex_to_rgb(mid)
        light_red, light_green, light_blue = _hex_to_rgb(light)
        rules.append(
            f"""
.vc-btn-{name}, .vc-btn-{name}.vc-btn-{name},
.vc-btn-{name} button, button.vc-btn-{name} {{
  background: linear-gradient(135deg, {dark} 0%, {mid} 55%, {light} 100%) !important;
  border-color: rgba({light_red}, {light_green}, {light_blue}, 0.76) !important;
  color: #f8fafc !important;
  text-shadow: 0 1px 2px rgba(2, 6, 23, 0.46) !important;
  box-shadow: 0 8px 20px rgba({red}, {green}, {blue}, 0.28), inset 0 1px 0 rgba(255,255,255,0.18) !important;
}}
.vc-btn-{name}:hover, .vc-btn-{name} button:hover, button.vc-btn-{name}:hover {{
  border-color: rgba({light_red}, {light_green}, {light_blue}, 0.98) !important;
  box-shadow: 0 11px 25px rgba({red}, {green}, {blue}, 0.40), inset 0 1px 0 rgba(255,255,255,0.24) !important;
}}
body:not(.dark) .vc-btn-{name}, body:not(.dark) .vc-btn-{name}.vc-btn-{name},
body:not(.dark) .vc-btn-{name} button, body:not(.dark) button.vc-btn-{name} {{
  background: linear-gradient(135deg, {dark} 0%, {mid} 64%, {light} 100%) !important;
  border-color: {mid} !important;
  color: #ffffff !important;
  box-shadow: 0 7px 17px rgba({red}, {green}, {blue}, 0.23), inset 0 1px 0 rgba(255,255,255,0.22) !important;
}}"""
        )
    return "\n".join(rules)


_BASE_CSS = r"""
:root {
  --vc-panel: rgba(15, 23, 42, 0.72);
  --vc-panel-strong: #111827;
  --vc-line: rgba(148, 163, 184, 0.20);
  --vc-muted: #94a3b8;
  --vc-ok: #34d399;
  --vc-warn: #fbbf24;
  --vc-err: #fb7185;
  --vc-log-bg: #070b12;
  --vc-log-fg: #cbd5e1;
  --vc-log-border: rgba(56,189,248,0.20);
}

.gradio-container { max-width: 1840px !important; }
.vc-shell { padding-bottom: 28px; }
.vc-header {
  padding: 15px 18px 13px;
  border-bottom: 1px solid rgba(129, 140, 248, 0.25);
  background: linear-gradient(112deg, rgba(30,41,59,0.96), rgba(17,24,39,0.96) 58%, rgba(12,74,110,0.76));
}
.vc-header h1 { font-size: 24px !important; line-height: 1.2 !important; margin: 0 !important; color: #f8fafc !important; letter-spacing: 0 !important; }
.vc-header p { margin: 4px 0 0 !important; color: #cbd5e1 !important; }
.vc-header a { color: #7dd3fc !important; font-weight: 700; }
.vc-header-meta { text-align: right; color: #cbd5e1; font-size: 13px; line-height: 1.45; }

.vc-preset-bar {
  padding: 10px 12px;
  border-bottom: 1px solid var(--vc-line);
  background: rgba(15, 23, 42, 0.34);
}
.vc-compact-row { gap: 8px !important; align-items: end !important; }
.vc-card, .vc-panel {
  border: 1px solid var(--vc-line) !important;
  border-radius: 8px !important;
  background: var(--vc-panel) !important;
  box-shadow: 0 10px 28px rgba(2, 6, 23, 0.14) !important;
}
.vc-card { padding: 10px !important; }
.vc-section-title { margin: 0 0 8px !important; font-weight: 750; }

#vc-input-tabs .tab-container.visually-hidden {
  display: none !important;
  visibility: hidden !important;
  position: absolute !important;
  height: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
#vc-input-tabs .tab-container:not(.visually-hidden) {
  display: flex !important;
  flex-wrap: nowrap !important;
  overflow-x: auto !important;
}
#vc-input-tabs .tab-container:not(.visually-hidden) > button {
  flex: 1 1 0 !important;
  min-width: 108px !important;
  white-space: nowrap !important;
}

#vc-main-tabs > .tab-wrapper {
  border: 1px solid rgba(99,102,241,0.22) !important;
  border-radius: 8px !important;
  background: rgba(15,23,42,0.38) !important;
  padding: 5px !important;
}
#vc-main-tabs > .tab-wrapper > .tab-container {
  background: linear-gradient(130deg, rgba(30,41,59,.42), rgba(15,23,42,.26)) !important;
}
#vc-main-tabs .tab-container button { font-weight: 700 !important; }
#vc-main-tabs .tab-container button.selected {
  color: #eef2ff !important;
  border-color: rgba(56,189,248,0.66) !important;
  background: linear-gradient(145deg, #4338ca, #0f766e) !important;
}

.vc-input-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 6px;
  max-height: 152px;
  overflow: auto;
}
.vc-input-tile {
  display: flex;
  gap: 7px;
  align-items: center;
  min-width: 0;
  padding: 7px 8px;
  border: 1px solid var(--vc-line);
  border-radius: 6px;
  background: rgba(30,41,59,0.48);
}
.vc-input-icon { flex: 0 0 auto; font-size: 17px; }
.vc-input-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
.vc-preview video, .vc-preview img { max-height: 390px !important; object-fit: contain !important; }
.vc-preview audio { max-height: 150px !important; }

.vc-result-panel textarea, .vc-scroll-result { max-height: 410px !important; overflow: auto !important; }
.vc-log textarea {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
  color: var(--vc-log-fg) !important;
  background: var(--vc-log-bg) !important;
  border-color: var(--vc-log-border) !important;
}
.vc-mono textarea { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace !important; }

.vc-progress { padding: 10px 12px; border: 1px solid var(--vc-line); border-radius: 7px; background: rgba(15,23,42,0.56); }
.vc-progress__labels { display: flex; justify-content: space-between; gap: 12px; font-size: 12px; color: var(--vc-muted); margin-bottom: 6px; }
.vc-progress__track, .vc-meter__track { height: 9px; overflow: hidden; border-radius: 4px; background: rgba(100,116,139,0.25); }
.vc-progress__fill, .vc-meter__fill { display: block; height: 100%; background: linear-gradient(90deg, #2563eb, #14b8a6, #22c55e); transition: width .18s ease; }
.vc-meter { display: grid; gap: 8px; padding: 9px 10px; border: 1px solid var(--vc-line); border-radius: 7px; background: rgba(15,23,42,0.46); }
.vc-meter__label { display: flex; justify-content: space-between; gap: 12px; color: var(--vc-muted); font-size: 12px; margin-bottom: 4px; }
.vc-meter__row:first-child .vc-meter__fill { background: linear-gradient(90deg, #2563eb, #06b6d4); }
.vc-meter__row:last-child .vc-meter__fill { background: linear-gradient(90deg, #7c3aed, #ec4899); }

.vc-ok { color: var(--vc-ok) !important; font-weight: 700; }
.vc-warn { color: var(--vc-warn) !important; font-weight: 700; }
.vc-err { color: var(--vc-err) !important; font-weight: 700; }
.vc-status { min-height: 24px; }
.vc-help { color: var(--vc-muted); font-size: 12px; }

.vcap-replace-chips { display: flex; flex-wrap: wrap; gap: 6px; min-height: 28px; align-items: center; }
.vcap-replace-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border: 1px solid rgba(56,189,248,.30); border-radius: 6px; background: rgba(8,145,178,.12); font-size: 12px; }
.vcap-replace-arrow { color: #38bdf8; font-weight: 800; }

.vc-btn, .vc-btn.vc-btn { border-radius: 7px !important; font-weight: 780 !important; letter-spacing: 0 !important; transition: transform .14s ease, filter .14s ease, box-shadow .14s ease !important; }
.vc-btn:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.04); }
.vc-btn:focus-visible { outline: 2px solid #bae6fd !important; outline-offset: 2px !important; }
.vc-btn:disabled { filter: grayscale(.46) opacity(.62) !important; transform: none !important; }

body:not(.dark) {
  --vc-panel: rgba(255,255,255,0.94);
  --vc-panel-strong: #ffffff;
  --vc-line: rgba(71,85,105,0.20);
  --vc-muted: #475569;
  --vc-ok: #047857;
  --vc-warn: #a16207;
  --vc-err: #be123c;
  --vc-log-bg: #f8fafc;
  --vc-log-fg: #0f172a;
  --vc-log-border: rgba(2,132,199,.25);
}
body:not(.dark) .vc-header {
  background: linear-gradient(112deg, #eef2ff, #f8fafc 55%, #ecfeff) !important;
  border-color: rgba(79,70,229,.22) !important;
}
body:not(.dark) .gradio-container .contain .vc-header.vc-header h1 { color: #0f172a !important; }
body:not(.dark) .gradio-container .contain .vc-header.vc-header p,
body:not(.dark) .gradio-container .contain .vc-header.vc-header .vc-header-meta { color: #334155 !important; }
body:not(.dark) .gradio-container .contain .vc-header.vc-header a { color: #0369a1 !important; }
body:not(.dark) .vc-preset-bar { background: rgba(241,245,249,.86) !important; }
body:not(.dark) .vc-card, body:not(.dark) .vc-panel { background: rgba(255,255,255,.96) !important; box-shadow: 0 8px 22px rgba(15,23,42,.07) !important; }
body:not(.dark) #vc-main-tabs > .tab-wrapper,
body:not(.dark) #vc-main-tabs > .tab-wrapper > .tab-container { background: linear-gradient(130deg, rgba(224,231,255,.78), rgba(236,254,255,.72)) !important; border-color: rgba(37,99,235,.20) !important; }
body:not(.dark) #vc-main-tabs .tab-container button { color: #0f172a !important; background: rgba(255,255,255,.84) !important; border-color: rgba(100,116,139,.28) !important; }
body:not(.dark) #vc-main-tabs .tab-container button.selected { color: #ffffff !important; background: linear-gradient(145deg, #4f46e5, #0d9488) !important; border-color: #0d9488 !important; }
body:not(.dark) .vc-input-tile { background: #f8fafc !important; }
body:not(.dark) .vc-progress, body:not(.dark) .vc-meter { background: #f8fafc !important; }
body:not(.dark) .gradio-container .vc-log textarea { color: #0f172a !important; background: #f8fafc !important; border-color: rgba(2,132,199,.25) !important; }
body:not(.dark) input[type=checkbox]:not(:checked),
body:not(.dark) input[type=radio]:not(:checked) { background: #ffffff !important; border: 1.5px solid #64748b !important; box-shadow: inset 0 0 0 1px rgba(100,116,139,.08) !important; }
body:not(.dark) .vcap-replace-chip { background: rgba(8,145,178,.08) !important; border-color: rgba(3,105,161,.34) !important; }
body:not(.dark) .vcap-replace-arrow { color: #0369a1 !important; }
body:not(.dark) .vc-btn:focus-visible { outline-color: #0369a1 !important; }

@media (max-width: 760px) {
  .vc-header-meta { text-align: left; }
  .vc-input-list { grid-template-columns: 1fr 1fr; }
  .vc-action-row { grid-template-columns: 1fr 1fr !important; }
}
"""


def build_theme() -> gr.themes.Soft:
    """Return the product theme; Gradio 6 receives it at ``launch()``."""

    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="sky",
        neutral_hue="slate",
        spacing_size="sm",
        radius_size="md",
        text_size="md",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#f8fafc",
        body_background_fill_dark="#080d16",
        block_background_fill="#ffffff",
        block_background_fill_dark="#0f172a",
        block_border_color="#dbe3ee",
        block_border_color_dark="#263449",
        body_text_color="#172033",
        body_text_color_dark="#e2e8f0",
        button_primary_background_fill="#4f46e5",
        button_primary_background_fill_dark="#6366f1",
        button_primary_background_fill_hover="#4338ca",
        button_primary_background_fill_hover_dark="#818cf8",
        loader_color="#0ea5e9",
    )


def build_css() -> str:
    """Return all static application CSS including generated button palettes."""

    return _BASE_CSS + "\n" + _build_sec_btn_css()


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

  function activeMainTab() {
    const root = document.getElementById('vc-main-tabs');
    if (!root) return null;
    const bar = root.querySelector(':scope > .tab-wrapper > .tab-container:not(.visually-hidden)') ||
      root.querySelector(':scope > .tab-wrapper .tab-container:not(.visually-hidden)');
    const selected = bar && bar.querySelector('[role="tab"][aria-selected="true"], button.selected');
    const label = (selected && selected.textContent ? selected.textContent : '').toLowerCase();
    if (label.includes('caption editor')) return 'editor';
    if (label.includes('caption')) return 'caption';
    const panels = root.querySelectorAll('[role="tabpanel"]');
    for (const panel of panels) {
      const style = window.getComputedStyle(panel);
      if (panel.getAttribute('aria-hidden') === 'true' || style.display === 'none' || style.visibility === 'hidden') continue;
      if (panel.querySelector('#hk_ed_save')) return 'editor';
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

  document.addEventListener('keydown', function (event) {
    const target = event.target;
    if (isDropdownSearch(target)) return;
    const tab = activeMainTab();
    const plain = !event.ctrlKey && !event.metaKey && !event.altKey;
    const primary = (event.ctrlKey || event.metaKey) && !event.altKey;

    if (tab === 'caption') {
      if (plain && event.key === 'F9' && !captionJobRunning()) clickHotkey('hk_caption_start', event);
      else if (plain && event.key === 'Escape' && captionJobRunning()) clickHotkey('hk_caption_cancel', event);
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


__all__ = ["HOTKEYS_HEAD", "THEME_CHANGE_JS", "build_css", "build_theme"]
