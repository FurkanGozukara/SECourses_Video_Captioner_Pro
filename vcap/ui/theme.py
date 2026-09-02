"""Theme, product CSS, and browser bootstrap scripts for the Gradio shell.

The look of the app comes from a stock Gradio theme. Only three things live in
this module's stylesheet, because Gradio genuinely cannot express them:

1. app-level layout the theme has no concept of (page width, the header band,
   rows that align a button against a labelled input),
2. the markup this app renders itself -- input tiles, progress and VRAM meters,
   find/replace chips -- which no Gradio component styles for us,
3. the multi-hue action-button palette, since a theme ships exactly one primary,
   one secondary, and one stop button.

Every colour below resolves to a Gradio theme variable, so light and dark stay
correct from a single rule instead of a duplicated palette.
"""

from __future__ import annotations

import gradio as gr


# (deep, mid, bright) gradient stops per action-button hue.
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
    ``.vc-btn`` in the base stylesheet; a hue class only supplies the colours.
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

/* ---------------------------------------------------------------- shell -- */
/* The container is also un-clipped so the tab bar below can stick; Gradio's
   own overflow:hidden would otherwise trap it in a non-scrolling box. */
.gradio-container {
  max-width: 1880px !important;
  margin-inline: auto !important;
  overflow: visible !important;
}

/* A caption run is four to five screens tall, so the tab bar follows the page
   and every tab stays one click away from wherever the user has scrolled. */
#vc-main-tabs > .tab-wrapper {
  position: sticky;
  top: 0;
  /* Above the z-index 40 Gradio puts on every block title, below the 1000-plus
     layers it reserves for dropdown menus and modals. */
  z-index: 60;
  padding-top: var(--spacing-lg);
  background: var(--body-background-fill);
}

/* --------------------------------------------------------------- header -- */
.vc-header {
  align-items: center !important;
  margin-bottom: var(--spacing-xxl) !important;
  padding: var(--spacing-xxl) calc(1.5 * var(--spacing-xxl)) !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: var(--container-radius) !important;
  box-shadow: var(--shadow-drop-lg) !important;
  background:
    linear-gradient(104deg, color-mix(in srgb, var(--primary-500) 18%, transparent) 0%, transparent 58%),
    var(--background-fill-secondary) !important;
}
.vc-header h1 { margin: 0 !important; font-size: var(--text-xxl) !important; line-height: 1.15 !important; }
.vc-header p { margin: var(--spacing-md) 0 0 !important; color: var(--body-text-color-subdued) !important; }
.vc-header-meta { text-align: right; color: var(--body-text-color-subdued); font-size: var(--text-sm); line-height: 1.5; }

/* ----------------------------------------------------------- preset bar -- */
.vc-preset-bar {
  gap: var(--spacing-lg) !important;
  margin-bottom: var(--spacing-xxl) !important;
  padding: var(--spacing-lg) var(--spacing-xl) !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: var(--container-radius) !important;
  box-shadow: var(--shadow-drop-lg) !important;
  background: var(--background-fill-secondary) !important;
}

/* --------------------------------------------------------------- layout -- */
/* Puts a bare button on the same baseline as the input it sits beside. */
.vc-compact-row { align-items: flex-end !important; }
/* ...unless the row is nothing but buttons, where every button should match
   the tallest one so a label that wraps to two lines keeps the row square. */
.vc-action-row,
.vc-compact-row:not(:has(> :not(button))) { align-items: stretch !important; }

.vc-card {
  padding: var(--spacing-xl) !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: var(--container-radius) !important;
  box-shadow: var(--shadow-drop-lg) !important;
  background: var(--background-fill-secondary) !important;
}
/* gr.Group repeats elem_classes on a nested wrapper; only the outer one is the card. */
.vc-card .vc-card { padding: 0 !important; border: 0 !important; box-shadow: none !important; background: none !important; }
.vc-section-title { margin: 0 0 var(--spacing-lg) !important; }
.vc-help { color: var(--body-text-color-subdued); font-size: var(--text-sm); }
.vc-status { min-height: 24px; }

/* ---------------------------------------------------------- media panes -- */
.vc-preview video, .vc-preview img { max-height: 390px !important; object-fit: contain !important; }
.vc-preview audio { max-height: 150px !important; }
.vc-result-panel textarea, .vc-scroll-result { max-height: 410px !important; overflow: auto !important; }
.vc-editor-gallery img { aspect-ratio: 16 / 10 !important; object-fit: cover !important; }
.vc-log textarea, .vc-mono textarea {
  font-family: var(--font-mono) !important;
  font-size: var(--text-sm) !important;
  line-height: 1.5 !important;
}

/* ------------------------------------------------- app-rendered widgets -- */
.vc-input-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: var(--spacing-md);
  max-height: 152px;
  overflow: auto;
}
.vc-input-tile {
  display: flex;
  gap: var(--spacing-lg);
  align-items: center;
  min-width: 0;
  padding: var(--spacing-lg) var(--spacing-xl);
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--block-background-fill);
}
.vc-input-icon { flex: 0 0 auto; font-size: 17px; }
.vc-input-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-sm); }

.vc-progress, .vc-meter {
  padding: var(--spacing-xl);
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--block-background-fill);
}
.vc-meter { display: grid; gap: var(--spacing-lg); }
.vc-progress__labels, .vc-meter__label {
  display: flex;
  justify-content: space-between;
  gap: var(--spacing-xxl);
  margin-bottom: var(--spacing-md);
  color: var(--body-text-color-subdued);
  font-size: var(--text-sm);
}
.vc-progress__track, .vc-meter__track {
  height: 9px;
  overflow: hidden;
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--body-text-color) 14%, transparent);
}
.vc-progress__fill, .vc-meter__fill {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--secondary-500), var(--primary-400));
  transition: width .18s ease;
}
/* VRAM, then host RAM, then shared memory: the ramp from accent to neutral
   tells the three bars apart at a glance without inventing a palette. */
.vc-meter__row:nth-child(2) .vc-meter__fill { background: linear-gradient(90deg, var(--primary-700), var(--primary-500)); }
.vc-meter__row:nth-child(3) .vc-meter__fill { background: linear-gradient(90deg, var(--neutral-500), var(--neutral-400)); }

.vcap-replace-chips { display: flex; flex-wrap: wrap; gap: var(--spacing-md); min-height: 28px; align-items: center; }
.vcap-replace-chip {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-sm) var(--spacing-lg);
  border: 1px solid var(--border-color-accent-subdued);
  border-radius: var(--radius-lg);
  background: var(--color-accent-soft);
  font-size: var(--text-sm);
}
.vcap-replace-arrow { color: var(--color-accent); font-weight: 800; }

.vc-ok { color: var(--vc-ok) !important; font-weight: 700; }
.vc-warn { color: var(--vc-warn) !important; font-weight: 700; }
.vc-err { color: var(--vc-err) !important; font-weight: 700; }

.vc-confirm-bar {
  align-items: center !important;
  padding: var(--spacing-lg) var(--spacing-xl) !important;
  border: 1px solid var(--error-border-color) !important;
  border-radius: var(--container-radius) !important;
  background: var(--error-background-fill) !important;
}
.vc-confirm-bar p { margin: 0 !important; color: var(--error-text-color); font-weight: 700; }

/* ------------------------------------------------------ action buttons --- */
/* The hue classes below supply --vc-h1/2/3; everything else is declared once. */
.vc-btn, .vc-btn.vc-btn {
  min-height: 38px;
  border: 1px solid rgb(var(--vc-hue-lt) / .74) !important;
  border-radius: var(--radius-lg) !important;
  background: linear-gradient(135deg, var(--vc-h1) 0%, var(--vc-h2) 55%, var(--vc-h3) 100%) !important;
  box-shadow: 0 6px 16px rgb(var(--vc-hue) / .26), inset 0 1px 0 rgb(255 255 255 / .18) !important;
  color: #f8fafc !important;
  font-weight: 600 !important;
  text-shadow: 0 1px 2px rgb(2 6 23 / .45);
  transition: transform .13s ease, filter .13s ease, box-shadow .13s ease !important;
}
.vc-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  filter: brightness(1.06);
  border-color: rgb(var(--vc-hue-lt) / 1) !important;
  box-shadow: 0 10px 22px rgb(var(--vc-hue) / .40), inset 0 1px 0 rgb(255 255 255 / .24) !important;
}
.vc-btn:active:not(:disabled) { transform: translateY(0); }
.vc-btn:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
.vc-btn:disabled { filter: grayscale(.5) opacity(.62) !important; transform: none !important; box-shadow: none !important; }
body:not(.dark) .vc-btn, body:not(.dark) .vc-btn.vc-btn {
  border-color: var(--vc-h2) !important;
  box-shadow: 0 5px 14px rgb(var(--vc-hue) / .22), inset 0 1px 0 rgb(255 255 255 / .22) !important;
}

/* ------------------------------------------------------------- polish ---- */
/* Thin, theme-coloured scrollbars: the app has many scroll panes and the OS
   default renders as a bright slab in dark mode. */
* { scrollbar-width: thin; scrollbar-color: var(--border-color-primary) transparent; }

@media (max-width: 760px) {
  .vc-header-meta { text-align: left; }
  .vc-input-list { grid-template-columns: 1fr 1fr; }
}
"""


def build_theme() -> gr.themes.Base:
    """Return the stock Gradio Ocean theme, tuned for a dense desktop tool.

    Only constructor arguments and design tokens are used here -- no CSS.

    * ``blue``/``cyan``/``slate`` keeps the chrome cool and neutral, which the
      several dozen coloured action buttons need in order to read as the accent.
    * Ocean ships pill-shaped ``radius_xxl`` corners and wide spacing; both are
      pulled in one notch so several hundred controls fit on a screen.
    * The fonts are the ones Gradio bundles, so a page load makes no request to
      Google Fonts -- this app is expected to run without internet access.
    """

    return gr.themes.Ocean(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_md,
        text_size=gr.themes.sizes.text_md,
    ).set(
        # Accordion and tab headings carry the section hierarchy of this app,
        # so they are weighted rather than left at the theme's body weight.
        section_header_text_weight="600",
        block_title_text_weight="500",
        # Ocean's light palette leaves labels and help text at neutral_400/500,
        # which is under 3:1 against white. Both are darkened for light mode
        # only; dark mode already has the contrast.
        body_text_color_subdued="*neutral_600",
        body_text_color_subdued_dark="*neutral_400",
        block_title_text_color="*neutral_700",
        block_title_text_color_dark="*neutral_200",
        # Every gr.Button size (sm/md/lg) renders with the same metrics so the
        # whole UI shares one button font size and height.
        button_small_text_size="*text_md",
        button_medium_text_size="*text_md",
        button_large_text_size="*text_md",
        button_small_padding="*spacing_lg calc(2 * *spacing_lg)",
        button_medium_padding="*spacing_lg calc(2 * *spacing_lg)",
        button_large_padding="*spacing_lg calc(2 * *spacing_lg)",
    )


def build_css() -> str:
    """Return all static application CSS including the action-button palette."""

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


# Runs entirely in the browser: Gradio's accordion header is a plain button, so
# opening or closing every section costs no server round-trip.
TOGGLE_ACCORDIONS_JS = r"""
() => {
  if (window.__vcapToggleAccordions) window.__vcapToggleAccordions();
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
    if (label.includes('caption')) return 'caption';
    if (label.includes('processing pipeline')) return 'caption';
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
    "build_css",
    "build_theme",
]
