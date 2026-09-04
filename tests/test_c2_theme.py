from __future__ import annotations

import re

import gradio as gr

from vcap.ui.theme import (
    HOTKEYS_HEAD,
    THEME_CHANGE_JS,
    TOGGLE_ACCORDIONS_JS,
    TOGGLE_THEME_JS,
    build_css,
    build_theme,
)


def test_head_script_has_dark_default_and_three_mode_theme_logic() -> None:
    assert "secourses_theme_mode" in HOTKEYS_HEAD
    assert "localStorage.setItem('secourses_theme_mode', 'dark')" in HOTKEYS_HEAD
    assert "stored !== 'dark' && stored !== 'light' && stored !== 'system'" in HOTKEYS_HEAD
    assert "matchMedia('(prefers-color-scheme: dark)')" in HOTKEYS_HEAD
    assert "addEventListener('change', onSystemThemeChange)" in HOTKEYS_HEAD
    assert "url.searchParams.set('__theme', effective)" in HOTKEYS_HEAD
    assert "document.body.classList.toggle('dark', effective === 'dark')" in HOTKEYS_HEAD
    assert "['dark', 'light', 'system']" in THEME_CHANGE_JS
    assert "window.__vcapNotifyJobDone" in HOTKEYS_HEAD
    assert "SECourses Video Captioner Pro" in HOTKEYS_HEAD
    assert "new Notification" in HOTKEYS_HEAD
    assert "createOscillator" in HOTKEYS_HEAD


def test_hotkeys_are_scoped_and_editor_save_allows_textarea_focus() -> None:
    assert "activeMainTab()" in HOTKEYS_HEAD
    assert "captionJobRunning()" in HOTKEYS_HEAD
    for element_id in (
        "hk_caption_start",
        "hk_caption_cancel",
        "hk_ed_prev",
        "hk_ed_next",
        "hk_ed_save",
        "hk_ed_approve",
        "hk_ed_reject",
    ):
        assert element_id in HOTKEYS_HEAD
    assert "isDropdownSearch(target)" in HOTKEYS_HEAD
    assert "primary && key === 's'" in HOTKEYS_HEAD


def test_open_close_all_accordions_runs_in_the_browser_only() -> None:
    assert "window.__vcapToggleAccordions" in TOGGLE_ACCORDIONS_JS
    assert "window.__vcapToggleAccordions = function" in HOTKEYS_HEAD
    # The visible tab decides the target state, and opening repeats so nested
    # accordions revealed by the first pass also open.
    assert "visibleAccordionHeaders" in HOTKEYS_HEAD
    assert "header.offsetParent !== null" in HOTKEYS_HEAD
    assert "shouldOpen && changed && pass < 6" in HOTKEYS_HEAD
    # F4 reaches the same helper from the keyboard.
    assert "event.key === 'F4'" in HOTKEYS_HEAD


def test_header_theme_button_flips_the_mode_and_mirrors_the_settings_radio() -> None:
    assert "document.body.classList.contains('dark') ? 'light' : 'dark'" in TOGGLE_THEME_JS
    assert "localStorage.setItem('secourses_theme_mode', mode)" in TOGGLE_THEME_JS
    assert "window.__secoursesApplyThemeMode(mode)" in TOGGLE_THEME_JS
    # The new mode is handed back as the radio's value so both controls agree.
    assert "return [mode];" in TOGGLE_THEME_JS


def test_product_css_carries_no_second_light_mode_palette() -> None:
    """Light and dark must come from theme variables, not a duplicated palette.

    The one exception is the action-button gradient, border, and shadow, which
    need a lighter drop shadow on a white page than on a dark one.
    """

    css = build_css()
    light_overrides = re.findall(r"body:not\(\.dark\)[^{]*\{", css)
    assert light_overrides == [
        "body:not(.dark) button.vc-btn {",
        "body:not(.dark) button.vc-btn:hover:not(:disabled) {",
    ], light_overrides

    # Every other colour resolves through a Gradio theme variable. Literal hex
    # is allowed only in the status-colour tokens and on the action buttons,
    # whose saturated gradients carry the same near-white label in both modes.
    for selector, declarations in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        selector = selector.rsplit("*/", 1)[-1].strip()
        if selector in {":root", ".dark"} or "vc-btn" in selector:
            continue
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", declarations), selector


def test_product_css_only_styles_markup_the_theme_cannot_reach() -> None:
    css = build_css()
    for selector in (
        ".vc-input-list",  # app-rendered file tiles
        ".vc-progress__track",  # app-rendered job progress
        ".vc-meter__fill",  # app-rendered VRAM/RAM meters
        ".vcap-replace-chip",  # app-rendered find/replace chips
        ".vc-ok",  # status words
        ".vc-header",  # header rule
        ".vc-preset-bar",  # preset strip button alignment
        ".vc-confirm-bar",  # inline confirmation bar
        ".vc-mono textarea",  # monospace logs
        "button.vc-btn",  # multi-hue action buttons
    ):
        assert selector in css, selector
    # Chrome the stock theme provides on its own must not be re-styled: page
    # width, tab bar, block cards, row alignment, media sizing, scrollbars.
    for removed in (
        ".gradio-container",
        "position: sticky",
        ".vc-card",
        ".vc-section-title",
        ".vc-compact-row",
        ".vc-action-row",
        ".vc-preview",
        ".vc-result-panel",
        ".vc-scroll-result",
        ".vc-editor-gallery",
        "scrollbar-width",
        "@media",
    ):
        assert removed not in css, removed


def test_action_buttons_use_the_indextts_recipe() -> None:
    """Both SECourses apps share one button recipe: one height, weight, glow."""

    css = build_css()
    button = re.search(r"^button\.vc-btn \{([^}]*)\}", css, re.MULTILINE)
    assert button is not None
    body = button.group(1)
    for declaration in (
        "min-height: 44px",
        "padding: var(--size-2) var(--size-4) !important",
        "font-size: var(--text-md) !important",
        "font-weight: 650 !important",
        "border-radius: var(--radius-lg) !important",
        "linear-gradient(135deg, var(--vc-h1) 0%, var(--vc-h2) 55%, var(--vc-h3) 100%)",
        "box-shadow: 0 8px 20px rgb(var(--vc-hue) / .30)",
    ):
        assert declaration in body, declaration
    # Buttons keep the uniform height inside rows instead of stretching to the
    # tallest neighbour, and the preset strip lines them up with its fields.
    assert ".row > button.vc-btn { align-self: center; }" in css
    assert ".row.vc-preset-bar > button.vc-btn { align-self: flex-end !important; margin-bottom: 12px; }" in css
    # Every hue is one custom-property line; the twenty IndexTTS hues use the
    # same stops as that app.
    for hue in (
        "emerald", "green", "lime", "teal", "cyan", "sky", "blue", "indigo", "violet", "purple",
        "fuchsia", "pink", "rose", "red", "crimson", "orange", "amber", "bronze", "slate", "gray",
    ):
        assert f".vc-btn-{hue}{{--vc-h1:" in css, hue
    assert ".vc-btn-crimson{--vc-h1:#4c0519;--vc-h2:#9f1239;--vc-h3:#fb7185;" in css
    assert ".vc-btn-bronze{--vc-h1:#5c3a21;--vc-h2:#8b5a2b;--vc-h3:#d4a373;" in css
    assert ".vc-btn-gold{" in css  # alias of yellow, used by the preset strip


def test_theme_is_stock_origin_with_offline_fonts() -> None:
    theme = build_theme()
    assert isinstance(theme, gr.themes.Origin)
    # Used exactly as shipped: no design token differs from a fresh Origin.
    assert theme.to_dict() == gr.themes.Origin().to_dict()
    # Bundled fonts only: a caption box with no internet must still render.
    for stylesheet in theme._stylesheets:
        assert not stylesheet.startswith("http"), stylesheet
