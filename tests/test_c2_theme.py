from __future__ import annotations

import re

import gradio as gr

from vcap.ui.theme import (
    HOTKEYS_HEAD,
    THEME_CHANGE_JS,
    TOGGLE_ACCORDIONS_JS,
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


def test_product_css_carries_no_second_light_mode_palette() -> None:
    """Light and dark must come from theme variables, not a duplicated palette.

    The one exception is the action-button border/shadow pair, which needs a
    lighter drop shadow on a white page than on a dark one.
    """

    css = build_css()
    light_overrides = re.findall(r"body:not\(\.dark\)[^{]*\{", css)
    assert light_overrides == [
        "body:not(.dark) .vc-btn, body:not(.dark) .vc-btn.vc-btn {"
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
        ".vc-header",  # product header band
        ".vc-btn",  # multi-hue action buttons
    ):
        assert selector in css
    # Chrome the stock theme now provides on its own must not be re-styled.
    for removed in ("#vc-main-tabs .tab-container", "#vc-input-tabs", ".vc-shell"):
        assert removed not in css


def test_sticky_tab_bar_clears_gradio_block_titles() -> None:
    css = build_css()
    assert "position: sticky" in css
    # Block titles sit at z-index 40; dropdown menus use --layer-top.
    assert "z-index: 60" in css
    assert "overflow: visible !important" in css


def test_every_button_size_shares_the_same_metrics() -> None:
    theme = build_theme()
    assert theme.button_small_text_size == theme.button_medium_text_size == theme.button_large_text_size == "*text_md"
    assert theme.button_small_padding == theme.button_medium_padding == theme.button_large_padding


def test_theme_is_a_stock_gradio_theme_with_offline_fonts() -> None:
    theme = build_theme()
    assert isinstance(theme, gr.themes.Ocean)
    # Bundled fonts only: a caption box with no internet must still render.
    for stylesheet in theme._stylesheets:
        assert not stylesheet.startswith("http"), stylesheet


def test_light_mode_label_and_help_text_meet_contrast() -> None:
    """Ocean leaves both at neutral_400/500, which is under 3:1 on white."""

    theme = build_theme()
    assert theme.body_text_color_subdued == "*neutral_600"
    assert theme.body_text_color_subdued_dark == "*neutral_400"
    assert theme.block_title_text_color == "*neutral_700"
    assert theme.block_title_text_color_dark == "*neutral_200"
