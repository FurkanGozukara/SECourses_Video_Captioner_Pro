from __future__ import annotations

from vcap.ui.theme import HOTKEYS_HEAD, THEME_CHANGE_JS, build_css


def test_head_script_has_dark_default_and_three_mode_theme_logic() -> None:
    assert "secourses_theme_mode" in HOTKEYS_HEAD
    assert "localStorage.setItem('secourses_theme_mode', 'dark')" in HOTKEYS_HEAD
    assert "stored !== 'dark' && stored !== 'light' && stored !== 'system'" in HOTKEYS_HEAD
    assert "matchMedia('(prefers-color-scheme: dark)')" in HOTKEYS_HEAD
    assert "addEventListener('change', onSystemThemeChange)" in HOTKEYS_HEAD
    assert "url.searchParams.set('__theme', effective)" in HOTKEYS_HEAD
    assert "document.body.classList.toggle('dark', effective === 'dark')" in HOTKEYS_HEAD
    assert "['dark', 'light', 'system']" in THEME_CHANGE_JS


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


def test_light_mode_css_covers_native_controls_log_tabs_and_focus() -> None:
    css = build_css()
    assert "body:not(.dark) input[type=checkbox]:not(:checked)" in css
    assert "body:not(.dark) input[type=radio]:not(:checked)" in css
    assert "body:not(.dark) .gradio-container .vc-log textarea" in css
    assert "body:not(.dark) #vc-main-tabs > .tab-wrapper > .tab-container" in css
    assert "body:not(.dark) .vcap-replace-arrow" in css
    assert "body:not(.dark) .vc-btn:focus-visible" in css

