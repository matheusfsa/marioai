from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from marioai import capture as capture_module
from marioai.capture import (
    CaptureBackendError,
    GameWindowCapture,
    WindowDisappearedError,
    WindowNotFoundError,
)


# ---------------------------------------------------------------------------
# helpers / fakes
# ---------------------------------------------------------------------------
def _fake_window(title: str, top: int = 0, left: int = 0, width: int = 800, height: int = 600):
    return SimpleNamespace(title=title, top=top, left=left, width=width, height=height)


def _fake_screenshot(width: int, height: int, fill: int = 128) -> SimpleNamespace:
    pixels = np.full((height, width, 3), fill, dtype=np.uint8)
    return SimpleNamespace(width=width, height=height, rgb=pixels.tobytes())


def _patch_pygetwindow(monkeypatch: pytest.MonkeyPatch, windows: list) -> MagicMock:
    fake_gw = MagicMock()
    fake_gw.getAllWindows.return_value = windows
    monkeypatch.setattr(capture_module, '_load_pygetwindow', lambda: fake_gw)
    return fake_gw


def _patch_mss(monkeypatch: pytest.MonkeyPatch, screenshot: SimpleNamespace) -> MagicMock:
    fake_mss_module = MagicMock()
    fake_mss_instance = MagicMock()
    fake_mss_instance.grab.return_value = screenshot
    fake_mss_module.mss.return_value = fake_mss_instance
    monkeypatch.setattr(capture_module, '_load_mss', lambda: fake_mss_module)
    return fake_mss_instance


# ---------------------------------------------------------------------------
# find_window
# ---------------------------------------------------------------------------
class TestFindWindow:
    def test_partial_case_insensitive_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pygetwindow(monkeypatch, [_fake_window('Infinite MARIO Bros!')])
        cap = GameWindowCapture('mario')
        cap.find_window()
        assert cap._window.title == 'Infinite MARIO Bros!'
        assert cap._bbox == {'top': 0, 'left': 0, 'width': 800, 'height': 600}

    def test_no_match_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pygetwindow(monkeypatch, [_fake_window('Firefox'), _fake_window('Terminal')])
        cap = GameWindowCapture('Mario')
        with pytest.raises(WindowNotFoundError, match='no window matches'):
            cap.find_window()

    def test_multi_match_picks_largest_then_index(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        windows = [
            _fake_window('Mario A', width=400, height=300),
            _fake_window('Mario B (big)', width=1600, height=900),
            _fake_window('Mario C', width=800, height=600),
        ]
        _patch_pygetwindow(monkeypatch, windows)
        cap = GameWindowCapture('Mario', window_index=0)
        with caplog.at_level(logging.WARNING, logger='marioai.capture'):
            cap.find_window()
        # largest first → 'Mario B (big)' wins for index 0
        assert cap._window.title == 'Mario B (big)'
        assert any('3 windows match' in rec.message for rec in caplog.records)

    def test_zero_size_filtered_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        windows = [_fake_window('Mario', width=0, height=0), _fake_window('Mario real', width=800, height=600)]
        _patch_pygetwindow(monkeypatch, windows)
        cap = GameWindowCapture('Mario')
        cap.find_window()
        assert cap._window.title == 'Mario real'


# ---------------------------------------------------------------------------
# capture_frame
# ---------------------------------------------------------------------------
class TestCaptureFrame:
    def test_happy_path_rgb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pygetwindow(monkeypatch, [_fake_window('Mario', width=200, height=100)])
        _patch_mss(monkeypatch, _fake_screenshot(200, 100, fill=200))
        cap = GameWindowCapture('Mario', log_fps_every=0)
        cap.start()
        frame = cap.capture_frame()
        cap.stop()
        assert frame is not None
        assert frame.shape == (100, 200, 3)
        assert frame.dtype == np.uint8
        assert int(frame.mean()) == 200

    def test_grayscale_no_cv2_uses_numpy_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pygetwindow(monkeypatch, [_fake_window('Mario', width=10, height=10)])
        _patch_mss(monkeypatch, _fake_screenshot(10, 10, fill=100))
        monkeypatch.setattr(capture_module, '_load_cv2', lambda: None)
        cap = GameWindowCapture('Mario', grayscale=True, log_fps_every=0)
        cap.start()
        frame = cap.capture_frame()
        cap.stop()
        assert frame is not None
        assert frame.shape == (10, 10)
        assert frame.dtype == np.uint8

    def test_resize_without_cv2_raises_on_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pygetwindow(monkeypatch, [_fake_window('Mario')])
        _patch_mss(monkeypatch, _fake_screenshot(800, 600))
        monkeypatch.setattr(capture_module, '_load_cv2', lambda: None)
        cap = GameWindowCapture('Mario', resize=(84, 84))
        with pytest.raises(CaptureBackendError, match='resize'):
            cap.start()

    def test_resize_with_cv2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pygetwindow(monkeypatch, [_fake_window('Mario', width=200, height=100)])
        _patch_mss(monkeypatch, _fake_screenshot(200, 100))
        fake_cv2 = MagicMock()
        fake_cv2.COLOR_RGB2GRAY = 7
        fake_cv2.INTER_AREA = 3
        # Simulate cv2.resize → (h, w[, c]) array
        fake_cv2.resize.side_effect = lambda img, dsize, interpolation: np.zeros(
            (dsize[1], dsize[0], img.shape[2] if img.ndim == 3 else 1), dtype=np.uint8
        ).squeeze()
        fake_cv2.cvtColor.side_effect = lambda img, code: np.zeros(img.shape[:2], dtype=np.uint8)
        monkeypatch.setattr(capture_module, '_load_cv2', lambda: fake_cv2)
        cap = GameWindowCapture('Mario', grayscale=True, resize=(84, 84), log_fps_every=0)
        cap.start()
        frame = cap.capture_frame()
        cap.stop()
        assert frame is not None
        assert frame.shape == (84, 84)
        assert frame.dtype == np.uint8

    def test_lost_window_recovered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        win = _fake_window('Mario', width=100, height=100)
        gw = _patch_pygetwindow(monkeypatch, [win])
        _patch_mss(monkeypatch, _fake_screenshot(100, 100))
        cap = GameWindowCapture('Mario', log_fps_every=0)
        cap.start()
        # First capture works.
        assert cap.capture_frame() is not None
        # Make the next find_window succeed too (still in the list).
        gw.getAllWindows.return_value = [win]
        # Inject a transient failure in the mss grab path.
        cap._sct.grab.side_effect = [RuntimeError('transient'), _fake_screenshot(100, 100)]
        recovered = cap.capture_frame()
        # Returns None on failure (after attempting reacquire), no exception raised.
        assert recovered is None
        cap.stop()

    def test_lost_window_unrecoverable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        win = _fake_window('Mario', width=100, height=100)
        gw = _patch_pygetwindow(monkeypatch, [win])
        _patch_mss(monkeypatch, _fake_screenshot(100, 100))
        cap = GameWindowCapture('Mario', log_fps_every=0, reacquire_on_failure=False)
        cap.start()
        cap._sct.grab.side_effect = RuntimeError('display gone')
        # With reacquire disabled, _handle_lost_window raises immediately.
        with pytest.raises(WindowDisappearedError):
            cap.capture_frame()
        cap.stop()
        gw.getAllWindows.assert_called()  # at least once during start


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------
class TestFPS:
    def test_fps_counter_increments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_pygetwindow(monkeypatch, [_fake_window('Mario', width=10, height=10)])
        _patch_mss(monkeypatch, _fake_screenshot(10, 10))
        cap = GameWindowCapture('Mario', log_fps_every=0)
        cap.start()
        for _ in range(5):
            cap.capture_frame()
        cap.stop()
        assert cap._frame_count == 5
        assert cap.fps >= 0.0  # may be huge in tests; just check it computes

    def test_fps_log_respects_interval(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        _patch_pygetwindow(monkeypatch, [_fake_window('Mario', width=10, height=10)])
        _patch_mss(monkeypatch, _fake_screenshot(10, 10))
        cap = GameWindowCapture('Mario', log_fps_every=999.0)  # effectively never
        cap.start()
        with caplog.at_level(logging.INFO, logger='marioai.capture'):
            for _ in range(3):
                cap.capture_frame()
        cap.stop()
        assert not any('fps' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------
class TestFromEnv:
    def test_no_window_var_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('MARIOAI_CAPTURE_WINDOW', raising=False)
        assert GameWindowCapture.from_env() is None

    def test_full_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('MARIOAI_CAPTURE_WINDOW', 'Mario')
        monkeypatch.setenv('MARIOAI_CAPTURE_GRAYSCALE', '1')
        monkeypatch.setenv('MARIOAI_CAPTURE_RESIZE', '84x84')
        monkeypatch.setenv('MARIOAI_CAPTURE_BACKEND', 'mss')
        cap = GameWindowCapture.from_env()
        assert cap is not None
        assert cap.window_title == 'Mario'
        assert cap.grayscale is True
        assert cap.resize == (84, 84)
        assert cap.backend == 'mss'

    def test_bad_resize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('MARIOAI_CAPTURE_WINDOW', 'Mario')
        monkeypatch.setenv('MARIOAI_CAPTURE_RESIZE', 'huge')
        with pytest.raises(ValueError, match='MARIOAI_CAPTURE_RESIZE'):
            GameWindowCapture.from_env()

    def test_bad_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('MARIOAI_CAPTURE_WINDOW', 'Mario')
        monkeypatch.setenv('MARIOAI_CAPTURE_BACKEND', 'gdi')
        with pytest.raises(ValueError, match='MARIOAI_CAPTURE_BACKEND'):
            GameWindowCapture.from_env()


# ---------------------------------------------------------------------------
# win32 backend on non-Windows
# ---------------------------------------------------------------------------
class TestWin32Backend:
    def test_non_windows_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'linux')
        cap = GameWindowCapture('Mario', backend='win32')
        with pytest.raises(CaptureBackendError, match='only available on Windows'):
            cap.start()


# ---------------------------------------------------------------------------
# Linux/X11 shim
# ---------------------------------------------------------------------------
class TestX11Shim:
    def test_load_pygetwindow_falls_back_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``pygetwindow`` raises ``NotImplementedError`` on Linux — the loader must fall through to the X11 shim."""
        monkeypatch.setattr(sys, 'platform', 'linux')

        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name == 'pygetwindow':
                raise NotImplementedError('PyGetWindow currently does not support Linux.')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        result = capture_module._load_pygetwindow()
        assert hasattr(result, 'getAllWindows'), f'fallback must expose getAllWindows, got {result!r}'

    def test_x11_shim_enumerates_and_queries_geometry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end shim test with fake ewmh clients — no real X11 dependency."""
        shim = capture_module._X11PyGetWindowShim()

        fake_geom = SimpleNamespace(x=5, y=7, width=320, height=240)
        fake_parent_tree = SimpleNamespace(parent=0)  # direct child of root
        fake_xwin = MagicMock()
        fake_xwin.id = 0xDEADBEEF
        fake_xwin.get_geometry.return_value = fake_geom
        fake_xwin.query_tree.return_value = fake_parent_tree

        fake_root = MagicMock()
        fake_ewmh = MagicMock()
        fake_ewmh.display.screen.return_value.root = fake_root
        fake_ewmh.getClientList.return_value = [fake_xwin]
        fake_ewmh.getWmName.return_value = b'Mario Window'

        fake_ewmh_mod = MagicMock()
        fake_ewmh_mod.EWMH.return_value = fake_ewmh
        monkeypatch.setitem(sys.modules, 'ewmh', fake_ewmh_mod)

        windows = shim.getAllWindows()
        assert len(windows) == 1
        w = windows[0]
        assert w.title == 'Mario Window'
        assert (w.left, w.top, w.width, w.height) == (5, 7, 320, 240)
        assert w._hWnd == 0xDEADBEEF

    def test_x11_shim_skips_nameless_and_broken_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shim = capture_module._X11PyGetWindowShim()
        nameless = MagicMock()
        broken = MagicMock()
        broken.get_geometry.side_effect = RuntimeError('zombie')
        good = MagicMock()
        good.id = 1
        good.get_geometry.return_value = SimpleNamespace(x=0, y=0, width=10, height=10)
        good.query_tree.return_value = SimpleNamespace(parent=0)

        fake_ewmh = MagicMock()
        fake_ewmh.display.screen.return_value.root = MagicMock()
        fake_ewmh.getClientList.return_value = [nameless, broken, good]
        fake_ewmh.getWmName.side_effect = [None, b'zombie', b'ok']

        fake_ewmh_mod = MagicMock()
        fake_ewmh_mod.EWMH.return_value = fake_ewmh
        monkeypatch.setitem(sys.modules, 'ewmh', fake_ewmh_mod)

        windows = shim.getAllWindows()
        assert [w.title for w in windows] == ['ok']
