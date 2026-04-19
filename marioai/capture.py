"""Live capture of the Mario server's game window.

The TCP protocol exposes only a symbolic state. To train pixel-based agents
(CNN policies, behavioural cloning from video, etc.) we additionally need the
rendered frames the server draws on screen.

This module wraps `mss <https://github.com/BoboTiG/python-mss>`_ for fast
region capture and `pygetwindow <https://github.com/asweigart/PyGetWindow>`_
to locate the window by partial title match. A Windows-only ``win32`` backend
is also available as a fallback.

Typical usage::

    from marioai.capture import GameWindowCapture

    with GameWindowCapture('Mario', grayscale=True, resize=(84, 84)) as cap:
        frame = cap.capture_frame()  # np.ndarray, shape (84, 84), uint8

The ``[capture]`` optional dependency must be installed::

    pip install marioai[capture]
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from typing import Literal

import numpy as np

__all__ = [
    'CaptureBackendError',
    'CaptureError',
    'GameWindowCapture',
    'WindowDisappearedError',
    'WindowNotFoundError',
]

logger = logging.getLogger(__name__)

_INSTALL_HINT = "install with: pip install 'marioai[capture]'"
_LUMA_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class CaptureError(Exception):
    """Base class for capture-related errors."""


class WindowNotFoundError(CaptureError):
    """No window matched the requested title."""


class WindowDisappearedError(CaptureError):
    """The previously found window vanished and could not be recovered."""


class CaptureBackendError(CaptureError):
    """A required backend (mss, pygetwindow, cv2, win32) is unavailable."""


# ---------------------------------------------------------------------------
# Lazy backend loaders. Kept as module-level helpers so tests can monkeypatch.
# ---------------------------------------------------------------------------
def _load_mss():
    try:
        import mss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CaptureBackendError(f'mss is not installed; {_INSTALL_HINT}') from exc
    return mss


def _load_pygetwindow():
    try:
        import pygetwindow  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CaptureBackendError(f'pygetwindow is not installed; {_INSTALL_HINT}') from exc
    return pygetwindow


def _load_cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    return cv2


def _load_win32():
    if sys.platform != 'win32':
        raise CaptureBackendError("backend='win32' is only available on Windows")
    try:
        import win32con  # type: ignore[import-not-found]
        import win32gui  # type: ignore[import-not-found]
        import win32ui  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CaptureBackendError(f'pywin32 is not installed; {_INSTALL_HINT}') from exc
    return win32gui, win32ui, win32con


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class GameWindowCapture:
    """Capture frames from a native game window.

    Args:
      window_title: Substring of the window title to match (case-insensitive).
      grayscale: Convert frames to single-channel uint8.
      resize: Optional ``(width, height)`` to resize each frame to.
      reacquire_on_failure: Try to re-find the window if a capture call fails.
      window_index: When multiple windows match the title, pick this one
        (sorted by descending area).
      log_fps_every: Emit an INFO log with the current FPS every N seconds.
        Set to ``0`` to disable.
      backend: ``'mss'`` (default, screen-pixel grab) or ``'win32'``
        (Windows-only ``PrintWindow`` — works for minimized/occluded windows).
      reacquire_every_n_frames: Refresh window position every N frames to
        track movement without paying the cost on every capture.
    """

    def __init__(
        self,
        window_title: str = 'Mario',
        grayscale: bool = False,
        resize: tuple[int, int] | None = None,
        reacquire_on_failure: bool = True,
        window_index: int = 0,
        log_fps_every: float = 5.0,
        backend: Literal['mss', 'win32'] = 'mss',
        reacquire_every_n_frames: int = 30,
    ) -> None:
        self.window_title = window_title
        self.grayscale = grayscale
        self.resize = resize
        self.reacquire_on_failure = reacquire_on_failure
        self.window_index = window_index
        self.log_fps_every = log_fps_every
        self.backend: Literal['mss', 'win32'] = backend
        self.reacquire_every_n_frames = max(1, reacquire_every_n_frames)

        self._sct = None  # mss instance, lazy
        self._window = None  # pygetwindow Window
        self._bbox: dict[str, int] | None = None  # {top, left, width, height}
        self._frame_count = 0
        self._fps_window: deque[float] = deque(maxlen=60)
        self._last_fps_log = 0.0
        self._cv2 = None
        self._started = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> GameWindowCapture:
        """Acquire backend resources and locate the window."""
        if self._started:
            return self
        if self.backend == 'mss':
            mss = _load_mss()
            self._sct = mss.mss()
        elif self.backend == 'win32':
            _load_win32()  # validate availability now
        else:
            raise CaptureBackendError(f'unknown backend: {self.backend!r}')
        self._cv2 = _load_cv2()
        if self.resize is not None and self._cv2 is None:
            raise CaptureBackendError(
                f'resize={self.resize!r} requires opencv-python; {_INSTALL_HINT}',
            )
        self.find_window()
        self._started = True
        return self

    def stop(self) -> None:
        """Release backend resources. Safe to call multiple times."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # noqa: BLE001 — defensive cleanup
                logger.debug('error closing mss instance', exc_info=True)
        self._sct = None
        self._window = None
        self._bbox = None
        self._started = False

    def __enter__(self) -> GameWindowCapture:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # window discovery
    # ------------------------------------------------------------------
    def find_window(self) -> None:
        """Locate the window matching ``window_title``.

        Raises:
          WindowNotFoundError: if no window matches.
        """
        gw = _load_pygetwindow()
        needle = self.window_title.lower()
        try:
            all_windows = gw.getAllWindows()
        except Exception as exc:  # noqa: BLE001 — backend-dependent
            raise CaptureBackendError(f'pygetwindow.getAllWindows() failed: {exc}') from exc

        # Filter by partial, case-insensitive match; ignore zero-sized phantoms.
        matches = [w for w in all_windows if needle in (w.title or '').lower() and w.width > 0 and w.height > 0]
        if not matches:
            visible_titles = sorted({(w.title or '<no title>') for w in all_windows if w.title})
            raise WindowNotFoundError(
                f'no window matches {self.window_title!r}. Visible titles: {visible_titles[:20]}{"..." if len(visible_titles) > 20 else ""}',
            )
        if len(matches) > 1:
            titles = [w.title for w in matches]
            logger.warning('[capture] %d windows match %r: %s — picking index %d (largest first)', len(matches), self.window_title, titles, self.window_index)
            matches.sort(key=lambda w: w.width * w.height, reverse=True)
        idx = min(self.window_index, len(matches) - 1)
        self._window = matches[idx]
        self._update_bbox_from_window()
        logger.info(
            "[capture] window found: '%s' (%dx%d at %d,%d)",
            self._window.title,
            self._bbox['width'],
            self._bbox['height'],
            self._bbox['left'],
            self._bbox['top'],
        )

    def update_window_position(self) -> None:
        """Re-read the bounding box (window may have moved)."""
        if self._window is None:
            self.find_window()
            return
        try:
            self._update_bbox_from_window()
        except Exception:  # noqa: BLE001 — window handle may be stale
            logger.debug('window handle stale, re-acquiring', exc_info=True)
            self._window = None
            self.find_window()

    def _update_bbox_from_window(self) -> None:
        w = self._window
        if w is None or w.width <= 0 or w.height <= 0:
            raise WindowDisappearedError(f'window {self.window_title!r} has zero size')
        self._bbox = {'top': int(w.top), 'left': int(w.left), 'width': int(w.width), 'height': int(w.height)}

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------
    def capture_frame(self) -> np.ndarray | None:
        """Return one frame of the window or ``None`` on transient failure.

        Raises:
          WindowDisappearedError: if the window can no longer be found and
            ``reacquire_on_failure`` is enabled but exhausted its retries.
        """
        if not self._started:
            self.start()

        # Periodic re-check of the window position (cheap; tracks dragging).
        if self._frame_count % self.reacquire_every_n_frames == 0 and self._frame_count > 0:
            try:
                self.update_window_position()
            except WindowNotFoundError as exc:
                self._handle_lost_window(exc)
                return None

        try:
            if self.backend == 'mss':
                frame = self._capture_mss()
            else:
                frame = self._capture_win32()
        except WindowDisappearedError:
            raise
        except Exception as exc:  # noqa: BLE001 — backends throw varied types
            logger.warning('[capture] capture failed: %s', exc)
            self._handle_lost_window(exc)
            return None

        if frame is None:
            return None

        if self.grayscale:
            frame = self._to_grayscale(frame)
        if self.resize is not None:
            frame = self._resize(frame)

        self._frame_count += 1
        self._tick_fps()
        return frame

    def _capture_mss(self) -> np.ndarray:
        assert self._sct is not None and self._bbox is not None
        shot = self._sct.grab(self._bbox)
        # mss returns BGRA; .rgb gives the same pixels in RGB order, no alpha.
        return np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)

    def _capture_win32(self) -> np.ndarray:
        # Lazy: grabs the window contents via PrintWindow. Works for minimized
        # or occluded windows because it asks the owning process to redraw
        # itself onto the provided device context — unlike mss, which reads
        # the on-screen framebuffer.
        win32gui, win32ui, win32con = _load_win32()
        assert self._window is not None
        hwnd = self._window._hWnd  # pygetwindow on Windows exposes the HWND
        left, top, right, bot = win32gui.GetClientRect(hwnd)
        width, height = right - left, bot - top
        if width <= 0 or height <= 0:
            raise WindowDisappearedError(f'window {self.window_title!r} has zero client size')

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bmp)
        # 1 = PW_CLIENTONLY; 3 = PW_CLIENTONLY | PW_RENDERFULLCONTENT (Win >= 8.1).
        ctypes_print_window = getattr(win32gui, 'PrintWindow', None)
        result = 0
        if ctypes_print_window is not None:
            result = ctypes_print_window(hwnd, save_dc.GetSafeHdc(), 3)
        bmp_info = bmp.GetInfo()
        bmp_str = bmp.GetBitmapBits(True)
        img = np.frombuffer(bmp_str, dtype=np.uint8).reshape(bmp_info['bmHeight'], bmp_info['bmWidth'], 4)
        # cleanup
        win32gui.DeleteObject(bmp.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        if result == 0:
            logger.debug('[capture] PrintWindow returned 0 — image may be blank')
        return img[:, :, :3][:, :, ::-1]  # BGRA→RGB

    # ------------------------------------------------------------------
    # post-processing
    # ------------------------------------------------------------------
    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if self._cv2 is not None:
            return self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2GRAY)
        # Pure-numpy NTSC luma — slower but dependency-free.
        return (frame.astype(np.float32) @ _LUMA_RGB).astype(np.uint8)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        assert self._cv2 is not None and self.resize is not None
        w, h = self.resize
        # INTER_AREA gives the best downscaling quality (Atari preprocessing
        # convention; see Mnih et al. 2015).
        return self._cv2.resize(frame, (w, h), interpolation=self._cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # error recovery
    # ------------------------------------------------------------------
    def _handle_lost_window(self, exc: BaseException) -> None:
        if not self.reacquire_on_failure:
            raise WindowDisappearedError(f'capture failed and reacquire disabled: {exc}') from exc
        for attempt in range(1, 4):
            time.sleep(0.05 * attempt)  # short linear backoff
            try:
                self.find_window()
                logger.info('[capture] window re-acquired on attempt %d', attempt)
                return
            except WindowNotFoundError:
                continue
        raise WindowDisappearedError(f'window {self.window_title!r} could not be re-acquired after 3 attempts') from exc

    # ------------------------------------------------------------------
    # FPS instrumentation
    # ------------------------------------------------------------------
    def _tick_fps(self) -> None:
        now = time.monotonic()
        self._fps_window.append(now)
        if self.log_fps_every <= 0:
            return
        if now - self._last_fps_log < self.log_fps_every:
            return
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            fps = (len(self._fps_window) - 1) / span if span > 0 else 0.0
            logger.info('[capture] %.1f fps', fps)
        self._last_fps_log = now

    @property
    def fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        if span <= 0:
            return 0.0
        return (len(self._fps_window) - 1) / span

    # ------------------------------------------------------------------
    # config helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> GameWindowCapture | None:
        """Build an instance from ``MARIOAI_CAPTURE_*`` env vars.

        Returns ``None`` if ``MARIOAI_CAPTURE_WINDOW`` is unset, so callers can
        no-op silently when the user hasn't opted in.
        """
        title = os.environ.get('MARIOAI_CAPTURE_WINDOW')
        if not title:
            return None
        grayscale = os.environ.get('MARIOAI_CAPTURE_GRAYSCALE', '0') in ('1', 'true', 'True')
        resize_env = os.environ.get('MARIOAI_CAPTURE_RESIZE')
        resize: tuple[int, int] | None = None
        if resize_env:
            try:
                w_str, h_str = resize_env.lower().split('x')
                resize = (int(w_str), int(h_str))
            except ValueError as exc:
                raise ValueError(f"MARIOAI_CAPTURE_RESIZE must look like '84x84', got {resize_env!r}") from exc
        backend = os.environ.get('MARIOAI_CAPTURE_BACKEND', 'mss')
        if backend not in ('mss', 'win32'):
            raise ValueError(f"MARIOAI_CAPTURE_BACKEND must be 'mss' or 'win32', got {backend!r}")
        return cls(window_title=title, grayscale=grayscale, resize=resize, backend=backend)  # type: ignore[arg-type]
