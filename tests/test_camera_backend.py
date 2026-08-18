import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

from scripts.inference_utils import build_capture_backends, open_video_capture


class FakeCapture:
    def __init__(self, source: int, backend: int, opened: bool):
        self.source = source
        self.backend = backend
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened


class CameraBackendTests(unittest.TestCase):
    def test_windows_prefers_direct_show_for_webcam_sources(self):
        with patch("scripts.inference_utils.os.name", "nt"):
            backends = build_capture_backends(0)

        self.assertEqual(backends[0], cv2.CAP_DSHOW)
        self.assertIn(cv2.CAP_ANY, backends)

    def test_open_video_capture_tries_multiple_indices(self):
        def fake_video_capture(source: int, backend: int):
            if source == 1 and backend == cv2.CAP_DSHOW:
                return FakeCapture(source, backend, True)
            return FakeCapture(source, backend, False)

        with patch("scripts.inference_utils.cv2.VideoCapture", side_effect=fake_video_capture):
            capture, backend_used = open_video_capture(0)

        self.assertIsNotNone(capture)
        self.assertEqual(backend_used, cv2.CAP_DSHOW)
        self.assertEqual(capture.source, 1)

    def test_open_video_capture_accepts_video_file_paths(self):
        video_path = Path("sample.mp4")

        def fake_video_capture(source, backend):
            self.assertEqual(source, str(video_path))
            self.assertEqual(backend, cv2.CAP_ANY)
            return FakeCapture(0, backend, True)

        with patch("scripts.inference_utils.cv2.VideoCapture", side_effect=fake_video_capture):
            capture, backend_used = open_video_capture(str(video_path))

        self.assertIsNotNone(capture)
        self.assertEqual(backend_used, cv2.CAP_ANY)


if __name__ == "__main__":
    unittest.main()
