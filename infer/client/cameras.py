"""One RGB source per registered camera, with bounded snapshot freshness."""

# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


import threading
import time
import numpy as np


class Cameras:
    """Keep blocking USB reads away from policy execution and Space handling."""

    def __init__(self, serials):
        import pyrealsense2 as rs  # pylint: disable=import-error

        self.frames = {}
        self.errors = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.pipelines = []
        self.threads = []
        try:
            for role, serial in serials.items():
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(
                    rs.stream.color, 640, 480, rs.format.rgb8, 30
                )
                pipeline.start(config)
                self.pipelines.append(pipeline)
                worker = threading.Thread(
                    target=self._read, args=(role, pipeline), daemon=True
                )
                self.threads.append(worker)
                worker.start()
        except BaseException:
            self.close()
            raise

    def _read(self, role, pipeline):
        try:
            while not self.done.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                image = np.asanyarray(
                    frames.get_color_frame().get_data()
                ).copy()
                with self.lock:
                    self.frames[role] = (time.monotonic(), image)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # A device-thread failure is forwarded to the control owner.
            if not self.done.is_set():
                with self.lock:
                    self.errors.append(str(error))

    def snapshot(self):
        """Receipt-time admission; capture-time synchronization needs
        shadow QA.
        """
        now = time.monotonic()
        with self.lock:
            if self.errors or len(self.frames) != 3:
                raise RuntimeError("camera source unavailable")
            stamps = [value[0] for value in self.frames.values()]
            if (
                any(not 0 <= now - stamp <= 0.15 for stamp in stamps)
                or max(stamps) - min(stamps) > 0.05
            ):
                raise RuntimeError("stale or skewed camera snapshot")
            return {
                role: value[1].copy() for role, value in self.frames.items()
            }

    def close(self):
        """Stop pipelines and join bounded readers."""
        self.done.set()
        for pipeline in self.pipelines:
            pipeline.stop()
        for worker in self.threads:
            worker.join(timeout=1.5)
