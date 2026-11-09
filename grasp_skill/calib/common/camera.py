# Description: 相机侧的两件共用东西——取图接口、去畸变包装、JPEG 编码。
#
# 具体相机类（S10 的 pyrealsense2、piper 的 pyorbbecsdk）各自放在自己的子树里，
# 因为它们绑的是不同厂家的 SDK；这里只放与厂家无关的部分。
from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    """取图接口：各部署的相机类都满足它（get_frames 返回 {"color": bgr}）。"""

    def get_frames(self) -> dict: ...

    def close(self) -> None: ...


class UndistortCamera:
    """给任意相机套一层去畸变（K/D 来自内参标定产物）。

    不直接改相机类，是为了让"取图"和"畸变修正"两件事分得开：内参标定本身
    就要看原始画面，不能套这一层。手眼标定则必须套上——标定输入要与运行时一致。
    """

    def __init__(self, cam: FrameSource, k_path, d_path):
        self._cam = cam
        self.K = np.load(k_path)
        self.D = np.load(d_path)

    def get_frames(self) -> dict:
        frames = self._cam.get_frames()
        if frames.get("color") is not None:
            frames["color"] = cv2.undistort(frames["color"], self.K, self.D)
        return frames

    def close(self) -> None:
        self._cam.close()


def encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes | None:
    """把一帧编码成 JPEG（各页面的 MJPEG 流都走它）。"""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def mjpeg_stream(frames_fn, fps: float = 25.0):
    """把一个"取当前帧"的回调变成 MJPEG 分块流（Flask Response 的生成器）。

    Args:
        frames_fn: 无参回调，返回 BGR 帧或 None（None 表示这一拍没图，跳过）。
        fps: 每秒最多推几帧。
    """
    import time

    interval = 1.0 / max(1.0, float(fps))
    while True:
        frame = frames_fn()
        if frame is None:
            time.sleep(interval)
            continue
        jpg = encode_jpeg(frame)
        if jpg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(interval)
