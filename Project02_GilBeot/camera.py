"""
카메라 입력 추상화

--source 값에 따라:
  v4l2           : USB 카메라 /dev/video0, GStreamer v4l2src 640x480 YUY2 (Jetson 기본값)
  v4l2:1         : /dev/video1
  csi / csi:N    : CSI 카메라 (nvarguscamerasrc, 06 노트북 파이프라인)
  0, 1, ...      : OpenCV 기본 백엔드 웹캠 (노트북 테스트용)
  path/to/x.mp4  : 영상 파일 (끝나면 처음부터 반복 → 시연 백업용)
"""

import platform
from pathlib import Path

import cv2

import config


class Camera:
    def __init__(self, source: str):
        self.source = source
        self.is_file = False
        self.cap = self._open(source)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                f"카메라를 열 수 없습니다: source={source}\n"
                "  - USB 카메라 번호 확인: ls /dev/video*  →  --source v4l2:N\n"
                "  - 다른 프로그램이 카메라를 쓰고 있지 않은지 확인"
            )

    @staticmethod
    def _index(source):
        return int(source.split(":")[1]) if ":" in source else 0

    def _open(self, source: str):
        if source.startswith("v4l2"):
            device = self._index(source)
            print(f"[Camera] USB /dev/video{device} (GStreamer v4l2src "
                  f"{config.USB_WIDTH}x{config.USB_HEIGHT})")
            return cv2.VideoCapture(config.v4l2_pipeline(device), cv2.CAP_GSTREAMER)

        if source.startswith("csi"):
            sensor_id = self._index(source)
            print(f"[Camera] CSI sensor-id={sensor_id} (GStreamer nvarguscamerasrc)")
            return cv2.VideoCapture(config.csi_pipeline(sensor_id), cv2.CAP_GSTREAMER)

        if source.isdigit():
            index = int(source)
            print(f"[Camera] Webcam index={index}")
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            elif platform.system() == "Linux":
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.USB_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.USB_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"영상 파일이 없습니다: {path}")
        print(f"[Camera] Video file: {path}")
        self.is_file = True
        return cv2.VideoCapture(str(path))

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.is_file:
            # 영상 파일이면 처음으로 되감기
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap:
            self.cap.release()


def default_source() -> str:
    # Jetson: USB 카메라 GStreamer 파이프라인, 그 외(노트북): 웹캠 0번
    return "v4l2" if config.IS_JETSON else "0"
