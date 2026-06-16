import cv2
import time


class CameraService:
    def __init__(self, rtsp_url: str, width: int = 1280, height: int = 720, reconnect_delay: int = 5):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.cap = None

    def connect(self):
        print(f"Conectando na câmera: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise ConnectionError(f"Falha ao conectar na câmera: {self.rtsp_url}")
        print("Câmera conectada com sucesso.")

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def reconnect(self):
        print(f"Câmera desconectada. Reconectando em {self.reconnect_delay}s...")
        self.release()
        time.sleep(self.reconnect_delay)
        self.connect()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
