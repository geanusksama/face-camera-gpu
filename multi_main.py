"""
Múltiplas câmeras em grade 2×2.
Configure CAMERA_COUNT e CAMERA_N_RTSP no .env e execute:
  python multi_main.py
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from gpu_check import check_gpu, get_best_providers
from image_database import ImageDatabase
from face_recognizer import FaceRecognizer
from config import (
    GPU_REQUIRED, FRAME_SKIP, DETECTION_CONFIDENCE, RECOGNITION_THRESHOLD,
    KNOWN_FACES_DIR, FLAT_FACES_DIR, USE_FLAT_FACES, EMBEDDINGS_CACHE
)


CELL_W = int(os.getenv("MULTI_CELL_W", "640"))
CELL_H = int(os.getenv("MULTI_CELL_H", "360"))


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))


def load_cameras() -> list[dict]:
    count = int(os.getenv("CAMERA_COUNT", "1"))
    cameras = []
    for i in range(1, count + 1):
        rtsp = os.getenv(f"CAMERA_{i}_RTSP", "")
        name = os.getenv(f"CAMERA_{i}_NAME", f"Camera {i}")
        if rtsp:
            cameras.append({"id": i, "name": name, "rtsp": rtsp})
    if not cameras:
        # fallback para câmera única do .env
        rtsp = os.getenv("CAMERA_RTSP_URL", "")
        name = os.getenv("CAMERA_1_NAME", "Camera 1")
        cameras.append({"id": 1, "name": name, "rtsp": rtsp})
    return cameras


class CameraWorker(threading.Thread):
    def __init__(self, cam_info: dict, face_app, recognizer, lock: threading.Lock):
        super().__init__(daemon=True)
        self.cam_info  = cam_info
        self.face_app  = face_app
        self.recognizer = recognizer
        self.lock      = lock
        self.frame     = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        self.fps       = 0.0
        self._running  = True

    def run(self):
        import cv2 as _cv2
        rtsp = self.cam_info["rtsp"]
        name = self.cam_info["name"]
        reconnect_delay = 5

        while self._running:
            cap = _cv2.VideoCapture(rtsp, _cv2.CAP_FFMPEG)
            cap.set(_cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print(f"[{name}] Falha ao conectar. Tentando em {reconnect_delay}s...")
                time.sleep(reconnect_delay)
                continue

            print(f"[{name}] Conectado.")
            frame_count = 0
            t0 = time.time()
            last_faces, last_results = [], []

            while self._running:
                ret, raw = cap.read()
                if not ret:
                    print(f"[{name}] Sem frame. Reconectando...")
                    break

                frame_count += 1
                if frame_count % FRAME_SKIP == 0:
                    with self.lock:
                        detected    = self.face_app.get(raw)
                    last_faces   = [f for f in detected if f.det_score >= DETECTION_CONFIDENCE]
                    last_results = [self.recognizer.recognize(f) for f in last_faces]

                elapsed = time.time() - t0
                if elapsed >= 1.0:
                    self.fps  = frame_count / elapsed
                    frame_count = 0
                    t0          = time.time()

                display = self._draw(raw.copy(), last_faces, last_results, name)
                display = _cv2.resize(display, (CELL_W, CELL_H))
                self.frame = display

            cap.release()
            time.sleep(reconnect_delay)

    def _draw(self, frame, faces, results, cam_name):
        for face, (person, conf) in zip(faces, results):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            color = (0, 220, 0) if person else (0, 0, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{person} ({conf:.2f})" if person else "Desconhecido"
            cv2.putText(frame, label, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{cam_name}  FPS:{self.fps:.1f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        return frame

    def stop(self):
        self._running = False


def build_grid(frames: list) -> np.ndarray:
    n = len(frames)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    blank = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    cells = frames + [blank] * (rows * cols - n)
    grid_rows = []
    for r in range(rows):
        row_frames = cells[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_frames))
    return np.vstack(grid_rows)


def main():
    if GPU_REQUIRED:
        gpu_name = check_gpu()
    else:
        gpu_name = "CPU"

    print("Inicializando InsightFace...")
    from insightface.app import FaceAnalysis
    providers = get_best_providers()
    face_app  = FaceAnalysis(name='buffalo_l', providers=providers)
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    known_dir = _resolve(KNOWN_FACES_DIR)
    flat_dir  = _resolve(FLAT_FACES_DIR) if FLAT_FACES_DIR else None
    cache     = _resolve(EMBEDDINGS_CACHE)

    db = ImageDatabase(face_app, known_dir, flat_dir, USE_FLAT_FACES, cache)
    db.load()
    recognizer = FaceRecognizer(db, RECOGNITION_THRESHOLD)

    cameras = load_cameras()
    print(f"{len(cameras)} câmera(s) configurada(s): {[c['name'] for c in cameras]}")

    lock    = threading.Lock()
    workers = [CameraWorker(c, face_app, recognizer, lock) for c in cameras]
    for w in workers:
        w.start()

    win = "Face Recognition - Multi GPU"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CELL_W * (2 if len(cameras) > 1 else 1),
                          CELL_H * ((len(cameras) + 1) // 2))

    print("Grade iniciada. Pressione ESC para encerrar.")
    while True:
        frames = [w.frame for w in workers]
        grid   = build_grid(frames)
        cv2.putText(grid, f"GPU: {gpu_name}", (8, grid.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, grid)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    for w in workers:
        w.stop()
    cv2.destroyAllWindows()
    print("Encerrado.")


if __name__ == "__main__":
    main()
