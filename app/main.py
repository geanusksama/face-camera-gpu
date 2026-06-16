import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from config import (
    GPU_REQUIRED, CAMERA_RTSP_URL, VIDEO_WIDTH, VIDEO_HEIGHT,
    FRAME_SKIP, DETECTION_CONFIDENCE, RECOGNITION_THRESHOLD,
    KNOWN_FACES_DIR, FLAT_FACES_DIR, USE_FLAT_FACES, EMBEDDINGS_CACHE
)
from gpu_check import check_gpu, get_best_providers
from camera_service import CameraService
from image_database import ImageDatabase
from face_recognizer import FaceRecognizer


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(base, path))


def draw_overlay(frame, faces, results, fps: float, gpu_name: str):
    for face, (name, conf) in zip(faces, results):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        color = (0, 220, 0) if name else (0, 0, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({conf:.2f})" if name else "Desconhecido"
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"GPU: {gpu_name}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


def main():
    # 1. Verificar GPU/CUDA
    if GPU_REQUIRED:
        gpu_name = check_gpu()
    else:
        gpu_name = "CPU (GPU não exigida)"

    # 2. Inicializar InsightFace com GPU
    print("Inicializando InsightFace (buffalo_l) na GPU...")
    from insightface.app import FaceAnalysis
    providers = get_best_providers()
    face_app = FaceAnalysis(name='buffalo_l', providers=providers)
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace inicializado.")

    # 3. Carregar base de imagens conhecidas
    known_dir = _resolve_path(KNOWN_FACES_DIR)
    flat_dir  = _resolve_path(FLAT_FACES_DIR) if FLAT_FACES_DIR else None
    cache     = _resolve_path(EMBEDDINGS_CACHE)

    db = ImageDatabase(
        face_app=face_app,
        known_faces_dir=known_dir,
        flat_faces_dir=flat_dir,
        use_flat=USE_FLAT_FACES,
        cache_path=cache
    )
    db.load()

    recognizer = FaceRecognizer(db, threshold=RECOGNITION_THRESHOLD)

    # 4. Conectar câmera
    camera = CameraService(CAMERA_RTSP_URL, VIDEO_WIDTH, VIDEO_HEIGHT)
    camera.connect()

    # 5. Loop principal
    frame_count  = 0
    fps          = 0.0
    t0           = time.time()
    last_faces   = []
    last_results = []

    WIN_NAME = "Face Recognition - GPU"
    WIN_W, WIN_H = 640, 360
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    print("Sistema iniciado. Pressione ESC para encerrar.")

    while True:
        ret, frame = camera.read_frame()
        if not ret:
            camera.reconnect()
            continue

        frame_count += 1

        # 6. Detectar e reconhecer a cada FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            detected = face_app.get(frame)
            last_faces   = [f for f in detected if f.det_score >= DETECTION_CONFIDENCE]
            last_results = [recognizer.recognize(f) for f in last_faces]

        # 7. Calcular FPS
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps         = frame_count / elapsed
            frame_count = 0
            t0          = time.time()

        # 8. Exibir resultado
        display = draw_overlay(frame.copy(), last_faces, last_results, fps, gpu_name)
        cv2.imshow(WIN_NAME, display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Sistema encerrado.")


if __name__ == "__main__":
    main()
