import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

CAMERA_IP       = os.getenv("CAMERA_IP", "192.168.0.10")
CAMERA_USER     = os.getenv("CAMERA_USER", "admin")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "")
CAMERA_RTSP_URL = os.getenv(
    "CAMERA_RTSP_URL",
    f"rtsp://{os.getenv('CAMERA_USER','admin')}:{os.getenv('CAMERA_PASSWORD','')}@{os.getenv('CAMERA_IP','192.168.0.10')}:554/cam/realmonitor?channel=1&subtype=0"
)

GPU_REQUIRED           = os.getenv("GPU_REQUIRED", "true").lower() == "true"
USE_CUDA               = os.getenv("USE_CUDA", "true").lower() == "true"
FRAME_SKIP             = int(os.getenv("FRAME_SKIP", "3"))
DETECTION_CONFIDENCE   = float(os.getenv("DETECTION_CONFIDENCE", "0.65"))
RECOGNITION_THRESHOLD  = float(os.getenv("RECOGNITION_THRESHOLD", "0.55"))
VIDEO_WIDTH            = int(os.getenv("VIDEO_WIDTH", "1280"))
VIDEO_HEIGHT           = int(os.getenv("VIDEO_HEIGHT", "720"))

KNOWN_FACES_DIR   = os.getenv("KNOWN_FACES_DIR", "./known_faces")
FLAT_FACES_DIR    = os.getenv("FLAT_FACES_DIR", "")
USE_FLAT_FACES    = os.getenv("USE_FLAT_FACES", "false").lower() == "true"
EMBEDDINGS_CACHE  = os.getenv("EMBEDDINGS_CACHE", "./embeddings_cache.pkl")
