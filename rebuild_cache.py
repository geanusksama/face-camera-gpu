"""
Regenera o cache de embeddings.
Use após adicionar/remover pessoas da base de imagens.

Uso:
  python rebuild_cache.py                  # usa configuração do .env
  python rebuild_cache.py --flat           # força modo plano (fotosFaceID)
  python rebuild_cache.py --known          # força modo subpastas (known_faces)
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

from config import (
    KNOWN_FACES_DIR, FLAT_FACES_DIR, USE_FLAT_FACES, EMBEDDINGS_CACHE, GPU_REQUIRED
)
from gpu_check import check_gpu
from image_database import ImageDatabase


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flat',  action='store_true', help='Força modo plano (fotosFaceID)')
    parser.add_argument('--known', action='store_true', help='Força modo subpastas (known_faces)')
    args = parser.parse_args()

    use_flat = USE_FLAT_FACES
    if args.flat:
        use_flat = True
    elif args.known:
        use_flat = False

    if GPU_REQUIRED:
        check_gpu()

    print("Inicializando InsightFace...")
    from insightface.app import FaceAnalysis
    from gpu_check import get_best_providers
    face_app = FaceAnalysis(name='buffalo_l', providers=get_best_providers())
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    known_dir = _resolve(KNOWN_FACES_DIR)
    flat_dir  = _resolve(FLAT_FACES_DIR) if FLAT_FACES_DIR else None
    cache     = _resolve(EMBEDDINGS_CACHE)

    if os.path.exists(cache):
        os.remove(cache)
        print(f"Cache anterior removido: {cache}")

    db = ImageDatabase(
        face_app=face_app,
        known_faces_dir=known_dir,
        flat_faces_dir=flat_dir,
        use_flat=use_flat,
        cache_path=cache
    )
    db.load(force_rebuild=True)
    print("Rebuild concluído.")


if __name__ == '__main__':
    main()
