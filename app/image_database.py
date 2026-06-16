import os
import pickle
from pathlib import Path

import cv2
import numpy as np


class ImageDatabase:
    def __init__(self, face_app, known_faces_dir: str, flat_faces_dir: str = None,
                 use_flat: bool = False, cache_path: str = None):
        self.face_app = face_app
        self.known_faces_dir = known_faces_dir
        self.flat_faces_dir = flat_faces_dir
        self.use_flat = use_flat
        self.cache_path = cache_path
        self.all_embeddings: np.ndarray = None
        self.all_names: list = []

    def load(self, force_rebuild: bool = False):
        if not force_rebuild and self.cache_path and os.path.exists(self.cache_path):
            self._load_cache()
            return
        self._generate_embeddings()
        if self.cache_path and len(self.all_names) > 0:
            self._save_cache()

    def _load_cache(self):
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        self.all_embeddings = data['embeddings']
        self.all_names = data['names']
        n_people = len(set(self.all_names))
        print(f"Cache carregado: {len(self.all_names)} embeddings de {n_people} pessoas")

    def _save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump({'embeddings': self.all_embeddings, 'names': self.all_names}, f)
        print(f"Cache salvo em: {self.cache_path}")

    def _generate_embeddings(self):
        all_embs = []
        all_names = []

        if self.use_flat and self.flat_faces_dir and os.path.exists(self.flat_faces_dir):
            print(f"Carregando imagens planas de: {self.flat_faces_dir}")
            exts = ('.jpg', '.jpeg', '.png', '.bmp')
            images = [p for p in Path(self.flat_faces_dir).iterdir() if p.suffix.lower() in exts]
            total = len(images)
            print(f"Total de imagens encontradas: {total}")
            for i, img_path in enumerate(images):
                name = img_path.stem
                emb = self._embed_image(str(img_path))
                if emb is not None:
                    all_embs.append(emb)
                    all_names.append(name)
                if (i + 1) % 100 == 0:
                    print(f"  Processadas: {i+1}/{total}  ({len(all_embs)} com rosto)")
        else:
            print(f"Carregando imagens de: {self.known_faces_dir}")
            base = Path(self.known_faces_dir)
            if not base.exists():
                print(f"AVISO: Pasta não encontrada: {self.known_faces_dir}")
            else:
                for person_dir in sorted(base.iterdir()):
                    if not person_dir.is_dir():
                        continue
                    name = person_dir.name
                    for img_path in person_dir.iterdir():
                        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                            emb = self._embed_image(str(img_path))
                            if emb is not None:
                                all_embs.append(emb)
                                all_names.append(name)

        if all_embs:
            self.all_embeddings = np.array(all_embs, dtype=np.float32)
            self.all_names = all_names
            print(f"Base carregada: {len(all_names)} embeddings de {len(set(all_names))} pessoas")
        else:
            print("AVISO: Nenhum embedding gerado. Verifique a pasta de imagens.")
            self.all_embeddings = np.zeros((0, 512), dtype=np.float32)
            self.all_names = []

    def _embed_image(self, img_path: str):
        img = cv2.imread(img_path)
        if img is None:
            return None
        faces = self.face_app.get(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        norm = np.linalg.norm(face.embedding)
        if norm == 0:
            return None
        return face.embedding / norm

    def find_match(self, embedding: np.ndarray, threshold: float = 0.55):
        if self.all_embeddings is None or len(self.all_embeddings) == 0:
            return None, 0.0
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None, 0.0
        emb_norm = embedding / norm
        similarities = self.all_embeddings @ emb_norm
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        if best_sim >= threshold:
            return self.all_names[best_idx], best_sim
        return None, best_sim
