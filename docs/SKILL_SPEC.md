# SKILL_SPEC — face-camera-gpu

## Skill 1: camera_intelbras_rtsp
**Arquivo:** `app/camera_service.py` — `CameraService`

- Conecta na câmera via `cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)`
- URL padrão: `rtsp://admin:SENHA@192.168.0.10:554/cam/realmonitor?channel=1&subtype=0`
- Buffer reduzido (CAP_PROP_BUFFERSIZE=1) para baixa latência
- Método `reconnect()`: libera, aguarda `reconnect_delay` segundos e reconecta
- Método `read_frame()` → `(ret, frame)`

---

## Skill 2: gpu_required_inference
**Arquivo:** `app/gpu_check.py` — `check_gpu()`

- Verifica `onnxruntime.get_available_providers()`
- Se `CUDAExecutionProvider` ausente → `sys.exit(1)` com mensagem clara
- Consulta `nvidia-smi` para exibir nome real da GPU
- Retorna `gpu_name: str` para exibição no overlay

---

## Skill 3: local_image_database
**Arquivo:** `app/image_database.py` — `ImageDatabase`

**Modo subpasta (padrão):**
```
known_faces/
  nome_pessoa/
    foto1.jpg
    foto2.jpg
```

**Modo arquivo plano (USE_FLAT_FACES=true):**
```
fotosFaceID/user_images/
  67.jpg   → ID: 67
  85.jpg   → ID: 85
```

- `load()`: carrega do cache `.pkl` se existir, senão gera embeddings
- `_embed_image()`: usa InsightFace para extrair embedding normalizado (L2)
- `find_match()`: produto interno (cosine similarity) contra todos os embeddings; retorna nome + confiança
- Cache salvo em `embeddings_cache.pkl` para reutilização

---

## Skill 4: face_detection
**Executado em:** `app/main.py` → `face_app.get(frame)`

- Modelo: InsightFace `buffalo_l` (detector RetinaFace + reconhecedor ArcFace)
- Provider: `CUDAExecutionProvider` (GPU obrigatória)
- Tamanho de detecção: 640×640
- Filtro por `det_score >= DETECTION_CONFIDENCE` (padrão 0.65)
- Executa a cada `FRAME_SKIP` frames para controlar uso de GPU

---

## Skill 5: face_recognition
**Arquivo:** `app/face_recognizer.py` — `FaceRecognizer`

- Recebe objeto `face` do InsightFace (contém `face.embedding` de 512 dims)
- Delega `find_match()` para `ImageDatabase`
- Threshold padrão: 0.55 (similaridade coseno)
- Retorna `(nome: str | None, confiança: float)`

---

## Skill 6: realtime_display
**Função:** `draw_overlay()` em `app/main.py`

- Caixa verde para pessoa reconhecida, vermelha para desconhecido
- Texto: `Nome (0.87)` ou `Desconhecido`
- Overlay superior: `FPS: 24.3` e `GPU: NVIDIA GeForce GTX 1650`
- Janela: `cv2.imshow("Face Recognition - GPU", frame)`
- ESC para encerrar

---

## Skill 7: performance_control
**Configurado em:** `.env`

| Variável             | Padrão | Descrição                              |
|----------------------|--------|----------------------------------------|
| FRAME_SKIP           | 3      | Processa 1 a cada N frames             |
| DETECTION_CONFIDENCE | 0.65   | Score mínimo para aceitar detecção     |
| RECOGNITION_THRESHOLD| 0.55   | Similaridade mínima para reconhecer    |
| VIDEO_WIDTH          | 1280   | Largura do frame                       |
| VIDEO_HEIGHT         | 720    | Altura do frame                        |

- InsightFace roda na GPU (GTX 1650 — 4GB VRAM)
- Buffer de câmera = 1 (evita acúmulo de frames)
- Último resultado de detecção reutilizado nos frames pulados
