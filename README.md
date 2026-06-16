# face-camera-gpu

Sistema de reconhecimento facial em tempo real usando GPU local.
Conecta em câmeras IP Intelbras via RTSP, detecta e reconhece rostos comparando com uma base de imagens local.

**Sem banco de dados. Sem API. Sem internet em produção. Câmera → GPU → Tela.**

---

## Demonstração

```
FPS: 30.9                                     16-06-2026 11:04:15
GPU: NVIDIA GeForce GTX 1650 [DirectML]

         ┌──────────────────┐
         │  João Silva      │  ← caixa verde = reconhecido
         │    (0.87)        │
         └──────────────────┘

         ┌──────────────────┐
         │  Desconhecido    │  ← caixa vermelha = não encontrado
         └──────────────────┘
```

---

## Requisitos de Máquina

### Obrigatório
| Item | Mínimo |
|------|--------|
| Python | 3.10 ou superior |
| GPU NVIDIA | Qualquer (GTX 1050+, RTX, Quadro) |
| Driver NVIDIA | 472+ (para DirectML) |
| RAM | 8 GB |
| Windows | 10 ou 11 (64-bit) |

### Testado em
- GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
- Python: 3.14.0
- Windows 11 Pro
- onnxruntime-directml 1.24.4 + InsightFace 1.0.1

### Não é necessário instalar
- CUDA Toolkit (o sistema usa DirectML, nativo do Windows)
- cuDNN
- Anaconda / conda
- Docker

---

## Instalação em Máquina Nova

```bash
# 1. Clonar o repositório
git clone https://github.com/geanusksama/face-camera-gpu.git
cd face-camera-gpu

# 2. Instalar dependências Python
pip install -r requirements.txt

# 3. Copiar e editar configuração
copy .env.example .env
# Editar .env com IP, usuário e senha da câmera

# 4. Executar (câmera única)
python app/main.py

# 5. Executar (múltiplas câmeras)
python multi_main.py
```

> O modelo InsightFace `buffalo_l` (~350MB) é baixado automaticamente na primeira execução.
> Salvo em `C:\Users\<usuario>\.insightface\models\buffalo_l\`

---

## Configuração (.env)

```env
# ── Câmera única ──────────────────────────────────────────
CAMERA_IP=192.168.0.10
CAMERA_USER=admin
CAMERA_PASSWORD=sua_senha
# @ na senha vira %40 na URL
CAMERA_RTSP_URL=rtsp://admin:sua_senha@192.168.0.10:554/cam/realmonitor?channel=1&subtype=0

# ── GPU ───────────────────────────────────────────────────
GPU_REQUIRED=true         # true = encerra se não houver GPU
USE_CUDA=true

# ── Performance ───────────────────────────────────────────
FRAME_SKIP=3              # processa 1 a cada N frames
DETECTION_CONFIDENCE=0.65 # 0.0–1.0 (mais alto = menos falsos positivos)
RECOGNITION_THRESHOLD=0.55 # similaridade mínima para reconhecer
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720

# ── Base de rostos ────────────────────────────────────────
# Modo pasta (recomendado para produção):
#   known_faces/nome_pessoa/foto1.jpg
USE_FLAT_FACES=false
KNOWN_FACES_DIR=./known_faces

# Modo arquivo plano (para teste com fotosFaceID):
#   fotosFaceID/user_images/123.jpg → ID: 123
# USE_FLAT_FACES=true
FLAT_FACES_DIR=../fotosFaceID/user_images

# ── Cache ─────────────────────────────────────────────────
EMBEDDINGS_CACHE=./embeddings_cache.pkl
```

---

## Como Adicionar Pessoas

### Modo Produção (por nome)
```
known_faces/
  joao_silva/
    foto_frente.jpg
    foto_lateral.jpg
  maria_souza/
    foto1.jpg
```

Depois regenerar o cache:
```bash
python rebuild_cache.py
```

### Modo Teste (pasta fotosFaceID)
```bash
# No .env:
USE_FLAT_FACES=true

# Regenerar cache:
python rebuild_cache.py --flat
```

---

## Múltiplas Câmeras

### Configuração no .env

```env
CAMERA_COUNT=4

CAMERA_1_NAME=Entrada Principal
CAMERA_1_RTSP=rtsp://admin:senha@192.168.0.10:554/cam/realmonitor?channel=1&subtype=0

CAMERA_2_NAME=Recepcao
CAMERA_2_RTSP=rtsp://admin:senha@192.168.0.11:554/cam/realmonitor?channel=1&subtype=0

CAMERA_3_NAME=Corredor
CAMERA_3_RTSP=rtsp://admin:senha@192.168.0.12:554/cam/realmonitor?channel=1&subtype=0

CAMERA_4_NAME=Estacionamento
CAMERA_4_RTSP=rtsp://admin:senha@192.168.0.13:554/cam/realmonitor?channel=1&subtype=0
```

### Executar múltiplas câmeras
```bash
python multi_main.py
```

Abre uma janela única com grade 2×2 mostrando todas as câmeras simultaneamente.

---

## Estrutura do Projeto

```
face-camera-gpu/
├── app/
│   ├── main.py             ← câmera única
│   ├── config.py           ← lê .env
│   ├── gpu_check.py        ← detecta GPU (CUDA ou DirectML)
│   ├── camera_service.py   ← RTSP + reconexão automática
│   ├── image_database.py   ← carrega fotos e gera embeddings
│   └── face_recognizer.py  ← compara embeddings (ArcFace)
├── multi_main.py           ← múltiplas câmeras em grade 2×2
├── rebuild_cache.py        ← regenera embeddings_cache.pkl
├── known_faces/            ← base de rostos (nome_pessoa/foto.jpg)
├── docs/
│   ├── MCP.md              ← contexto completo do projeto
│   ├── SKILL_SPEC.md       ← especificação de cada skill
│   ├── GPU_SETUP.md        ← configuração de GPU
│   └── CAMERA_INTELBRAS.md ← configuração da câmera
├── .env                    ← configuração local (NÃO vai pro git)
├── .env.example            ← template de configuração
├── requirements.txt        ← dependências Python
└── .gitignore
```

---

## Pipeline de Funcionamento

```
Iniciar
  │
  ├─ 1. Verificar GPU (DirectML ou CUDA) → erro se não houver
  ├─ 2. Carregar InsightFace buffalo_l na GPU
  ├─ 3. Carregar base de rostos (known_faces ou fotosFaceID)
  ├─ 4. Gerar embeddings ArcFace (512 dims por rosto)
  ├─ 5. Salvar cache em embeddings_cache.pkl
  └─ 6. Conectar câmera via RTSP
         │
         └─ Loop:
              ├─ Capturar frame
              ├─ A cada FRAME_SKIP frames:
              │    ├─ Detectar rostos (RetinaFace na GPU)
              │    ├─ Gerar embedding do rosto (ArcFace na GPU)
              │    └─ Comparar com base (cosine similarity)
              └─ Exibir: caixa + nome + confiança + FPS + GPU
```

---

## Troubleshooting

| Problema | Causa | Solução |
|----------|-------|---------|
| "GPU/CUDA não disponível" | Driver NVIDIA antigo ou ausente | Atualizar driver NVIDIA |
| Câmera não conecta | IP/senha errado ou porta bloqueada | `ping 192.168.0.10` e verificar .env |
| @ na senha não funciona | Caractere especial na URL | Usar `%40` no lugar de `@` na `CAMERA_RTSP_URL` |
| Lento na 1ª execução | Gerando cache de embeddings | Normal — próximas execuções são instantâneas |
| Modelo não baixa | Sem internet | Copiar pasta `~/.insightface/models/buffalo_l/` de outra máquina |
| Reconhecimento impreciso | Threshold muito baixo | Aumentar `RECOGNITION_THRESHOLD` para 0.65+ |

---

## Levar para Outra Máquina

```bash
# Na máquina nova:
git clone https://github.com/geanusksama/face-camera-gpu.git
cd face-camera-gpu
pip install -r requirements.txt
copy .env.example .env
# Editar .env
python app/main.py
```

> Se não tiver internet na máquina nova, copie a pasta:
> `C:\Users\<usuario>\.insightface\models\buffalo_l\`
> da máquina atual para o mesmo caminho na nova.

---

## Tecnologias

| Tecnologia | Função |
|------------|--------|
| InsightFace 1.0.1 | Detecção (RetinaFace) + Reconhecimento (ArcFace) |
| ONNX Runtime DirectML | Inferência na GPU (Windows nativo) |
| OpenCV 4.13 | Captura RTSP + exibição |
| NumPy | Cálculo de similaridade (produto interno) |
| python-dotenv | Configuração via .env |
