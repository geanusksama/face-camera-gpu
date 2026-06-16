# GPU Setup — face-camera-gpu

## Hardware Detectado
- **GPU:** NVIDIA GeForce GTX 1650
- **VRAM:** 4 GB
- **Driver:** 596.08
- **Provider ativo:** DirectML (nativo Windows 10/11)

## Stack de Inferência
Este projeto usa **ONNX Runtime DirectML** com **InsightFace**.
DirectML é a API de GPU do Windows 10/11, funciona com qualquer GPU (NVIDIA, AMD, Intel)
sem precisar instalar o CUDA Toolkit separadamente.

O `gpu_check.py` detecta automaticamente o melhor provider disponível:
1. `CUDAExecutionProvider` — se CUDA Toolkit 12.x instalado
2. `DmlExecutionProvider` — DirectML (Windows nativo, recomendado)
3. CPU → encerra com erro (quando `GPU_REQUIRED=true`)

## Verificação Rápida
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Esperado: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

python -c "import subprocess; print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)"
```

## Instalação (caso precisar reinstalar)
```bash
pip install onnxruntime-gpu>=1.27.0
pip install insightface>=1.0.1
pip install opencv-python>=4.13.0
pip install python-dotenv numpy scipy scikit-image
```

## Modelos InsightFace (download automático no 1º uso)
O modelo `buffalo_l` é baixado automaticamente para:
```
C:\Users\<usuario>\.insightface\models\buffalo_l\
```
Arquivos (~500MB):
- `1k3d68.onnx` — 3D landmarks
- `2d106det.onnx` — detector 2D
- `det_10g.onnx` — RetinaFace detection
- `genderage.onnx` — gênero/idade
- `w600k_r50.onnx` — ArcFace recognition

## Troubleshooting GPU
| Problema | Solução |
|----------|---------|
| `CUDAExecutionProvider` ausente | `pip install --upgrade onnxruntime-gpu` |
| CUDA out of memory | Reduzir `VIDEO_WIDTH/HEIGHT` ou aumentar `FRAME_SKIP` |
| GPU não detectada | Verificar driver com `nvidia-smi` |
| Lento mesmo com GPU | Verificar se `ctx_id=0` está configurado no InsightFace |
