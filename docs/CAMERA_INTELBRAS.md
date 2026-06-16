# Camera Intelbras — Configuração RTSP

## Dados da Câmera
- **IP:** 192.168.0.10
- **Marca:** Intelbras
- **Protocolo:** RTSP

## URL RTSP Padrão Intelbras
```
rtsp://USUARIO:SENHA@192.168.0.10:554/cam/realmonitor?channel=1&subtype=0
```

Parâmetros:
| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| channel   | 1     | Canal principal (câmera 1) |
| subtype   | 0     | Stream principal (alta qualidade) |
| subtype   | 1     | Sub-stream (menor resolução, mais rápido) |

## Configuração no .env
```env
CAMERA_IP=192.168.0.10
CAMERA_USER=admin
CAMERA_PASSWORD=SUA_SENHA_AQUI
CAMERA_RTSP_URL=rtsp://admin:SUA_SENHA_AQUI@192.168.0.10:554/cam/realmonitor?channel=1&subtype=0
```

## Teste de Conectividade
```bash
# 1. Ping
ping 192.168.0.10

# 2. Stream com ffplay (se instalado)
ffplay "rtsp://admin:SENHA@192.168.0.10:554/cam/realmonitor?channel=1&subtype=0"

# 3. Teste Python
python -c "
import cv2
url = 'rtsp://admin:SENHA@192.168.0.10:554/cam/realmonitor?channel=1&subtype=0'
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
print('Conectado:', cap.isOpened())
cap.release()
"
```

## Sub-stream (menor latência)
Para menor resolução e maior FPS, use `subtype=1`:
```env
CAMERA_RTSP_URL=rtsp://admin:SENHA@192.168.0.10:554/cam/realmonitor?channel=1&subtype=1
VIDEO_WIDTH=704
VIDEO_HEIGHT=576
```

## Troubleshooting
| Problema | Causa provável | Solução |
|----------|---------------|---------|
| cap.isOpened() = False | Senha errada ou câmera offline | Verificar ping e credenciais |
| Frame congelado | Pacotes perdidos na rede | Usar subtype=1 (menor bitrate) |
| Alta latência | Buffer grande | CAP_PROP_BUFFERSIZE=1 já configurado |
| Reconexão constante | Câmera reiniciando | Verificar alimentação da câmera |
