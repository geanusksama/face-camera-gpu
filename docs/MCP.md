# MCP — Contexto do Projeto face-camera-gpu

## 1. Contexto
Sistema de reconhecimento facial em tempo real usando GPU local.
Sem banco de dados, sem API REST, sem dashboard — apenas câmera → reconhecimento → tela.

## 2. Objetivo
Conectar na câmera Intelbras (IP 192.168.0.10 via RTSP), detectar e reconhecer
rostos em tempo real comparando com uma base de imagens locais. Exibir resultado
na tela com nome da pessoa e nível de confiança.

## 3. Limites Atuais
- Sem salvamento de presenças
- Sem banco de dados (PostgreSQL, Supabase etc.)
- Sem API REST
- Sem painel web
- Sem cadastro por tela
- Sem autenticação/login
- Sem fallback silencioso para CPU

## 4. Fluxo do Reconhecimento
```
Iniciar → Verificar GPU → Carregar Base → Gerar Embeddings → Conectar Câmera
   ↓
Capturar Frame (RTSP) → Detectar Rostos (GPU) → Gerar Embedding (GPU)
   ↓
Comparar com Base → Exibir Nome + Confiança + FPS na Tela
```

## 5. Estrutura de Pastas
```
face-camera-gpu/
  app/
    main.py             ← entrada principal
    config.py           ← lê .env
    gpu_check.py        ← verifica CUDA
    camera_service.py   ← conexão RTSP + reconexão
    image_database.py   ← carrega fotos e gera embeddings
    face_recognizer.py  ← compara embeddings
  known_faces/
    nome_pessoa/
      foto1.jpg
      foto2.jpg
  docs/
    MCP.md
    SKILL_SPEC.md
    GPU_SETUP.md
    CAMERA_INTELBRAS.md
  .env
  .env.example
  requirements.txt
```

## 6. Regras de GPU Obrigatória
- GPU_REQUIRED=true no .env
- Se CUDAExecutionProvider não disponível → sys.exit(1)
- Mensagem: "GPU/CUDA não disponível. Reconhecimento não iniciado."
- InsightFace inicializado com ctx_id=0 (GPU 0)
- Provider ordem: CUDAExecutionProvider → CPUExecutionProvider (fallback apenas para ops não suportadas)

## 7. Como Cadastrar uma Pessoa
1. Crie uma subpasta em `known_faces/` com o nome da pessoa:
   ```
   known_faces/
     joao_silva/
       foto_frente.jpg
       foto_lateral.jpg
   ```
2. Delete o arquivo `embeddings_cache.pkl` (ou rode `rebuild_cache.py`)
3. Reinicie o sistema — os embeddings serão regenerados

### Modo Base Plana (fotosFaceID)
Alternativamente, usar imagens numeradas (IDs de sistema de ponto):
- Defina `USE_FLAT_FACES=true` e `FLAT_FACES_DIR=../fotosFaceID/user_images` no .env
- O sistema reconhece cada pessoa pelo número do arquivo (ex: `67.jpg` → ID `67`)

## 8. Como Testar a Câmera Intelbras
```bash
# Teste de conectividade básica
ping 192.168.0.10

# Teste de stream RTSP com ffplay
ffplay rtsp://admin:SENHA@192.168.0.10:554/cam/realmonitor?channel=1^&subtype=0

# Teste com Python
python -c "import cv2; cap=cv2.VideoCapture('rtsp://admin:SENHA@192.168.0.10:554/cam/realmonitor?channel=1&subtype=0', cv2.CAP_FFMPEG); print(cap.isOpened())"
```

## 9. Como Executar o Projeto
```bash
# Instalar dependências (uma vez)
pip install -r requirements.txt

# Configurar câmera e senha
# Editar face-camera-gpu/.env

# Executar
python app/main.py

# Rebuild de embeddings (após adicionar pessoas)
python rebuild_cache.py
```

## 10. Critérios de Aceite
- [ ] `python app/main.py` verifica GPU e exibe nome da GPU
- [ ] Se não houver GPU, exibe erro e encerra
- [ ] Carrega imagens da pasta `known_faces` (ou `fotosFaceID`)
- [ ] Conecta na câmera 192.168.0.10
- [ ] Exibe vídeo ao vivo com caixa nos rostos
- [ ] Exibe nome da pessoa reconhecida com confiança
- [ ] Exibe "Desconhecido" quando não reconhece
- [ ] Exibe FPS e status da GPU na tela
- [ ] Não grava nada em banco
- [ ] Reconecta automaticamente se câmera cair
