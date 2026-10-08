# Dockerfile — LA VOIE DES MASQUES DE RECODANSE, SEULE.
#
# Elle reconstruit l'environnement dans lequel `scripts/masques_for_file.py` tourne en
# production : même image de base, mêmes versions de Python et des dépendances, mêmes
# modèles vérifiés par empreinte, mêmes variables d'environnement. En production, ce
# programme vit dans une image plus large qui porte aussi d'autres traitements (le tempo,
# la transcription, la classification) et le binaire qui les lance : ce sont des
# programmes séparés, qui n'importent pas ultralytics — voir README.md.
#
#   docker build -t masques .
#   docker run --rm --network none -v "$PWD/videos:/videos" masques /videos/cours.mp4 --out /videos/sortie
#
# L'image tourne SANS réseau : les modèles y sont embarqués, et YOLO_OFFLINE empêche
# ultralytics d'en chercher un en ligne.

# ── 1. Les modèles, téléchargés une fois et vérifiés ─────────────────────────────
# Un fichier amont qui changerait de contenu casse la construction au lieu de changer
# le résultat en silence.
FROM debian:bookworm-slim AS modeles
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/*
# rtmlib, mode `balanced` (détecteur YOLOX-m HumanArt + RTMW-x, OpenMMLab, Apache-2.0) :
# rtmlib prend directement `<nom du zip>.onnx` s'il le trouve sous
# $TORCH_HOME/hub/checkpoints.
WORKDIR /out/torch/hub/checkpoints
RUN set -e; for nom in \
      rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a \
      rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122; do \
      base="$(basename "$nom")"; \
      curl -fsSL -o /tmp/m.zip "https://download.openmmlab.com/mmpose/v1/projects/$nom.zip"; \
      rm -rf /tmp/m && mkdir /tmp/m && unzip -q /tmp/m.zip -d /tmp/m; \
      mv "$(find /tmp/m -name '*.onnx' | head -1)" "$base.onnx"; \
    done \
 && printf '%s\n' \
    "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23  yolox_m_8xb8-300e_humanart-c2c7a14a.onnx" \
    "9a9bc17c13ff0a1e37507f45a037a21bc410cd08374b0e664c57d0080843aa58  rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx" \
    | sha256sum -c -
# YOLOv8n-pose d'Ultralytics (AGPL-3.0), poids officiels de la version v8.3.0, NON
# réentraînés : il sert au suivi BoT-SORT et à la détection de l'orientation.
# `pipeline/pose.py` le charge RELATIVEMENT au dossier courant.
WORKDIR /out/yolo
RUN curl -fsSL -o yolov8n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt \
 && echo "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212  yolov8n-pose.pt" | sha256sum -c -

# ── 2. L'image ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm
# libgl1 et libglib2.0-0 : OpenCV (exigé par rtmlib et ultralytics) les charge à
# l'import. Pas de ffmpeg système : OpenCV décode avec celui qu'embarque sa roue.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements-torch.txt /tmp/requirements-torch.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-torch.txt
# Les versions FIGÉES, installées sans résolution : la même image se reconstruit à
# l'identique, et `pip check` refuse une fermeture incomplète.
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-deps -r /tmp/requirements.lock.txt && pip check

ENV TORCH_HOME=/opt/modeles/torch
COPY --from=modeles /out/torch/hub/checkpoints/ /opt/modeles/torch/hub/checkpoints/

# Le code, et le modèle YOLO à côté de lui (chargé relativement au dossier courant).
WORKDIR /opt/masques
COPY pipeline ./pipeline
COPY scripts ./scripts
COPY --from=modeles /out/yolo/yolov8n-pose.pt ./
# Les fils de calcul bornés au quota du conteneur (KADANS_FILS) — sans cette variable,
# le module ne fait rien.
COPY sitecustomize.py /usr/local/lib/python3.11/site-packages/sitecustomize.py

# Un utilisateur sans droits ; tout ce qui s'écrit va sous /tmp (caches d'ultralytics,
# de matplotlib), et la sortie dans le dossier passé à --out.
RUN useradd --uid 10001 --create-home --home-dir /home/analyse analyse
USER 10001
ENV HOME=/home/analyse \
    YOLO_CONFIG_DIR=/tmp \
    YOLO_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib
ENTRYPOINT ["python", "scripts/masques_for_file.py"]
