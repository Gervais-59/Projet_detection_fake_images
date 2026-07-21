# ============================================================
# Image de production pour l'API de detection
#
# Construction multi-etapes : les dependances sont installees dans une
# premiere image, puis seuls les artefacts utiles sont copies dans
# l'image finale. Resultat : image plus legere, sans outils de build.
# ============================================================

FROM python:3.11-slim AS build

WORKDIR /install

# Installation des dependances dans un prefixe isole
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements_deploy.txt


# ------------------------------------------------------------
FROM python:3.11-slim

# Utilisateur non privilegie : un service ne doit jamais tourner en root
RUN useradd --create-home --uid 1000 apiuser

WORKDIR /app

COPY --from=build /install/deps /usr/local
COPY --chown=apiuser:apiuser app.py .

# Cache des modeles HuggingFace dans le repertoire de l'utilisateur
ENV HF_HOME=/home/apiuser/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER apiuser
EXPOSE 8000

# Verification de sante : Docker sait ainsi si le conteneur est reellement pret
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/sante')"

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]