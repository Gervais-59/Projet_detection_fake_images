"""
API de détection d'images générées par IA.

Service REST exposant un modèle Vision Transformer affiné.

Lancement local :
    uvicorn app:app --reload --port 8000

Documentation interactive : http://127.0.0.1:8000/docs
"""

import io
import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from transformers import ViTForImageClassification, ViTImageProcessor

# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("detecteur-ia")

# Modèle local s'il existe, sinon Hub HuggingFace.
# Le même code fonctionne ainsi en développement et en production.
_MODELE_LOCAL = Path(r"C:\Users\KONAN GERVAIS\Desktop\projet_detection_image\model\modele_extrait\modele_vit_detecteur_384")

MODEL_ID = os.environ.get(
    "MODEL_ID",
    str(_MODELE_LOCAL) if _MODELE_LOCAL.exists() else "Konan59/vit-detecteur-image-ia-384",
)

SEUIL_FAKE = float(os.environ.get("SEUIL_FAKE", "0.3"))
SEUIL_DOUTE = float(os.environ.get("SEUIL_DOUTE", "0.75"))
TAILLE_MAX_MO = float(os.environ.get("TAILLE_MAX_MO", "10"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Conteneur des objets chargés une seule fois au démarrage
etat = {"modele": None, "processor": None}


# ============================================================
# CYCLE DE VIE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère les ressources à l'arrêt.

    Pourquoi ce mécanisme plutôt qu'un chargement au niveau du module :
    le modèle (~350 Mo) n'est chargé qu'une fois, à l'initialisation du
    service, et non à chaque requête. C'est la différence entre une API
    qui répond en 200 ms et une qui répond en 8 s.
    """
    logger.info("Chargement du modèle %s sur %s...", MODEL_ID, DEVICE)
    debut = time.time()

    etat["modele"] = ViTForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
    etat["processor"] = ViTImageProcessor.from_pretrained(MODEL_ID)
    etat["modele"].eval()

    logger.info(
        "Modèle prêt en %.1f s — classes : %s",
        time.time() - debut, etat["modele"].config.id2label,
    )
    yield

    logger.info("Arrêt du service, libération des ressources.")
    etat.clear()


app = FastAPI(
    title="Détecteur d'images générées par IA",
    description=(
        "Classifie une image comme photographie authentique ou image générée par IA. "
        "**Limite connue** : le rappel sur les générateurs récents est d'environ 63 % — "
        "un verdict « authentique » ne constitue pas une preuve."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# SCHÉMAS DE RÉPONSE
# ============================================================

class Prediction(BaseModel):
    """Résultat d'une analyse d'image."""

    verdict: str = Field(..., description="FAKE (générée) ou REAL (authentique)")
    confiance: float = Field(..., ge=0, le=1, description="Confiance dans le verdict")
    probabilite_fake: float = Field(..., ge=0, le=1)
    probabilite_reelle: float = Field(..., ge=0, le=1)
    incertain: bool = Field(..., description="True si la confiance est sous le seuil de doute")
    duree_ms: float = Field(..., description="Temps d'inférence en millisecondes")
    avertissement: str


class Sante(BaseModel):
    """État du service."""

    statut: str
    modele_charge: bool
    modele_id: str
    appareil: str


# ============================================================
# LOGIQUE MÉTIER
# ============================================================

def analyser(image: Image.Image) -> Prediction:
    """Analyse une image et retourne la prédiction structurée.

    Séparée de la couche HTTP : cette fonction est testable unitairement
    sans lancer de serveur, et réutilisable dans un autre contexte
    (traitement par lots, tâche planifiée).
    """
    debut = time.perf_counter()

    entrees = etat["processor"](image.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = etat["modele"](**entrees).logits
    probas = F.softmax(logits, dim=-1)[0].cpu()

    p_fake, p_real = float(probas[0]), float(probas[1])
    est_fake = p_fake > SEUIL_FAKE
    confiance = p_fake if est_fake else p_real

    return Prediction(
        verdict="FAKE" if est_fake else "REAL",
        confiance=round(confiance, 4),
        probabilite_fake=round(p_fake, 4),
        probabilite_reelle=round(p_real, 4),
        incertain=confiance < SEUIL_DOUTE,
        duree_ms=round((time.perf_counter() - debut) * 1000, 1),
        avertissement=(
            "Rappel d'environ 63 % sur les générateurs récents : un verdict "
            "REAL ne constitue pas une preuve d'authenticité."
        ),
    )


def lire_image(contenu: bytes, nom: str) -> Image.Image:
    """Décode les octets reçus en image PIL, avec messages d'erreur explicites."""
    if len(contenu) > TAILLE_MAX_MO * 1_000_000:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (max {TAILLE_MAX_MO} Mo).",
        )
    try:
        return Image.open(io.BytesIO(contenu))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail=f"'{nom}' n'est pas une image exploitable (formats : JPEG, PNG, WebP).",
        )


# ============================================================
# ROUTES
# ============================================================

@app.get("/sante", response_model=Sante, tags=["Service"])
async def sante():
    """Vérifie que le service est opérationnel et le modèle chargé.

    Route indispensable en production : les orchestrateurs (Kubernetes,
    Cloud Run, load balancers) l'interrogent pour savoir si l'instance
    peut recevoir du trafic.
    """
    return Sante(
        statut="ok" if etat.get("modele") is not None else "modele_non_charge",
        modele_charge=etat.get("modele") is not None,
        modele_id=MODEL_ID,
        appareil=DEVICE,
    )


@app.post("/predire", response_model=Prediction, tags=["Détection"])
async def predire(fichier: UploadFile = File(..., description="Image à analyser")):
    """Analyse une image et retourne le verdict.

    Exemple :
        curl -X POST http://127.0.0.1:8000/predire -F "fichier=@photo.jpg"
    """
    if etat.get("modele") is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    image = lire_image(await fichier.read(), fichier.filename or "image")
    resultat = analyser(image)

    logger.info(
        "%s -> %s (%.0f%%) en %.0f ms",
        fichier.filename, resultat.verdict, resultat.confiance * 100, resultat.duree_ms,
    )
    return resultat


@app.post("/predire-lot", tags=["Détection"])
async def predire_lot(fichiers: list[UploadFile] = File(...)):
    """Analyse plusieurs images en une seule requête.

    Les erreurs sur un fichier n'interrompent pas le traitement des autres :
    chaque résultat porte son propre statut.
    """
    if etat.get("modele") is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    if len(fichiers) > 20:
        raise HTTPException(status_code=413, detail="Maximum 20 images par requête.")

    resultats = []
    for f in fichiers:
        try:
            image = lire_image(await f.read(), f.filename or "image")
            resultats.append({
                "fichier": f.filename,
                "statut": "ok",
                **analyser(image).model_dump(),
            })
        except HTTPException as e:
            resultats.append({"fichier": f.filename, "statut": "erreur", "detail": e.detail})

    n_fake = sum(1 for r in resultats if r.get("verdict") == "FAKE")
    return {
        "total": len(resultats),
        "detectees_generees": n_fake,
        "detectees_reelles": sum(1 for r in resultats if r.get("verdict") == "REAL"),
        "resultats": resultats,
    }


@app.get("/", response_class=HTMLResponse, tags=["Service"])
async def accueil():
    """Page de test minimale : dépose une image et vois le résultat."""
    return """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Détecteur d'images IA</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 60px auto;
         padding: 0 20px; color: #1a1a1a; line-height: 1.6; }
  h1 { font-size: 1.6rem; margin-bottom: .3rem; }
  .sous { color: #666; margin-bottom: 2rem; }
  .zone { border: 2px dashed #ccc; border-radius: 10px; padding: 36px;
          text-align: center; cursor: pointer; transition: .2s; }
  .zone:hover { border-color: #2563eb; background: #f8fafc; }
  #apercu { max-width: 100%; margin-top: 18px; border-radius: 8px; display: none; }
  #res { margin-top: 24px; padding: 18px; border-radius: 8px; display: none; }
  .fake { background: #fef2f2; border-left: 4px solid #dc2626; }
  .real { background: #f0fdf4; border-left: 4px solid #16a34a; }
  .verdict { font-size: 1.2rem; font-weight: 600; margin-bottom: .5rem; }
  .barre { height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; margin: 10px 0; }
  .barre span { display: block; height: 100%; background: #dc2626; }
  .note { font-size: .85rem; color: #666; margin-top: .8rem; }
  a { color: #2563eb; }
</style>
</head>
<body>
  <h1>🔍 Détecteur d'images générées par IA</h1>
  <p class="sous">Vision Transformer affiné · <a href="/docs">Documentation de l'API</a></p>

  <div class="zone" onclick="document.getElementById('f').click()">
    <p><strong>Choisir une image</strong><br><small>JPEG, PNG ou WebP</small></p>
    <input type="file" id="f" accept="image/*" hidden onchange="envoyer(this.files[0])">
  </div>
  <img id="apercu">
  <div id="res"></div>

<script>
async function envoyer(fichier) {
  if (!fichier) return;
  const apercu = document.getElementById('apercu');
  apercu.src = URL.createObjectURL(fichier);
  apercu.style.display = 'block';

  const res = document.getElementById('res');
  res.style.display = 'block';
  res.className = '';
  res.textContent = 'Analyse en cours...';

  const donnees = new FormData();
  donnees.append('fichier', fichier);

  try {
    const rep = await fetch('/predire', { method: 'POST', body: donnees });
    if (!rep.ok) throw new Error((await rep.json()).detail);
    const d = await rep.json();

    res.className = d.verdict === 'FAKE' ? 'fake' : 'real';
    const pct = Math.round(d.probabilite_fake * 100);
    res.innerHTML = `
      <div class="verdict">${d.verdict === 'FAKE' ? '🤖 Probablement générée par IA' : '📷 Probablement authentique'}</div>
      <div>Confiance : <strong>${(d.confiance * 100).toFixed(1)} %</strong>${d.incertain ? ' — <em>verdict incertain</em>' : ''}</div>
      <div class="barre"><span style="width:${pct}%"></span></div>
      <div style="font-size:.85rem">P(générée) = ${pct} % · P(réelle) = ${100 - pct} % · ${d.duree_ms} ms</div>
      <div class="note">${d.avertissement}</div>`;
  } catch (e) {
    res.className = '';
    res.textContent = 'Erreur : ' + e.message;
  }
}
</script>
</body>
</html>
"""