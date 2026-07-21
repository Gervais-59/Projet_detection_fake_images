"""Publication du modele sur le Hub HuggingFace."""

from huggingface_hub import login
from transformers import ViTForImageClassification, ViTImageProcessor

login()   # colle ton token "Write" : https://huggingface.co/settings/tokens

# 1. Chargement depuis le dossier decompresse
CHEMIN = r"C:\Users\KONAN GERVAIS\Desktop\projet_detection_image\model\modele_extrait\modele_vit_detecteur_384"
modele = ViTForImageClassification.from_pretrained(CHEMIN)
processor = ViTImageProcessor.from_pretrained(CHEMIN)
print("Charge —", modele.config.id2label, "| entree :", processor.size)

# 2. Publication (remplace par TON pseudo HuggingFace)
NOM = "Konan59/vit-detecteur-image-ia-384"
modele.push_to_hub(NOM)
processor.push_to_hub(NOM)

print(f"Publie : https://huggingface.co/{NOM}")