import os
import joblib
from deepseek_mlops.config import MODEL_PATH
from deepseek_mlops.utils import log

# Model dosyasının kaydedileceği dizini oluştur (eğer yoksa)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def save_trained_model(model):
    """Eğitilmiş modeli dosyaya kaydeder."""
    joblib.dump(model, MODEL_PATH)
    log(f"[INFO] Model başarıyla kaydedildi: {MODEL_PATH}")