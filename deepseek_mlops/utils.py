import time
import logging
import os
from deepseek_mlops.config import LOG_FILE

# Logs klasörünü oluştur (eğer yoksa)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Logger yapılandırması
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    encoding="utf-8"  # Türkçe karakter desteği
)

def log(message):
    """Logları terminale ve dosyaya yazan yardımcı fonksiyon."""
    print(message)
    logging.info(message)

def measure_time(func):
    """Fonksiyonun çalışma süresini ölçer."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log(f"[TIMER] {func.__name__} süresi: {elapsed_time:.4f} saniye")
        return result
    return wrapper