from sklearn.datasets import load_iris
import pandas as pd
from deepseek_mlops.utils import log

def load_data():
    """Veriyi yükler ve bir pandas DataFrame olarak döndürür."""
    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target

    log("[INFO] Veri başarıyla yüklendi.")
    return df