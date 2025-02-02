from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deepseek_mlops.config import TRAIN_TEST_SPLIT, RANDOM_STATE
from deepseek_mlops.utils import log

def preprocess(data):
    """Veriyi ölçeklendirir ve eğitim/test setlerine böler."""
    X = data.drop(columns=['target'])
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE
    )
    
    log("[INFO] Veri başarıyla ön işlendi.")
    return X_train, X_test, y_train, y_test