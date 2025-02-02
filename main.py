from deepseek_mlops import load_data, preprocess, train_model, evaluate_model, save_trained_model
from deepseek_mlops.utils import log
from deepseek_mlops.mlflow_tracking import start_run, end_run, log_metrics, log_model
from sklearn.metrics import accuracy_score

def main():
    log("[INFO] Model eğitme süreci başlatıldı.")
    start_run()

    # 1. Veriyi yükle
    data = load_data()

    # 2. Ön işleme
    X_train, X_test, y_train, y_test = preprocess(data)

    # 3. Modeli eğit
    model = train_model(X_train, y_train)

    # 4. Modeli değerlendir
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    evaluate_model(model, X_test, y_test)
    log_metrics({"accuracy": accuracy})
    log_model(model)

    # 5. Modeli kaydet
    save_trained_model(model)

    log("[INFO] Model eğitme süreci tamamlandı.")
    end_run()

if __name__ == "__main__":
    main()