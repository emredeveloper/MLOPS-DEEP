from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from deepseek_mlops.utils import log, measure_time

@measure_time
def tune_hyperparameters(X_train, y_train):
    """Tunes hyperparameters using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    log(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_
