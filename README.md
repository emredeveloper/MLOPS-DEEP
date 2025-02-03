# DeepSeek MLOps Project

## Quick Start (Windows)

1. Install dependencies:
```bash
run install
```

2. Train the model:
```bash
run train
```

3. Run tests:
```bash
run test
```

4. Get coverage report:
```bash
run coverage
```

5. Clean up:
```bash
run clean
```

## Model Training Pipeline

The project implements a complete ML training pipeline:
- Data loading and validation
- Preprocessing
- Model training and evaluation
- MLflow experiment tracking
- Model saving and versioning

## Services

- MLflow UI: http://localhost:5000
- FastAPI Swagger: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
