# Credit Risk Engine: Production MLOps Pipeline

## Project Overview
This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline designed to evaluate consumer credit risk using ensemble classification models. This project couples data engineering and model training with a container-ready REST API microservice, automated testing, and a continuous integration (CI) pipeline.

## Architecture & Subsystem Layout

### 1. Model Training Pipeline (`train.py`)
* **Feature Engineering:** Synthesises consumer financial indicators and enforces operational safety boundaries using isolated feature scaling (`StandardScaler`) to prevent data leakage.
* **Ensemble Optimisation:** Trains a `RandomForestClassifier` adjusted via structural class weighting configurations to systematically balance predictive accuracy against financial risk metrics.
* **Artifact Serialisation:** Compiles and serialises the trained model and scaling definitions into production-ready binaries (`.pkl`).

### 2. Live Inference API Microservice (`app.py`)
* **FastAPI Gateway:** Implements a high-velocity HTTP interface utilising the ASGI FastAPI framework for sub-millisecond response profiles.
* **Network Edge Validation:** Leverages Pydantic schemas to enforce strict data typing, numeric constraints, and zero-trust boundary criteria prior to inference execution.

### 3. DevOps & Reliability Layer
* **Containerisation:** Fully containerised via **Docker**, ensuring isolated and reproducible execution environments across all deployment targets.
* **Automated Testing:** Integrated `pytest` suite (`test_app.py`) validating API health, endpoint routing, and ML model inference accuracy.
* **CI/CD Pipeline:** Utilises **GitHub Actions** to automatically provision remote Linux runners, install dependencies, compile models, and execute the test suite on every push to the main branch.

## Local Execution & Deployment

### Standard Python Runtime
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model and generate .pkl binaries
python train.py

# 3. Boot the live inference server
uvicorn app:app --reload
```

### Docker Container Deployment
```bash
# Build the isolated container image
docker build -t credit-risk-api .

# Run the containerised microservice on port 8000
docker run -p 8000:8000 credit-risk-api
```

### Interactive Testing
Once the server is running natively or via Docker, access the auto-generated Swagger UI dashboard at http://localhost:8000/docs to send payload data and receive real-time credit risk predictions.
