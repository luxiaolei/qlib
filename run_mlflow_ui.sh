#!/bin/bash

# Set the path to MLflow runs directory using absolute path
MLFLOW_DIR="$(pwd)/examples/mlruns"

# Create the directory if it doesn't exist
mkdir -p "$MLFLOW_DIR"

# Start MLflow UI
echo "Starting MLflow UI server..."
echo "Access the UI at http://localhost:5000 or http://<your-server-ip>:5000"

mlflow ui --backend-store-uri "file://${MLFLOW_DIR}" --host 0.0.0.0 --port 5001
