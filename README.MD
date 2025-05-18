# ML Pipeline with GitHub Actions Automation

This project automates an end-to-end machine learning pipeline using GitHub Actions. It uses the Titanic dataset to build a classification model that predicts passenger survival. The entire process — from data preprocessing to model training and testing — is executed automatically on every push or pull request.

## Features

- Loads and preprocesses the Titanic dataset (handles missing values, encodes categorical features, scales numerical data).
- Trains a `RandomForestClassifier` using `scikit-learn`.
- Saves the trained model as a `.joblib` file.
- Includes unit tests for data preprocessing and model accuracy (e.g., accuracy ≥ 80%).
- Automates all steps using GitHub Actions:
  - Sets up the Python environment
  - Installs dependencies
  - Runs unit tests
  - Trains the model
  - Uploads the trained model artifact

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. **(Optional) Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate     # For Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the pipeline locally:**

    ```bash
    python ml_pipeline.py
    ```

5. **Run unit tests:**

    ```bash
    pytest test_ml_pipeline.py
    ```

## GitHub Actions

The workflow is defined in `.github/workflows/ml_pipeline.yml` and is triggered on:

- Every `push`
- Every `pull_request`

It performs the following steps automatically:
- Sets up a Python 3.9 environment
- Installs required packages
- Runs tests
- Trains the ML model
- Uploads the model artifact (`random_forest_model.joblib`) for download
