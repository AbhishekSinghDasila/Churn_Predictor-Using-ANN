# Churn Predictor Using ANN

This project is a customer churn prediction application built with an Artificial Neural Network (ANN) using TensorFlow/Keras and Scikit-learn. It includes preprocessing, training, and a Streamlit interface for live prediction.

## Project Structure

- `Churn_Modelling.csv` - source dataset for training the model
- `app.py` - Streamlit application for making churn predictions
- `experiments.ipynb` - notebook containing preprocessing, model training, and evaluation
- `prediction.ipynb` - notebook for loading the trained model and testing individual predictions
- `requirements.txt` - Python package dependencies
- `churn_model.h5` - saved Keras model
- `scaler.pkl`, `label_encoder_gender.pkl`, `one_hot_encoder_geography.pkl` - saved preprocessing objects

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> For deployment on managed platforms, prefer `tensorflow-cpu` instead of `tensorflow` if the environment does not support GPU packages.

## Running Locally

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Start the Streamlit app:

```powershell
streamlit run app.py
```

## How It Works

- The dataset is preprocessed by encoding categorical variables and scaling numerical features.
- A neural network model is trained in `experiments.ipynb`.
- The trained model and preprocessing objects are saved to disk.
- `app.py` loads the saved model, scaler, and encoders, then predicts churn probability for user-provided customer inputs.

## Notes

- Ensure the same Python and package versions are used during training and deployment to avoid compatibility issues.
- If deployment fails due to package installation errors, use a lighter dependency list and `tensorflow-cpu`.
- The app currently expects the saved files `churn_model.h5`, `scaler.pkl`, `label_encoder_gender.pkl`, and `one_hot_encoder_geography.pkl` in the project root.
