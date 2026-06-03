import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

# We'll load scaler and encoders first, then attempt to load the model.
from tensorflow.keras.layers import Dense as KerasDense

class DenseFix(KerasDense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        # Ignore quantization_config if present in saved config
        super().__init__(*args, **kwargs)

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# The geography encoder was saved as a OneHotEncoder instance named
# `one_hot_encoder_geography.pkl` in the preprocessing notebook.
with open('one_hot_encoder_geography.pkl', 'rb') as file:
    one_hot_encoder_geography = pickle.load(file)

# Load the trained model. Use DenseFix to ignore legacy quantization config if present.
model = tf.keras.models.load_model('churn_model.h5', custom_objects={'Dense': DenseFix})

# Streamlit app
st.title("Customer Churn Prediction Using ANN")

# Input fields

gender=st.selectbox("Gender", label_encoder_gender.classes_)
geography=st.selectbox("Geography", one_hot_encoder_geography.categories_[0])
credit_score=st.number_input("Credit Score",max_value=850, min_value=300)
age=st.slider("Age", min_value=18, max_value=100)
tenure=st.slider("Tenure", min_value=0, max_value=10)
balance=st.number_input("Balance")
num_of_products=st.slider("Number of Products", min_value=0, max_value=4)
has_cr_card=st.selectbox("Has Credit Card", [0, 1])
is_active_member=st.selectbox("Is Active Member", [0, 1])
estimated_salary=st.number_input("Estimated Salary")


# Prepare the input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#one-hot encode the 'Geography' column
geography_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()
geography_df = pd.DataFrame(
    geography_encoded,
    columns=one_hot_encoder_geography.get_feature_names_out(['Geography'])
)
input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)

#scale the input data
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]
st.write(f"Churn Probability: {prediction_probability:.2f}")
if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

