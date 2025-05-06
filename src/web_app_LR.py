import pandas as pd
import numpy as np
import streamlit as st
from preprocess import load_data, preprocess_data
from model import LifeExpectancyModel
from sklearn.linear_model import LinearRegression

# Modeli eğit
data = load_data('data/life-expectancy.xlsx')
feature_columns = [col for col in data.columns if col != 'Life']
X, y = preprocess_data(data, feature_columns)
model = LifeExpectancyModel(LinearRegression())
model.train(X, y)

st.title("Tahmini Ömür Beklentisi (LinearRegression)")

user_inputs = []
for col in feature_columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    user_inputs.append(val)

if st.button("Tahmin Et"):
    input_df = pd.DataFrame([user_inputs], columns=feature_columns)
    prediction = model.predict(input_df)
    st.success(f"Tahmini Yaşam Süresi: {prediction[0]:.2f} yıl")
