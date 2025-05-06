import pandas as pd
import numpy as np
import streamlit as st
from preprocess import load_data, preprocess_data
from model import LifeExpectancyModel
from sklearn.tree import DecisionTreeRegressor

# Modeli eğit
data = load_data('data/life-expectancy.xlsx')
feature_columns = [col for col in data.columns if col != 'Life']
X, y = preprocess_data(data, feature_columns)
model = LifeExpectancyModel(DecisionTreeRegressor(random_state=42))
model.train(X, y)

st.title("Tahimini Ömür Beklentisi (DecisionTreeRegressor)")

user_inputs = []
for col in feature_columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    user_inputs.append(val)

if st.button("Tahmin Et"):
    input_df = pd.DataFrame([user_inputs], columns=feature_columns)
    prediction = model.predict(input_df)
    st.success(f"Tahmini Yaşam Süresi: {prediction[0]:.2f} yıl")

# Rastgele test için:
if st.button("Rastgele Değerlerle Test Et"):
    mins = X.min()
    maxs = X.max()
    random_inputs = [np.random.uniform(mins[col], maxs[col]) for col in feature_columns]
    input_df = pd.DataFrame([random_inputs], columns=feature_columns)
    prediction = model.predict(input_df)
    st.info("Rastgele değerlerle tahmini yaşam süresi: {:.2f} yıl".format(prediction[0]))
    for col, val in zip(feature_columns, random_inputs):
        st.write(f"{col}: {val:.2f}")
