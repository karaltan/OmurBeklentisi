import pandas as pd
import tkinter as tk
from tkinter import messagebox
from preprocess import load_data, preprocess_data
from model import LifeExpectancyModel
from sklearn.linear_model import LinearRegression

# Modeli eğit
data = load_data('data/life-expectancy.xlsx')
feature_columns = [col for col in data.columns if col != 'Life']
X, y = preprocess_data(data, feature_columns)
model = LifeExpectancyModel(LinearRegression())
model.train(X, y)

def predict_life_expectancy(user_inputs):
    input_df = pd.DataFrame([user_inputs], columns=feature_columns)
    prediction = model.predict(input_df)
    return prediction[0]

def on_predict():
    try:
        user_inputs = []
        for entry in entries:
            val = float(entry.get())
            user_inputs.append(val)
        result = predict_life_expectancy(user_inputs)
        messagebox.showinfo("Tahmin", f"Tahmini Yaşam Süresi: {result:.2f} yıl")
    except Exception as e:
        messagebox.showerror("Hata", f"Girdi hatası: {e}")

# Tkinter arayüzü
root = tk.Tk()
root.title("Tahmini Ömür Beklentisi Uygulaması")

entries = []
for idx, col in enumerate(feature_columns):
    tk.Label(root, text=col).grid(row=idx, column=0, padx=5, pady=5, sticky="e")
    entry = tk.Entry(root)
    entry.grid(row=idx, column=1, padx=5, pady=5)
    entries.append(entry)

tk.Button(root, text="Tahmin Et", command=on_predict).grid(row=len(feature_columns), column=0, columnspan=2, pady=10)

root.mainloop()