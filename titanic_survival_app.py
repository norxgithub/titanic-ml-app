
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

# Preprocessing
df = df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'])
df = df.dropna()

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])         # female=0, male=1
df['embarked'] = le.fit_transform(df['embarked'])

# Features and Target
X = df[['age', 'pclass', 'sex', 'sibsp']]
y = df['survived']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🚢 Ramalan Survival Titanic")
st.write("Masukkan maklumat penumpang untuk lihat kemungkinan selamat.")

age = st.slider("Umur Penumpang", 1, 80, 25)
pclass = st.selectbox("Kelas Tiket", [1, 2, 3])
sex = st.radio("Jantina", ["Perempuan", "Lelaki"])
sibsp = st.slider("Bilangan Saudara / Pasangan", 0, 5, 0)

# Encode input
sex_encoded = 0 if sex == "Perempuan" else 1

# Prediction
input_data = pd.DataFrame([[age, pclass, sex_encoded, sibsp]],
                          columns=['age', 'pclass', 'sex', 'sibsp'])

prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

# Output
if prediction == 1:
    st.success(f"✅ Penumpang ini dijangka SELAMAT (Probabiliti: {prob:.2f})")
else:
    st.error(f"❌ Penumpang ini dijangka TIDAK SELAMAT (Probabiliti: {prob:.2f})")

# Optional: tunjukkan prestasi model
if st.checkbox("Tunjukkan ketepatan model (test set)"):
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    st.write(f"Ketepatan model: {acc:.2f}")
