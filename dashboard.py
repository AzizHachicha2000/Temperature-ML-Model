import streamlit as st
import pandas as pd
import joblib

# Chargement des données et du modèle
data = pd.read_csv("data/sensor_data.csv")
model = joblib.load("models/iot_temperature_model.pkl")

st.title("Tableau de Bord IoT en Temps Réel")

# Affichage des données
st.write("Aperçu des données")
st.dataframe(data.head())

# Prédiction de température
st.write("Prédiction de Température")

humidity = st.slider("Humidité (%)", min_value=30, max_value=90, value=50)
pressure = st.slider("Pression (hPa)", min_value=950, max_value=1050, value=1013)

prediction = model.predict([[humidity, pressure]])[0]
st.write(f"Température prédite : {prediction:.2f}°C")