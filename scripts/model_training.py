import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Chargement des données
data = pd.read_csv("data/sensor_data.csv")

# Sélection des features et cible
x = data[["humidity", "pressure"]]
y = data["temperature"]

# Division en ensemble d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LinearRegression()
model.fit(x_train, y_train)

# Évaluation du modèle
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Précision du modèle : MAE = {mae:.2f}, MSE = {mse:.2f}")

# Sauvegarde du modèle
joblib.dump(model, "models/iot_temperature_model.pkl")
print("Modèle entraîné et sauvegardé dans models/iot_temperature_model.pkl")