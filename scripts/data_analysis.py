import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
data = pd.read_csv("data/sensor_data.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])

# Analyse statistique
print("\nStatistiques descriptives :\n", data.describe())

# Visualisation des distributions
sns.pairplot(data, hue="sensor_id")
plt.show()

# Détection d’anomalies : températures extrêmes
anomalies = data[(data["temperature"] > 38) | (data["temperature"] < 22)]
print("\nAnomalies détectées :\n", anomalies)

# Sauvegarde des anomalies
anomalies.to_csv("data/anomalies.csv", index=False)