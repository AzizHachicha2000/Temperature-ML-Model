import pandas as pd
import numpy as np
import datetime

# Generation de données aléatoires
def generate_data(n_samples=1000):
    timestamps = [datetime.datetime.now() - datetime.timedelta(minutes=i) for i in range(n_samples)]
    sensor_ids = np.random.randint(1, 6, size=n_samples)
    temperature = np.random.uniform(20, 40, size=n_samples)
    humidity = np.random.uniform(30, 90, size=n_samples)
    pressure = np.random.uniform(950, 1050, size=n_samples)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_id": sensor_ids,
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure
    })

    return data

# Stockage des données
data = generate_data(1000)
data.to_csv("data/sensor_data.csv", index=False)
print("📝 Données générées et stockées dans data/sensor_data.csv")