import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# Caricamento del dataset
df = pd.read_csv("C:/Users/-/Desktop/archive/car_price_dataset.csv")
df.head()
df.info()
df.describe()

# Preprocessing dei dati
df = df.dropna()  # Rimuovere valori mancanti

# Convertire variabili categoriche in numeriche
categorical_features = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separazione delle feature e del target
X = df.drop(['Price'], axis=1)  # Target: "Price"
y = df['Price']

# Normalizzazione delle feature
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Suddivisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento dei modelli
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Dizionario per salvare le feature importance
feature_importance_data = {}

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    print(f"{name} Results:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"R2 Score: {r2_score(y_test, y_pred)}")
    print(f"Training Time: {end_time - start_time:.2f} seconds\n")

    # Salvare le importanze delle feature per il grafico
    feature_importance_data[name] = model.feature_importances_

feature_names = df.drop(['Price'], axis=1).columns

# Visualizzazione della feature importance per Decision Tree
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_data['Decision Tree'], y=feature_names, color='red')
plt.xlabel("Importanza")
plt.ylabel("Feature")
plt.title("Modello Decision Tree")
plt.show()

# Visualizzazione della feature importance per Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_data['Random Forest'], y=feature_names, color='blue')
plt.xlabel("Importanza")
plt.ylabel("Feature")
plt.title("Modello Random Forest")
plt.show()

# Esempio di nuova auto per la predizione
new_car = pd.DataFrame({
    'Brand': ['Toyota'],
    'Model': ['Corolla'],
    'Year': [2020],
    'Engine_Size': [1.8],
    'Fuel_Type': ['Petrol'],
    'Transmission': ['Automatic'],
    'Mileage': [15000],
    'Doors': [5],
    'Owner_Count': [3]

})

# Applicare Label Encoding alle variabili categoriche
for col in categorical_features:
    new_car[col] = label_encoders[col].transform(new_car[col])

# Applicare la normalizzazione
new_car_scaled = scaler.transform(new_car)

# Predizione del prezzo usando il modello Random Forest
predicted_price = models['Random Forest'].predict(new_car_scaled)

print(f"Prezzo predetto per l'auto: {predicted_price[0]:,.2f}")

predicted_price_dt = models['Decision Tree'].predict(new_car_scaled)
print(f"Prezzo predetto dal Decision Tree: {predicted_price_dt[0]:,.2f}")


