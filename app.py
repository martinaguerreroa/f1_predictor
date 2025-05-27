import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and prep data
results = pd.read_csv('data/results.csv')
races = pd.read_csv('data/races.csv')
drivers = pd.read_csv('data/drivers.csv')
constructors = pd.read_csv('data/constructors.csv')

df = results.merge(races, on='raceId')
df = df.merge(drivers, on='driverId')
df = df.merge(constructors, on='constructorId')
df = df[['year', 'round', 'grid', 'positionOrder', 'driverRef', 'constructorRef']]
df = df.dropna()
df['podium'] = df['positionOrder'].apply(lambda x: 1 if x <= 3 else 0)

# Encode driver and constructor
driver_le = LabelEncoder()
constructor_le = LabelEncoder()
df['driverRef'] = driver_le.fit_transform(df['driverRef'])
df['constructorRef'] = constructor_le.fit_transform(df['constructorRef'])

# Train model
X = df[['year', 'round', 'grid', 'driverRef', 'constructorRef']]
y = df['podium']
model = RandomForestClassifier(class_weight='balanced')
model.fit(X, y)

# Streamlit UI
st.title("🏎️ F1 Podium Predictor")

year = st.slider("Year", int(df['year'].min()), int(df['year'].max()), 2023)
round_num = st.number_input("Race Round", min_value=1, max_value=30, value=5)
grid = st.number_input("Grid Position", min_value=1, max_value=30, value=1)
driver_name = st.selectbox("Driver", driver_le.classes_)
team_name = st.selectbox("Constructor", constructor_le.classes_)

# Encode input
driver_encoded = driver_le.transform([driver_name])[0]
team_encoded = constructor_le.transform([team_name])[0]

# Predict
input_data = pd.DataFrame([[year, round_num, grid, driver_encoded, team_encoded]],
                          columns=['year', 'round', 'grid', 'driverRef', 'constructorRef'])

prediction = model.predict(input_data)[0]
result = "🏆 Podium Finish!" if prediction == 1 else "❌ No Podium"

if st.button("Predict"):
    st.subheader(result)
