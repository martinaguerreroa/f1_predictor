# libraries (tools to help with data)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the F1 race data from files
results = pd.read_csv('data/results.csv')
races = pd.read_csv('data/races.csv')
drivers = pd.read_csv('data/drivers.csv')
constructors = pd.read_csv('data/constructors.csv')

# Step 2: Combine the files into one big table of useful data
df = results.merge(races, on='raceId')
df = df.merge(drivers, on='driverId')
df = df.merge(constructors, on='constructorId')

# Step 3: Keep only important columns (stuff we care about)
df = df[['year', 'round', 'grid', 'positionOrder', 'driverRef', 'constructorRef']]
df = df.dropna()  # remove empty data

# Step 4: Add a new column for whether a driver finished Top 3
df['podium'] = df['positionOrder'].apply(lambda x: 1 if x <= 3 else 0)

# Step 5: Turn driver/team names into numbers (computers need numbers)
df['driverRef'] = LabelEncoder().fit_transform(df['driverRef'])
df['constructorRef'] = LabelEncoder().fit_transform(df['constructorRef'])

# Step 6: Choose what goes into the model (features) and what we want to predict (target)
X = df[['year', 'round', 'grid', 'driverRef', 'constructorRef']]
y = df['podium']

# Step 7: Split into training (to learn) and testing (to check accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 8: Train the model (make it learn)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 9: Test how good it is
print("Accuracy:", model.score(X_test, y_test))
print(classification_report(y_test, model.predict(X_test)))
