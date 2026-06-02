import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("Error: dataset.csv not found. Run data_generator.py first.")
    exit()

X = df[['Experience', 'Education']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)

joblib.dump(model, 'model.pkl')

print(f"Model trained! R2 Score (Accuracy): {r2:.4f}")
if r2 > 0.8:
    print("Status: 🟢 Model is healthy for production.")
else:
    print("Status: 🔴 WARNING! Model performance is severely degraded!")
