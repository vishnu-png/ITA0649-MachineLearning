from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

data = pd.DataFrame({
    'Income': [30000, 60000, 50000, 120000, 70000, 20000, 100000, 40000],
    'Debt': [10000, 20000, 15000, 30000, 25000, 5000, 20000, 15000],
    'CreditScore': [0, 1, 1, 1, 1, 0, 1, 0]
})

X = data[['Income', 'Debt']]
y = data['CreditScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
