from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

data = pd.DataFrame({
    'RAM': [2, 4, 6, 8, 12],
    'Battery': [3000, 4000, 5000, 6000, 7000],
    'PriceCategory': [0, 1, 1, 1, 2]
})

X = data[['RAM', 'Battery']]
y = data['PriceCategory']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
