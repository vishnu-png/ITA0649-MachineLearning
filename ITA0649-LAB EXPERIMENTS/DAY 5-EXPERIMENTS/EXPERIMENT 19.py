from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5, 6],
    'Feature2': [5, 10, 15, 20, 25, 30],
    'Target': [0, 1, 0, 1, 0, 1]
})

X = data[['Feature1', 'Feature2']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
