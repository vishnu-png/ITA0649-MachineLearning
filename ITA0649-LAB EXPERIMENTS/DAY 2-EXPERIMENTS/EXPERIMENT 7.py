from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8],
    'Feature2': [5, 10, 15, 10, 5, 10, 15, 10],
    'Target': [0, 1, 0, 1, 0, 1, 0, 1]
})

X = data[['Feature1', 'Feature2']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted values:", y_pred)
