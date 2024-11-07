from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame({
    'Month': [1, 2, 3, 4, 5, 6],
    'Sales': [200, 220, 250, 300, 310, 330]
})

X = data[['Month']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted Future Sales:", y_pred)
