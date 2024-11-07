from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

data = pd.DataFrame({
    'Feature': [1, 2, 3, 4, 5, 6, 7, 8],
    'Target': [2, 4, 6, 8, 10, 12, 14, 16]
})

X = data[['Feature']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Predicted values:", y_pred)
