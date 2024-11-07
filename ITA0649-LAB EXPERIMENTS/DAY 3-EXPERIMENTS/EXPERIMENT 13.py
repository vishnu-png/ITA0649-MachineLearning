from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

data = pd.DataFrame({
    'Mileage': [15000, 30000, 45000, 60000, 75000, 90000],
    'Age': [1, 2, 3, 4, 5, 6],
    'Price': [20000, 15000, 13000, 10000, 8000, 6000]
})

X = data[['Mileage', 'Age']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Predicted Car Prices:", y_pred)
