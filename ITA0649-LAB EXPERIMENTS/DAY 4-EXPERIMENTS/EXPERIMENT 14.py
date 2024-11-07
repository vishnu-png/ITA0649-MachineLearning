from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame({
    'Size': [1500, 2500, 3000, 3500, 4000],
    'LocationScore': [3, 5, 4, 4, 5],
    'Price': [300000, 500000, 450000, 500000, 600000]
})

X = data[['Size', 'LocationScore']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted House Prices:", y_pred)
