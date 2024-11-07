from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame({
    'Feature': [1, 2, 3, 4, 5, 6, 7, 8],
    'Target': [2, 3, 5, 7, 11, 13, 17, 19]
})

X = data[['Feature']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(poly.fit_transform(X_test))

print("Linear Regression predictions:", linear_pred)
print("Polynomial Regression predictions:", poly_pred)
