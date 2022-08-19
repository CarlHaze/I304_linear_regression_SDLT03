import numpy as np  # linear algebra
import pandas as pd  # data processing
import matplotlib.pyplot as plt  # drawgraph
import reg as reg
from sklearn.metrics import r2_score

# dataset = pd.read_csv("heart_disease_uci_cleaned.csv")
# simple scatter of the data
dataset = pd.read_csv("../CSV-Files/AgeCholesterol.csv")
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
plt.ylabel("Age")
plt.xlabel("Cholesterol")
plt.title("Scatter graph of Correlation between Age and Cholesterol")
plt.scatter(x, y)
plt.show()

# linear regression model using sklearn
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
# "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1,
# -1) if it contains a single sample."
x = x.reshape(-1, 1) #we can add a dimension?
y = y.reshape(-1, 1)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x, y)
# Predicting the Test set results
y_pred = regressor.predict(x)
plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='blue')
plt.title('Correlation between Age and Cholesterol sklearn')
plt.xlabel('Cholesterol')
plt.ylabel('Age')
plt.show()
# get the slope and Y-Intercept of the model
print("sklearn Slope: ", regressor.coef_)
print("sklearn Y-Intercept: ", regressor.intercept_)
# print the R-Square value of the model
print("sklearn R-Square: ", r2_score(y, y_pred))  # R2 score is pretty bad for this model.

# Linear Regression using polyfit
# apply linear Regression using polyfit method Degree of 1 is used
# doesn't mean too much to me shows same stuff as other graphs
from sklearn.preprocessing import PolynomialFeatures

x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
poly_reg = PolynomialFeatures(degree=1)  # degree is the power of the polynomial ?
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
y_pred = lin_reg_2.predict(x_poly)
plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='blue')
plt.title('Correlation between Age and Cholesterol polyfit')
plt.xlabel('Cholesterol')
plt.ylabel('Age')
plt.show()
# Print the linear regression polyfit model
# Linear Regression using polyfit proper?
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
linear_model_csv = np.polyfit(x, y, 1)
print("Poly Linear Regression: ", linear_model_csv, sep="\n\t")
print("Poly Linear Regression Slope: ", linear_model_csv[0], sep="\n\t")
print("Poly Linear Regression Y-Intercept: ", linear_model_csv[1], sep="\n\t")
#print the R-Square value of the model

# Plot the predicted values over the scatter graph same result??
plt.scatter(x, y, color='red')
plt.plot(x, linear_model_csv[0] * x + linear_model_csv[1], color='blue')
plt.title('predicted values, Age and Cholesterol polyfit')
plt.xlabel('Cholesterol')
plt.ylabel('Age')
plt.show()

x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
# apply Linear Regression analysis on this dataset using linregress method
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(x, y)
# print linregress results
print("Linregress: ", "Slope:",slope, "intercept:", intercept, "r_value:",r_value, "p_value:",p_value, "std_err",std_err, sep="\n\t")
print("Linregress Slope: ", slope)
print("Linregress Y-Intercept: ", intercept)
print("Linregress R-Square: ", r_value ** 2)
# Plot the predicted values
plt.scatter(x, y, color='red')
plt.plot(x, slope * x + intercept, color='blue')
plt.title('predicted values, Age and Cholesterol linregress')
plt.xlabel('Cholesterol')
plt.ylabel('Age')
plt.show()



