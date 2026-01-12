import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

filename = "data.csv"  # change this to your actual CSV file name
data = pd.read_csv(filename)

# ~~~ Extract columns ~~~
x = data.iloc[:, 0].values.reshape(-1, 1)  # first column (independent variable / x-value)
y = data.iloc[:, 1].values                 # second column (dependent variable / y-value)

# ~~~ Fit Linear Regression ~~~
model = LinearRegression()
model.fit(x, y)

# ~~~ Predict values for line of best fit ~~~
y_pred = model.predict(x)

# ~~~ Plot data and regression line ~~~
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', linewidth=2, label='Line of Best Fit')

# ~~~ Add labels and legend ~~~
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title("Data Visualization with Line of Best Fit")
plt.legend()
plt.grid(True)
plt.show()
