#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_wine

# Load the Wine Quality dataset from scikit-learn
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='quality')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error and R-squared score to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Create a scatter plot of actual vs. predicted wine quality using seaborn
plt.figure(figsize=(8, 6))

# Set the style using seaborn
sns.set(style='whitegrid', font_scale=1.2)

# Plot the data points with a color map to represent the density
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, cmap='Blues', s=100,label='Data Points')

# Plot the regression line
sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color='red', linestyle='--',label='Regression Line')

# Add labels and title
plt.xlabel('Actual Wine Quality', fontsize=14)
plt.ylabel('Predicted Wine Quality', fontsize=14)
plt.title('Actual vs. Predicted Wine Quality', fontsize=16)

# Add grid lines
plt.grid(True)

plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# In[ ]:




