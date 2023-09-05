#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset from scikit-learn
iris_data = load_iris()
X=iris_data.data
y=iris_data.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Now, let's add the user interactive part to predict species for new flowers
print("\n--- Predicting Species for New Flowers ---")
while True:
    sepal_length = float(input("Enter the sepal length: "))
    sepal_width = float(input("Enter the sepal width: "))
    petal_length = float(input("Enter the petal length: "))
    petal_width = float(input("Enter the petal width: "))

    # Reshape the user input into a 2D array to make predictions (one sample, four features)
    new_sample = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    # Use the trained model to predict the species of the new flower
    predicted_species = model.predict(new_sample)

    # Convert the predicted species code to the actual species name
    species_names = data.target_names
    predicted_species_name = species_names[predicted_species][0]

    print(f"Predicted Species for the New Flower: {predicted_species_name}")

    # Ask if the user wants to enter another example
    another_example = input("Do you want to enter another example? (yes/no): ").lower()
    if another_example != 'yes':
        break

