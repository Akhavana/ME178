import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
data = pd.read_csv('mental_health4.csv')

# Split data into features (X) and target (y)
X = data.drop(columns='Mental_Health_Status')
y = data['Mental_Health_Status']

# Select two features for visualization (e.g., Feature1, Feature2)
X = X[['Gaming_Hours', 'Sleep_Hours']]  # Replace with actual feature names

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a classifier
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Create a grid for visualization
d1_grid, d2_grid = np.meshgrid(
    np.arange(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 0.02),
    np.arange(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 0.02)
)

# Flatten the grid and make predictions
X0 = d1_grid.ravel()
X1 = d2_grid.ravel()
d12_array = np.c_[X0, X1]  # Combine into feature matrix
y_array = model.predict(d12_array)  # Predict on grid points
y_grid = y_array.reshape(d1_grid.shape)  # Reshape predictions for plotting

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(d1_grid, d2_grid, y_grid, alpha=0.3, cmap='viridis')

# Scatter plot of the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap='viridis', label='Training Data')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary")
plt.colorbar()
plt.legend()
plt.show()
