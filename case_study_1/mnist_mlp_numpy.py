import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------
# 1. Load and Preprocess MNIST Dataset
# ---------------------------------------------------------
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.values / 255.0  # Normalize pixel values
y = mnist.target.astype(int).values.reshape(-1, 1)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False) # Updated for newer sklearn versions
y = encoder.fit_transform(y)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 2. Define Activation Functions
# ---------------------------------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return x > 0

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ---------------------------------------------------------
# 3. Network Architecture & Initialization
# ---------------------------------------------------------
input_size = 784
hidden1 = 256
hidden2 = 64
output_size = 10

# Weight initialization (Random small values)
W1 = np.random.randn(input_size, hidden1) * 0.01
b1 = np.zeros((1, hidden1))

W2 = np.random.randn(hidden1, hidden2) * 0.01
b2 = np.zeros((1, hidden2))

W3 = np.random.randn(hidden2, output_size) * 0.01
b3 = np.zeros((1, output_size))

# Hyperparameters
learning_rate = 0.01
epochs = 50

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
for epoch in range(epochs):
    # --- Forward Propagation ---
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    
    # --- Loss Calculation (Cross-Entropy) ---
    loss = -np.mean(y_train * np.log(A3 + 1e-8))
    
    # --- Backpropagation ---
    # Output Layer
    dZ3 = A3 - y_train
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    
    # Hidden Layer 2
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden Layer 1
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # --- Update Weights ---
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# ---------------------------------------------------------
# 5. Evaluation / Testing
# ---------------------------------------------------------
Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = relu(Z2)

Z3 = np.dot(A2, W3) + b3
A3 = softmax(Z3)

predictions = np.argmax(A3, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == true_labels)
print("Test Accuracy:", accuracy)
