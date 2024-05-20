# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10   
# En este caso solo usamos tensorflow por la simplicidad de que ya tiene el dataset incluído en la librería

# 1. Carga y Preprocesamiento de Datos
def load_and_preprocess_data():
    # Para que vean que no somos tan malos:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Normalizar los valores de los píxeles
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # Aplanar las imágenes
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    # Aplanar las etiquetas
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return X_train, y_train, X_test, y_test

# 2. Implementación de Funciones de Activación
def relu(x):
    pass  # Completar

def relu_derivative(x):
    pass  # Completar

def softmax(x):
    pass  # Completar

# 3. Función de Pérdida y Métrica de Precisión
def cross_entropy_loss(y_pred, y_true):
    pass  # Completar

def accuracy(y_pred, y_true):
    pass  # Completar

# 4. Inicialización de Pesos y Biases
def initialize_parameters(input_size, hidden_size, output_size):
    pass # Completar

# 5. Forward Propagation
def forward_propagation(X, W1, W2):
    pass  # Completar

# 6. Backward Propagation
def backward_propagation(X, y, z1, a1, z2, a2, W2):
    pass  # Completar

# 7. Actualización de Parámetros
def update_parameters(W1, W2, dW1, dW2, learning_rate):
    pass  # Completar

# 8. Entrenamiento del Modelo
def train(X_train, y_train, epochs, learning_rate):
    pass # Completar


# Main
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    epochs = ...
    learning_rate = ...
    losses, accuracies = train(X_train, y_train, epochs, learning_rate)
    # Grafica las funciones de Loss y Accuracy
