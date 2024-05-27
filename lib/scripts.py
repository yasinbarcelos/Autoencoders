import matplotlib.pyplot as plt
import numpy as np    
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping
from lib.models import *


def load_and_preprocess_mnist(noise_factor=0):
    # Carregando o dataset MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizando os dados para o intervalo [0, 1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Remodelando os dados para vetores de 784 dimensões (28*28)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # Dividindo o conjunto de treinamento em treinamento e validação
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Adicionando ruído aos dados
    x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
    x_val_noisy = x_val + noise_factor * tf.random.normal(shape=x_val.shape)
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

    # Garantindo que os valores permaneçam no intervalo [0, 1]
    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
    x_val_noisy = tf.clip_by_value(x_val_noisy, clip_value_min=0., clip_value_max=1.)
    x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

    return (x_train_noisy, x_train, y_train), (x_val_noisy, x_val, y_val), (x_test_noisy, x_test, y_test)

def visualize_data(x_original, x_noisy, x_decoded, n=10):
    """
    Visualiza os dados originais, ruidosos e reconstruídos.

    Args:
        x_original: Dados originais.
        x_noisy: Dados com ruído (pode ser None).
        x_decoded: Dados reconstruídos pelo autoencoder.
        n: Número de exemplos a serem visualizados. Default é 10.
    """
    rows = 3 if x_noisy is not None else 2  # Definindo o número de linhas para a visualização

    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Dados originais
        ax = plt.subplot(rows, n, i + 1)
        plt.imshow(x_original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if not np.array_equal(x_original, x_noisy):
            # Dados ruidosos
            ax = plt.subplot(rows, n, i + 1 + n)
            plt.imshow(x_noisy[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # Dados reconstruídos
        ax = plt.subplot(rows, n, i + 1 + (2 if x_noisy is not None else 1) * n)
        plt.imshow(x_decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
        
def plot_history(history):
    """
    Plota a perda de treinamento e validação ao longo das épocas.

    Args:
        history: Histórico de treinamento retornado pelo método `fit` do Keras.
    """
    plt.figure(figsize=(12, 6))
    
    # Plotando a perda de treinamento e validação
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()     
