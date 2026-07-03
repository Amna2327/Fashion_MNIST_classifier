import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # normalize
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # split
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, train_size=54000, random_state=42
    )
    
    # reshape
    train_images = np.expand_dims(train_images, axis=-1)
    val_images = np.expand_dims(val_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    
    return train_images, val_images, test_images, train_labels, val_labels, test_labels