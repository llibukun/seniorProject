# Layiwola Ibukun
# EGN4912: Switchgrass Root and Panicle Analysis
# Created: Monday, July 29th, 2019
# tf_practice.py

# for python 2 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
# The Usual
import numpy as np
import matplotlib.pyplot as plt
# TensorFlow Libraries
import tensorflow as tf
from tensorflow import keras


# show the TensorFlow Version
print(f'TensorFlow Version {tf.__version__}')

# This example uses the Fashion MNIST Dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the Data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



