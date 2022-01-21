from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# import para ignorar o erro do certificado expirado
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(x, y), (x_teste, y_teste) = cifar10.load_data()
