import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.utils import np_utils
from keras.layers import BatchNormalization

(x_treinamento, y_treinamento), (x_teste, y_teste) = mnist.load_data()

plt.imshow(x_treinamento[0], cmap='gray')
plt.title('Classe ' + str(y_treinamento[0]))

# deixo as imagens em escala de cinza, já que não preciso da cor para identificar
# além que eu diminuo a quantidade de entradas

previsores_treinamento = x_treinamento.reshape(
    x_treinamento.shape[0], 28, 28, 1)
previsores_teste = x_teste.reshape(x_teste.shape[0], 28, 28, 1)

# trocando o tipo de dados
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# diminuindo as estradas de 0 a 255 para 0 a 1
previsores_treinamento /= 255
previsores_teste /= 255


classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()

# Camada de convolução
# 32 mapas de caracteristicas, no caso vão ser usados 32 kernels, recomendável 64 para iniciar
# tamanho do detector de caracteristicas, ou seja, a matriz que vou multiplicar a imagem (3,3)
# input_shape = Como é a imagem que vou trabalhar?
classificador.add(
    Conv2D(32,
           (3, 3),
           input_shape=(28, 28, 1),
           activation='relu')
)

# Normalização de 0 a 1 nos mapas de caracteristicas
classificador.add(BatchNormalization())

# pooling
classificador.add(MaxPool2D(pool_size=(2, 2)))

# flattening
# classificador.add(Flatten())


# Nova camada de convolução

classificador.add(
    Conv2D(32, (3, 3), activation='relu')
)

classificador.add(BatchNormalization())
classificador.add(MaxPool2D(pool_size=(2, 2)))
classificador.add(Flatten()) # flattening apenas no final

# rede neural densa

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=10, activation='softmax'))

classificador.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
# validation_data: faz o treinamento, depois faz o teste de maneira automática, para cada epoca
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size=128, epochs=5, validation_data=(previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)
