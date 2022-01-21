import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')


classificador = Sequential()
classificador.add(Dense(
    units=8,
    activation='relu',
    kernel_initializer='normal',
    input_dim=30
))

classificador.add(Dropout(0.2))

classificador.add(Dense(
    units=8,
    activation='relu',
    kernel_initializer='random_uniform',
))

classificador.add(Dropout(0.2))

classificador.add(Dense(
    units=1,
    activation='sigmoid'
))

classificador.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

classificador.fit(previsores, classe, batch_size=10, epochs=100)

# estrutura da rede
classificador_json = classificador.to_json()

with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
    
# salvando pesos
classificador.save_weights('classificador_breast.h5')