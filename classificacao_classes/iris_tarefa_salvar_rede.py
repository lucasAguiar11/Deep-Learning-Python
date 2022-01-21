import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.utils import np_utils

base = pd.read_csv('iris.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)


classificador = Sequential()

classificador.add(Dense(
    units=16,
    activation='relu',
    input_dim=4
))

classificador.add(Dropout(0.2))

classificador.add(Dense(
    units=16,
    activation='relu'
))

classificador.add(Dropout(0.2))

classificador.add(Dense(
    units=3,
    activation='softmax'
))

classificador.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics='accuracy')

classificador.fit(previsores, classe, batch_size=10, epochs=1000)


classificador_json = classificador.to_json()

with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
    

classificador.save_weights('classificador_iris.h5')


arquivo = open('classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador_carregado = model_from_json(estrutura_rede)
classificador_carregado.load_weights('classificador_iris.h5')

# após isso a rede esta pronta para o uso

novo = np.array([[15.80, 8.34, 1.18, 9.00]])

previsao = classificador.predict(novo)
previsao = previsao > 0.5
