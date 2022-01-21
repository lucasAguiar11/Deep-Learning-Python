import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)

classe_dummy = np_utils.to_categorical(classe)
# iris setosa           1 0 0
# iris virginica        0 1 0
# iris versicolor       0 0 1


def criar_rede():
    classificador = Sequential()

    classificador.add(Dense(
        units=4,
        activation='relu',
        input_dim=4
    ))

    classificador.add(Dense(
        units=4,
        activation='relu'
    ))

    classificador.add(Dense(
        units=3,
        activation='softmax'
    ))

    classificador.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
    return classificador


classificador = KerasClassifier(
    build_fn=criar_rede,
    epochs=1000,
    batch_size=10)

resultados = cross_val_score(
    estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

media = resultados.mean()
desvio = resultados.std()