import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)


def criar_rede(neurons, metrics, dropout, activation):
    classificador = Sequential()

    classificador.add(Dense(
        units=neurons,
        activation=activation,
        input_dim=4
    ))

    classificador.add(Dropout(dropout))

    classificador.add(Dense(
        units=neurons,
        activation=activation
    ))

    classificador.add(Dropout(dropout))

    classificador.add(Dense(
        units=3,
        activation='softmax'
    ))

    classificador.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                          metrics=metrics)
    return classificador


classificador = KerasClassifier(build_fn=criar_rede)
parametros = {
    'batch_size': [10],
    'epochs': [500],
    'neurons': [16],
    'metrics': ['accuracy'],
    'dropout': [0.2],
    'activation': ['relu'],
}


grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros, cv=10)

grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

# 0.946666669845581
# {'activation': 'relu', 'batch_size': 10, 'dropout': 0.2, 'epochs': 1000, 'metrics': 'accuracy', 'neurons': 16}