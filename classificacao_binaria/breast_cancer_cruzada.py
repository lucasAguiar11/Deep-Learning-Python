import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')


def criarRede():
    classificador = Sequential()
    classificador.add(Dense(
        units=16,
        activation='relu',
        kernel_initializer='random_uniform',
        input_dim=30  # para a primeira camada oculta
    ))

    # add camada de dropout
    # 0.2, vai pegar 20% dos neuronios da camada de entrada e zerar os valores
    classificador.add(Dropout(0.2))

    classificador.add(Dense(
        units=16,
        activation='relu',
        kernel_initializer='random_uniform',
    ))

    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(
        units=1,
        activation='sigmoid'
    ))

    otimizador = keras.optimizers.Adam(
        learning_rate=0.001, decay=0.0001, clipvalue=0.5)

    classificador.compile(
        optimizer=otimizador,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    return classificador


classificador = KerasClassifier(build_fn=criarRede,
                                epochs=100,
                                batch_size=100)

# cv = k do k-fold cross validation, ou seja, quantas vezes a base vai ser dividida
# scoring como vou retornar esses dados
resultados = cross_val_score(estimator=classificador,
                             X=previsores,
                             y=classe,
                             cv=10,
                             scoring='accuracy')
# média do score
media = resultados.mean()

# desvio padrão desses dados (o quanto cada valor está longe da média)
# quanto maior esse valor mais a rede está adaptada demais a rede, ela não consegue trabalhar com
# novos dados - Isso de chama - overfitting
desvio = resultados.std()
