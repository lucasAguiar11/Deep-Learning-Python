from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model


base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis=1)
base = base.drop('Global_Sales', axis=1)
base = base.drop('Developer', axis=1)

# remove todas a linhas que tem NAN
base = base.dropna(axis=0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base['Name']

base = base.drop('Name', axis=1)

previsores = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
vendas_na = base.iloc[:, 4].values
vendas_eu = base.iloc[:, 5].values
vendas_jp = base.iloc[:, 6].values


labelEncoder = LabelEncoder()

previsores[:, 0] = labelEncoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelEncoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelEncoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelEncoder.fit_transform(previsores[:, 8])


# isso faz isso, cria as variáveis tipo dummy
# 0 0 0 1
# 1 0 0 0
onehotencoder = ColumnTransformer(transformers=[(
    "OneHot", OneHotEncoder(), [0, 2, 3, 8])], remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()


camada_entrada = Input(shape=(61, ))
camada_oculta1 = Dense(units=32, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units=32, activation='sigmoid')(camada_oculta1)

camada_saida1 = Dense(units=1, activation='linear')(camada_oculta2)
camada_saida2 = Dense(units=1, activation='linear')(camada_oculta2)
camada_saida3 = Dense(units=1, activation='linear')(camada_oculta2)

regressor = Model(inputs=camada_entrada,
                  outputs=[camada_saida1, camada_saida2, camada_saida3])

# mse - Penaliza o erro, já que ele faz o erro ao quadrado. Tende a ser mais eficiente
regressor.compile(optimizer='adam', loss='mse')
regressor.fit(previsores, [vendas_na, vendas_eu,
              vendas_jp], epochs=10000, batch_size=100)


previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)








