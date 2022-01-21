from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')

base = base.drop('Other_Sales', axis=1)
base = base.drop('Developer', axis=1)
base = base.drop('Name', axis=1)
base = base.drop('NA_Sales', axis=1)
base = base.drop('JP_Sales', axis=1)
base = base.drop('EU_Sales', axis=1)

base = base.dropna(axis=0)

base = base.loc[base['Global_Sales'] > 1]

previsores = base.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].values
valor_global = base.iloc[:, 4].values


labelEncoder = LabelEncoder()

previsores[:, 0] = labelEncoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelEncoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelEncoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelEncoder.fit_transform(previsores[:, 8])

onehotencoder = ColumnTransformer(transformers=[(
    "OneHot", OneHotEncoder(), [0, 2, 3, 8])], remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()


camada_entrada = Input(shape=99)

relu = Activation('relu')

camada_oculta1 = Dense(units=50, activation=relu)(camada_entrada)
camada_drop1 = Dropout(0.2)(camada_oculta1)
camada_oculta2 = Dense(units=50, activation=relu)(camada_oculta1)
camada_drop2 = Dropout(0.2)(camada_oculta2)

camada_saida = Dense(units=1, activation='linear')(camada_oculta2)

regressor = Model(inputs=camada_entrada, outputs=[camada_saida])

regressor.compile(optimizer='adam', loss='mse')

regressor.fit(previsores, valor_global, batch_size=100, epochs=5000)

previsao_total = regressor.predict(previsores)

previsao_total.mean()
valor_global.mean()
