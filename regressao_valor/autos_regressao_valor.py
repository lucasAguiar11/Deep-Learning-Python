from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

# pré-processamento da base

base = pd.read_csv('autos.csv', encoding='ISO-8859-1')

# removendo colunas que não preciso
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)

base = base.drop('name', axis=1)    # muita variação
base = base.drop('seller', axis=1)  # pouca variação
base = base.drop('offerType', axis=1)  # pouca variação

# consultando variação do registro
base['vehicleType'].value_counts()

# i = inconsistencia
# loc = localizar
i1 = base.loc[base.price <= 10]
base.price.mean()
base = base[base.price > 10]

i2 = base.loc[base.price > 350000]
base = base[base.price < 350000]


base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts()  # limousine - 93614

base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts()  # manuell - 266547

base.loc[pd.isnull(base['model'])]
base['model'].value_counts()  # golf - 28989

base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts()  # benzin - 2175829

base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts()  # nein - 259301

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein', }

base = base.fillna(value=valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values


labelEnconder_previsores = LabelEncoder()

previsores[:, 0] = labelEnconder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelEnconder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEnconder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEnconder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelEnconder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEnconder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelEnconder_previsores.fit_transform(previsores[:, 10])

onehotencoder = ColumnTransformer(transformers=[("OneHot",
                                                 OneHotEncoder(),
                                                 [0, 1, 3, 5, 8, 9, 10])],
                                  remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()


regressor = Sequential()
regressor.add(Dense(units=158, activation='relu', input_dim=316))
regressor.add(Dense(units=158, activation='relu'))
regressor.add(Dense(units=1, activation='linear'))

regressor.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_absolute_error'])


regressor.fit(previsores, preco_real, batch_size=300, epochs=100)
previsoes = regressor.predict(previsores)

















