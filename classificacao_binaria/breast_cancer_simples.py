from sklearn.model_selection import train_test_split
import pandas as pd

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# dividir em classes de trainamento e de teste. No caso 25% de teste

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsores,classe, test_size=0.25)

# https://keras.io/
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential


# Sequential - classe para rede, devido as varias camadas
# Dense - vamos usar camadas densas (cada neuronio é ligado em cada um dos neuronios da canada sequente)
#       - conexão total dos neuronios


# Criando uma nova rede
# Nessa rede add de maneira sequencial as camadas

classificador = Sequential()

# add camada oculta
# units = quantos neuronios camada oculta ( Nº de entradas + Nº Saida ) / 2
# (30+1)/2 = 15.5 -> 16

# função de ativação relu costuma da resultados melhores para deep learning
# kernel_initializer como serão inicializados os pesos
# input_dim quantos elementos na entrada

classificador.add(Dense(
    units = 16,
    activation = 'relu',
    kernel_initializer = 'random_uniform',
    input_dim = 30 # para a primeira camada oculta
))

# outra camada oculta

classificador.add(Dense(
    units = 16,
    activation = 'relu',
    kernel_initializer = 'random_uniform',
))


# camada de saida

# units: Sim ou não, beniguino ou maligno
# units: valor entre 0 e 1
classificador.add(Dense(
    units = 1,
    activation = 'sigmoid'
))

# compilando rede
# optimizer função para o ajustes dos pesos - descida do gradiente
# adam = otimização da descida do gradiente estocatisco - Sempre começar por esse (o mais indicado)
# loss - calculo do erro
# binary_crossentropy = padrão para classificação binária, o mais recomendado

# usando o otimizador padrão

# classificador.compile(
#     optimizer = 'adam',
#     loss = 'binary_crossentropy',
#     metrics=[ 'binary_accuracy' ]
# )

# otimizador

# learning rate (learning_rate) é o "tamanho" do passo para chegar no mínimo global, se for muito grande ele pode passar
# do mínimo, porem é mais "rápido", pois os passos são maiores

# decay, o lr começa com 0.001 ai para cada interção o lr é alterado para lr-=decay (0.001 - 0.0001)
# clipvalue valor máximo que o peso pode chegar 
otimizador = keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue = 0.5 )


classificador.compile(
    optimizer = otimizador,
    loss = 'binary_crossentropy',
    metrics=[ 'binary_accuracy' ]
)

# treinamento da rede
# buscar correlação

# batch_size = o ajuste dos pesos é associado ao calculo do erro desses 10 registros
# estocasticos mas em lote
# epochs = quantas vezes vai repetir o processo

classificador.fit(
    previsores_treinamento, 
    classe_treinamento, 
    batch_size=10, 
    epochs=100 
)


# De fato o aprendizado
# pegar os pesos da camada de entrada para a primeira camada oculta
# e o bias com os seus pesos

pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))

# pegar os pesos da primeira camada oculta para a segunda camada oculta
# e o bias com os seus pesos
pesos1 = classificador.layers[1].get_weights()

# camada oculta para a saida
# e o bias
pesos2 = classificador.layers[2].get_weights()



# realizar as previsões, testar minha rede

previsoes = classificador.predict(previsores_teste)
previsoes = previsoes > 0.5

# medir acerto, comparar vetores da classe com as previsões

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

# Resulta no erro e o acerto no vertor, nessa ordem
resultado = classificador.evaluate(previsores_teste, classe_teste)









