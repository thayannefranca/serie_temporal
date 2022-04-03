import math
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import svm


dados = pd.read_csv('jena_climate_2009_2016.csv')

# Utilizando uma subamostra dos dados de intervalos de 10 minutos para intervalos de uma hora
dados = dados[5::6]

#Tratamento da coluna “Date Time” para converter a coluna de object para datetime 
dados["Date Time"]= pd.to_datetime(dados["Date Time"])

# Utilizar tal coluna como o índice das linhas do DataFrame
#Assim focaremos nos dados que queremos
dados = dados.set_index('Date Time')

#Obtendo a variável independente
y = dados['T (degC)']
#Obtendo as variáveis explicativas
#O r2 nos dará a resposta se as colunas selecionadas são o suficiente para entender
X = dados[['p (mbar)','rh (%)','VPact (mbar)','wv (m/s)','max. wv (m/s)','wd (deg)']]

#Separando os dados de treino e teste
#70% será usado para treinamento e 30% para testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)


def previsao_y(modelo):
  #Realizando o treinamento do modelo
  modelo.fit(X_train, y_train)
  y_pred = modelo.predict(X_test)
  #Avaliando pelas métricas mse, rmse e r2
  mse = mean_squared_error(y_test, y_pred)
  rmse =  math.sqrt(mse)
  r2 = r2_score(y_test, y_pred)

  print('{} obteve: \n mse: {}\n rmse:{}\n r2: {}'.format(modelo, mse, rmse, r2))

  return mse, rmse, r2


def main():

    mlflow.set_experiment('variacao-clima')
    with mlflow.start_run():
        classifiers = [
            svm.SVR(),
            LinearRegression()]

        for item in classifiers:
            modelo = item
            mse, rmse, r2 = previsao_y(modelo)
            mlflow.log_metric('mse',mse)
            mlflow.log_metric('rmse',rmse)
            mlflow.log_metric('r2',r2)

#Nos dados acima, o modelo de regressão linear obtive a melhor acurácia (R2) em comparação ao SVM.