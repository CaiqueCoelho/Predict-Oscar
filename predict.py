from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn import neural_network
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
	k = 10
	scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
	taxa_de_acerto = np.mean(scores)
	
	msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
	print msg
	return taxa_de_acerto

def gettingDistributionOfDatas():
	#Contando a distribuicao dos dados para vencedor - 1 e perdedor - 0
	vencedor = int(list(Y_labels).count(1))
	print('Quantidade vencedores: ' +str(vencedor))
	perdedor = int(list(Y_labels).count(0))
	print('Quantidade perdedores: ' +str(perdedor))
	total = len(Y_labels)
	print('Quantidade total: ' +str(total))
	distribuicao_vencedor = int(vencedor/total * 100)
	distribuicao_perdedor = int(perdedor/total * 100)
	print('\nDistribuicao dados vencedores: ' +str(distribuicao_vencedor) + '%')
	print('Distribuicao dados perdedores: ' +str(distribuicao_perdedor) + '% \n')
	print('class_weight:')
	class_weight_count = {1: distribuicao_vencedor, 0: distribuicao_perdedor}
	print(class_weight_count)

#get dataset to train/test and to predict
df = pd.read_csv('datasets/Base_Director.csv')
df_to_predict = pd.read_csv('datasets/Dados_Director.csv')

df.fillna(0, inplace = True)
df_to_predict.fillna(0, inplace = True)

#getting the importants attributes from original dataset to our dataset to train and test 
X_train_test = df[['BAFTA', 'Golden Globe', 'Guild', 'running_time', 'box_office', 'imdb_score', 'rt_audience_score', 'rt_critic_score', 'produced_USA', 'R', 'PG', 'PG13', 'G', 'q1_release', 'q2_release', 'q3_release', 'q4_release']]
Y_labels = df['Oscar']

#getting the importants attributes from original dataset to our dataset to predict
X_to_predict = df_to_predict[['BAFTA', 'Golden Globe', 'Guild', 'running_time', 'box_office', 'imdb_score', 'rt_audience_score', 'rt_critic_score', 'produced_USA', 'R', 'PG', 'PG13', 'G', 'q1_release', 'q2_release', 'q3_release', 'q4_release']]


# Transforma as variaveis categoricas em variaveis binarias
X_train_test = pd.get_dummies(X_train_test)
X_to_predict = pd.get_dummies(X_to_predict)

# Transforma de data_frames para arrays
X_train_test = X_train_test.values
X_to_predict = X_to_predict.values
Y_labels = Y_labels.values

gettingDistributionOfDatas()

#Getting class_weirght distribution
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_labels), Y_labels)
class_weight_dict = {1: class_weight[0], 0: class_weight[1]}
print(class_weight_dict)
print('')

#Percentage train
porcentagem_de_treino = 0.8
tamanho_de_treino = porcentagem_de_treino * len(Y_labels)
treino_dados = X_train_test[:int(tamanho_de_treino)]
treino_marcacoes = Y_labels[:int(tamanho_de_treino)]


#Percentage test
validacao_dados = X_train_test[int(tamanho_de_treino):]
validacao_marcacoes = Y_labels[int(tamanho_de_treino):]

#Contando quantidade de predicoes como ganhas
qtd_candidatos = len(X_to_predict)
candidato = [0] * qtd_candidatos
candidato0 = [0] * qtd_candidatos
resultados = {}

#Predict Multinomial Naive Bayes
modelo = MultinomialNB(alpha= 0.001, fit_prior= True)
resultado = fit_and_predict("MultinomialNB Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = MultinomialNB()
resultado = fit_and_predict("MultinomialNB", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

#Predixt Adaboost
modelo = AdaBoostClassifier(n_estimators= 45, learning_rate= 0.01)
resultado = fit_and_predict("AdaBoostClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1


modelo = AdaBoostClassifier()
resultado = fit_and_predict("AdaBoostClassifier", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

#Predict SVC
modelo = svm.SVC(C= 0.03, max_iter= -1, decision_function_shape= 'ovo', tol= 0.001, class_weight= None)
resultado = fit_and_predict("SVC Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = svm.SVC(C= 0.03, max_iter= -1, decision_function_shape= 'ovo', tol= 0.001, class_weight= 'balanced')
resultado = fit_and_predict("SVC Grided and class_weight balanced", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = svm.SVC(C= 0.03, max_iter= -1, decision_function_shape= 'ovo', tol= 0.001, class_weight= class_weight_dict)
resultado = fit_and_predict("SVC Grided and class_weight count", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = svm.SVC()
resultado = fit_and_predict("SVC", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1


#Predict KNN
modelo = neighbors.KNeighborsClassifier(n_neighbors = 4, metric = 'euclidean', weights = 'uniform', algorithm = 'auto')
resultado = fit_and_predict("KNN Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = neighbors.KNeighborsClassifier()
resultado = fit_and_predict("KNN", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1


#Predict MLP
modelo = neural_network.MLPClassifier(hidden_layer_sizes=(200, 500), max_iter = 100)
resultado = fit_and_predict("MLP Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = neural_network.MLPClassifier()
resultado = fit_and_predict("MLP", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1


#Predict Logistic Regression
modelo = LogisticRegression(C= 0.01, fit_intercept= True, tol= 1e-05, max_iter= 1, class_weight= None)
resultado = fit_and_predict("LogisticRegression Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = LogisticRegression(C= 0.01, fit_intercept= True, tol= 1e-05, max_iter= 1, class_weight= 'balanced')
resultado = fit_and_predict("LogisticRegression Grided and class_weigh balanced", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = LogisticRegression(C= 0.01, fit_intercept= True, tol= 1e-05, max_iter= 1, class_weight= class_weight_dict)
resultado = fit_and_predict("LogisticRegression Grided and class_weight count", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = LogisticRegression()
resultado = fit_and_predict("LogisticRegression", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1


#Predict Bagging
modelo = BaggingClassifier(n_estimators= 42, max_samples= 0.087)
resultado = fit_and_predict("BaggingClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = BaggingClassifier()
resultado = fit_and_predict("BaggingClassifier", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1


#Predict Gradient Boosting
modelo = GradientBoostingClassifier(n_estimators= 25, learning_rate= 0.2, max_depth= 1)
resultado = fit_and_predict("GradientBoostingClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = GradientBoostingClassifier()
resultado = fit_and_predict("GradientBoostingClassifier", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

#Predict Random Forest
modelo = RandomForestClassifier(n_estimators= 10, criterion= 'gini', max_depth= 3)
resultado = fit_and_predict("RandomForestClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = RandomForestClassifier(n_estimators= 10000, criterion= 'gini', max_depth= 3, class_weight= class_weight_dict)
resultado = fit_and_predict("RandomForestClassifier Grided and class_weight count", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1



#Predict Extra Tress
modelo = ExtraTreesClassifier(criterion = 'gini', max_depth = 3, n_estimators = 150)
resultado = fit_and_predict("ExtraTreesClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = ExtraTreesClassifier(criterion = 'gini', max_depth = 3, n_estimators = 150, class_weight= 'balanced')
resultado = fit_and_predict("ExtraTreesClassifier Grided and class_weight balanced", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = ExtraTreesClassifier(criterion = 'gini', max_depth = 3, n_estimators = 150, class_weight= class_weight_dict)
resultado = fit_and_predict("ExtraTreesClassifier Grided and class_weight count", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = ExtraTreesClassifier()
resultado = fit_and_predict("ExtraTreesClassifier", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1


from sklearn import tree
modelo = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
resultado = fit_and_predict("DecisionTree Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, class_weight= 'balanced')
resultado = fit_and_predict("DecisionTree Grided and class_weight balanced", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, class_weight= class_weight_dict)
resultado = fit_and_predict("DecisionTree Grided and class_weight count", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

modelo = tree.DecisionTreeClassifier()
resultado = fit_and_predict("DecisionTree", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X_train_test, Y_labels)
resultadoOscar = modelo.predict(X_to_predict)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1




#A eficacia do algoritmo que chuta tudo 0 ou 1 ou um unico valor
acerto_base = max(Counter(validacao_marcacoes).itervalues()) #Devolve a quantidade do maior elemento
acerto_de_um = list(Y_labels).count('sim')
acerto_de_zero = list(Y_labels).count('nao')
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base nos dados de validacao: %f" %taxa_de_acerto_base)

#print resultados
maximo = max(resultados)
vencedor = resultados[maximo]
print('\n\n')
print(vencedor)
print('\n\n')

total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = metrics.accuracy_score(validacao_marcacoes, resultado)

print("Taxa de acerto do algoritmo melhor no mundo real" + " foi de: " 
	+ str(taxa_de_acerto) + "% " + "de " + str(total_de_elementos) + " elementos\n\n")


name_and_films = df_to_predict[['film', 'name']]
print(name_and_films)
print('\n')
vencedor.fit(X_train_test, Y_labels)
vencedor_result = vencedor.predict(X_to_predict)
print(name_and_films.iloc[[vencedor_result.index(max(vencedor_result))]])

print('\nWithout accuracy validation:')
print(candidato0)
print(name_and_films.iloc[[candidato0.index(max(candidato0))]])
print("\n")
print('\nOnly if accuracy > 79%:')
print(candidato)
print(name_and_films.iloc[[candidato.index(max(candidato))]])