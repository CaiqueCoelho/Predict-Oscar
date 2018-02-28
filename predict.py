from __future__ import division
import pandas as pd
#contador
from collections import Counter
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.utils import class_weight

#data_frame, importando planilha

#Atriz Coadjuvante
#df = pd.read_csv('Base_SupActress.csv')
#df_futuro = pd.read_csv('Dados_SupActress.csv')

#Ator Coadjuvante
#df = pd.read_csv('Base_SupActor.csv')
#df_futuro = pd.read_csv('Dados_SupActor.csv')

#Atriz
#df = pd.read_csv('Base_Actress.csv')
#df_futuro = pd.read_csv('Dados_Actress.csv')

#Ator
#df = pd.read_csv('Base_Actor.csv')
#df_futuro = pd.read_csv('Dados_Actor.csv')

#Filme
#df = pd.read_csv('Base_Picture.csv')
#df_futuro = pd.read_csv('Dados_Picture.csv')

#Diretor
df = pd.read_csv('Base_Director.csv')
df_futuro = pd.read_csv('Dados_Director.csv')

#Adaptacao
#df = pd.read_csv('Base_Adapted.csv')
#df_futuro = pd.read_csv('Dados_Adapted.csv')

#Original
#df = pd.read_csv('Base_Original.csv')
#df_futuro = pd.read_csv('Dados_Original.csv')

X_df = df[['BAFTA', 'Golden Globe', 'Guild', 'running_time', 'box_office', 'imdb_score', 'rt_audience_score', 'rt_critic_score', 'produced_USA', 'R', 'PG', 'PG13', 'G', 'q1_release', 'q2_release', 'q3_release', 'q4_release']]
Y_df = df['Oscar']


X_df_futuro = df_futuro[['BAFTA', 'Golden Globe', 'Guild', 'running_time', 'box_office', 'imdb_score', 'rt_audience_score', 'rt_critic_score', 'produced_USA', 'R', 'PG', 'PG13', 'G', 'q1_release', 'q2_release', 'q3_release', 'q4_release']]


# Transforma as variaveis categoricas em variaveis binarias
Xdummies_df = pd.get_dummies(X_df)
Xdummies_df_futuro = pd.get_dummies(X_df_futuro)
Ydummies_df = Y_df

# Transforma de data_frames para arrays
X = Xdummies_df.values
X_futuro = Xdummies_df_futuro.values
Y = Ydummies_df.values

#Contando a distribuicao dos dados para vencedor - 1 e perdedor - 0
vencedor = int(list(Y).count(1))
print('Quantidade vencedores: ' +str(vencedor))
perdedor = int(list(Y).count(0))
print('Quantidade perdedores: ' +str(perdedor))
total = len(Y)
print('Quantidade total: ' +str(total))
distribuicao_vencedor = int(vencedor/total * 100)
distribuicao_perdedor = int(perdedor/total * 100)
print('')
print('Distribuicao dados vencedores: ' +str(distribuicao_vencedor) + '%')
print('Distribuicao dados perdedores: ' +str(distribuicao_perdedor) + '%')
print('')

print('class_weight:')
class_weight_count = {1: distribuicao_vencedor, 0: distribuicao_perdedor}
print(class_weight_count)

class_weight = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
class_weight_dict = {1: class_weight[0], 0: class_weight[1]}
print(class_weight_dict)
print('')

#Treino do algorimo
porcentagem_de_treino = 0.8

#treino 0:799
tamanho_de_treino = porcentagem_de_treino * len(Y)
treino_dados = X[:int(tamanho_de_treino)]
treino_marcacoes = Y[:int(tamanho_de_treino)]


#validacao 800:999
validacao_dados = X[int(tamanho_de_treino):]
validacao_marcacoes = Y[int(tamanho_de_treino):]

#Contando quantidade de predicoes como ganhas
qtd_candidatos = len(X_futuro)
candidato = [0] * qtd_candidatos
candidato0 = [0] * qtd_candidatos


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
	k = 10
	scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
	taxa_de_acerto = np.mean(scores)
	
	msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
	print msg
	return taxa_de_acerto


resultados = {}

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB(alpha= 0.001, fit_prior= True)
resultado = fit_and_predict("MultinomialNB Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

'''
param_grid = {
	'alpha': (0.001, 0.0009, 0.002),
	'fit_prior': (True, False)
}

multinomialNB = MultinomialNB()

multinomialNB_grid = GridSearchCV(multinomialNB, param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
multinomialNB_grid.fit(X, Y)
multinomialNBBestParams = multinomialNB_grid.best_params_
multinomialNB_clf = multinomialNB_grid.best_estimator_
print("MultinomialNB Best Params: ")
print(multinomialNBBestParams)
print("")
'''

from sklearn.ensemble import AdaBoostClassifier
modelo = AdaBoostClassifier(n_estimators= 45, learning_rate= 0.01)
resultado = fit_and_predict("AdaBoostClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

'''
param_grid = {
	'n_estimators': (45, 47, 43),
	'learning_rate': (0.02, 0.01),
}

adaBoost = AdaBoostClassifier()

ada_grid = GridSearchCV(adaBoost, param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
ada_grid.fit(X, Y)
adaBestParams = ada_grid.best_params_
ada_clf = ada_grid.best_estimator_
print("AdaBoost Best Params: ")
print(adaBestParams)
print("")
'''

from sklearn import svm
modelo = svm.SVC(C= 0.03, max_iter= -1, decision_function_shape= 'ovo', tol= 0.001, class_weight= None)
resultado = fit_and_predict("SVC Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

'''
'kernel': ['linear', 'rbf', 'poly'], 
param_grid = {
	'tol': (0.001, 0.0015),
	'C': [0.03, 0.035],
	'decision_function_shape': ('ovo', 'ovr'),
	'max_iter': [-1, 200, 2000],
	'class_weight': [None, 'balanced'],
}
random_search = GridSearchCV(modelo, param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
random_search.fit(X, Y)
svmBestParams = random_search.best_params_
print("SVM Best Params: ")
print(svmBestParams)
print("")
'''

from sklearn import neighbors
modelo = neighbors.KNeighborsClassifier(n_neighbors = 4, metric = 'euclidean', weights = 'uniform', algorithm = 'auto')
resultado = fit_and_predict("KNN Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

'''
knn = neighbors.KNeighborsClassifier()
param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 
	'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	'metric': ['euclidean', 'minkowski'], 
	'weights': ['uniform', 'distance']
}
random_search = GridSearchCV(knn, param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
random_search.fit(X, Y)
knnBestParams = random_search.best_params_
knn_clf = random_search.best_estimator_
print("KNN Best Params: ")
print(knnBestParams)
print("")
'''

from sklearn import neural_network
modelo = neural_network.MLPClassifier(hidden_layer_sizes=(200, 500), max_iter = 100)
resultado = fit_and_predict("MLP Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

from sklearn.linear_model import LogisticRegression
modelo = LogisticRegression(C= 0.01, fit_intercept= True, tol= 1e-05, max_iter= 1, class_weight= None)
resultado = fit_and_predict("LogisticRegression Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

'''
param_grid = {'C': [0.01, 0.015, 0.009],
				'tol': (0.00001, 0.000001),
				'fit_intercept': (True, False),
				'class_weight': (None, 'balanced'),
				'max_iter': (1, 2, 3, 4, 5, 6),
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
grid_search.fit(X, Y)
logisticRegressionBestParams = grid_search.best_params_
print("LogisticRegression Best Params: ")
print(logisticRegressionBestParams)
print("")
'''

from sklearn.ensemble import BaggingClassifier
modelo = BaggingClassifier(n_estimators= 42, max_samples= 0.087)
resultado = fit_and_predict("BaggingClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

'''
param_grid = {
    'max_samples' : [0.083, 0.085, 0.087],
    'n_estimators': (42, 41), 
}

bagging_grid = GridSearchCV(BaggingClassifier(), param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
bagging_grid.fit(X, Y)
baggingBestParams = bagging_grid.best_params_

print("Bagging Best Params: ")
print(baggingBestParams)
print("")
'''

from sklearn.ensemble import GradientBoostingClassifier
modelo = GradientBoostingClassifier(n_estimators= 25, learning_rate= 0.2, max_depth= 1)
resultado = fit_and_predict("GradientBoostingClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1
'''
param_grid = {
	'learning_rate': (0.2, 0.1),
	'n_estimators': (22, 25), 
	'max_depth': (None, 1, 2, 3),
}

gradientBoosting = GradientBoostingClassifier()

gradientBoosting_grid = GridSearchCV(gradientBoosting, param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
#TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
gradientBoosting_grid.fit(X, Y)
gradientBoostingBestParams = gradientBoosting_grid.best_params_

print("Gradient Boosting Best Params: ")
print(gradientBoostingBestParams)
print("")
'''

from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators= 10, criterion= 'gini', max_depth= 3)
resultado = fit_and_predict("RandomForestClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1



'''
param_grid = {
				'n_estimators': (1500, 1000, 800),
				'criterion': ('gini', 'entropy'),
				'max_depth': (None, 2, 3, 4, 5),
}

random = RandomForestClassifier()

random_grid = GridSearchCV(random, param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
random_grid.fit(X, Y)
randomBestParams = random_grid.best_params_

print("Random Forest Best Params: ")
print(randomBestParams)
print("")
'''

from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier(criterion = 'gini', max_depth = 3, n_estimators = 150)
resultado = fit_and_predict("ExtraTreesClassifier Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1
'''
params = {
			'criterion' : ["gini", "entropy"],
			'max_depth': [2, 3, 4],
			'n_estimators':(150, 125, 170),
          }

extraTrees = ExtraTreesClassifier()
extraTrees_grid = GridSearchCV(extraTrees, params, scoring = 'accuracy', verbose = 1, n_jobs=-1)
extraTrees_grid.fit(X, Y)
extraTreesBestParams = extraTrees_grid.best_params_
print("Extra Trees Best Params: ")
print(extraTreesBestParams)
print("")
'''

from sklearn import tree
modelo = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
resultado = fit_and_predict("DecisionTree Grided", modelo, treino_dados, treino_marcacoes)
resultados[resultado] = modelo
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
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
modelo.fit(X, Y)
resultadoOscar = modelo.predict(X_futuro)
print('Oscar 2018: ' + str(resultadoOscar))
print('')
for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato0[i] = candidato0[i] + 1
if(resultado > 0.79):
	for i in range(qtd_candidatos):
		if(resultadoOscar[i] == 1.0):
			candidato[i] = candidato[i] + 1

'''
decision_tree = tree.DecisionTreeClassifier()
param_grid = {'max_depth': [None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
'criterion': ('gini', 'entropy'),
'random_state': (None, 0, 42)
}
random_search = GridSearchCV(decision_tree, param_grid, cv = 3, scoring = 'accuracy', verbose = 1, n_jobs=-1)
random_search.fit(X, Y)
decisionTreeBestParams = random_search.best_params_
print("Decision Tree Best Params: ")
print(decisionTreeBestParams)
print("")
'''


#A eficacia do algoritmo que chuta tudo 0 ou 1 ou um unico valor
acerto_base = max(Counter(validacao_marcacoes).itervalues()) #Devolve a quantidade do maior elemento
acerto_de_um = list(Y).count('sim')
acerto_de_zero = list(Y).count('nao')
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base nos dados de validacao: %f" %taxa_de_acerto_base)

#print resultados
maximo = max(resultados)

vencedor = resultados[maximo]
print('')
print('')
print vencedor
print('')
print('')

vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
print(resultado)


#diferencas = resultado - teste_marcacoes para 0 ou 1 no resultado
#acertos = (resultado == validacao_marcacoes)

#acertos = [d for d in diferencas if d == True] #para 0 ou 1 no resultado
#total_de_acertos = sum(acertos) #len(acertos) para 0 ou 1 no resultado
total_de_elementos = len(validacao_marcacoes)
#taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos

taxa_de_acerto = metrics.accuracy_score(validacao_marcacoes, resultado)

print("Taxa de acerto do algoritmo melhor no mundo real" + " foi de: " + str(taxa_de_acerto) + "% " + "de " + str(total_de_elementos) + " elementos")


name_and_films = df_futuro[['film', 'name']]
print(name_and_films)
print('')
vencedor.fit(X, Y)
print(vencedor.predict(X_futuro))

print('')
print('Without accuracy validation:')
print(candidato0)
print('')
print('Only if accuracy > 79%:')
print(candidato)