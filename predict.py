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
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn import tree

#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler

def fit_and_predict(modelName, model, train_x, train_y):
    k = 10
    scores = cross_val_score(model, train_x, train_y, cv = k)
    hit_hate = np.mean(scores)
    
    msg = "Hit Rate from {0}: {1}".format(modelName, hit_hate)
    print(msg)
    
    return hit_hate

def gettingDistributionOfDatas():
    winners = int(list(Y_labels).count(1))
    print('Quantity winners: ' +str(winners))
    
    losers = int(list(Y_labels).count(0))
    print('Quantity losers: ' +str(losers))
    
    total = len(Y_labels)
    print('Total Quantity: ' +str(total))
    
    distribution_winners = int(winners/total * 100)
    distribution_losers = int(losers/total * 100)
    print('\nDistribution of winning data: ' +str(distribution_winners) + '%')
    print('Distribution of losing data: ' +str(distribution_losers) + '% \n')
    
    print('class_weight:')
    class_weight_count = {1: distribution_winners, 0: distribution_losers}
    print(class_weight_count)
    
    return class_weight_count

#get dataset to train/test and to predict
df = pd.read_csv('datasets/Base_Actor.csv')
df_to_predict = pd.read_csv('datasets/Dados_Actor.csv')

#check if exist any NaN values
df.isnull().values.any()

#check if exist any NaN values
df_to_predict.isnull().values.any()

df_to_predict.head()

# if exist any Nan values we need to handle with this
#imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
#df_to_predict_np_array =  np.array(df_to_predict) 
#imputer = imputer.fit(df_to_predict_np_array[:, 4:])
# df_to_predict_np_array[:, 4:] = imputer.transform(X[:, :])

#getting the importants attributes from original dataset to our dataset to train and test 
X_train_test = df[['BAFTA', 'Golden Globe', 'Guild', 'running_time', 'box_office', 'imdb_score', 'rt_audience_score', 'rt_critic_score', 'produced_USA', 'R', 'PG', 'PG13', 'G', 'q1_release', 'q2_release', 'q3_release', 'q4_release']]
Y_labels = df['Oscar']

#getting the importants attributes from original dataset to our dataset to predict
X_to_predict = df_to_predict[['BAFTA', 'Golden Globe', 'Guild', 'running_time', 'box_office', 'imdb_score', 'rt_audience_score', 'rt_critic_score', 'produced_USA', 'R', 'PG', 'PG13', 'G', 'q1_release', 'q2_release', 'q3_release', 'q4_release']]

# to improve our analysis, we will do a pre-processing to normalize the data
attributes_to_normalize = ['running_time', 'box_office', 'imdb_score', 'rt_audience_score', 'rt_critic_score']
X_train_robust = X_train_test.copy()
X_train_to_predict = X_to_predict

X_train_robust[attributes_to_normalize] = RobustScaler().fit_transform(X_train_test[attributes_to_normalize])
X_train_to_predict[attributes_to_normalize] = RobustScaler().fit_transform(X_train_to_predict[attributes_to_normalize])

gettingDistributionOfDatas()

#Getting class_weight distribution
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_labels), Y_labels)
class_weight_dict = {1: class_weight[0], 0: class_weight[1]}
print(class_weight_dict)

# for simplicity lets transform ours dataframes in arrays
X_train_test = X_train_robust.values
X_to_predict = X_train_to_predict.values
Y_labels = Y_labels.values

#Percentage train
percentage_train = 0.8
size_train = percentage_train * len(Y_labels)
train_data_X = X_train_test[:int(size_train)]
train_data_Y = Y_labels[:int(size_train)]

#Percentage test
test_data_X = X_train_test[int(size_train):]
test_data_Y = Y_labels[int(size_train):]

#Counting quantity of predictions as won
qt_candidates = len(X_to_predict)
candidate = [0] * qt_candidates
candidate0 = [0] * qt_candidates
results = {}

def predict_results(model, result):
    results[result] = model
    model.fit(train_data_X, train_data_Y)
    resultOscar = model.predict(X_to_predict)
    print('Oscar 2019: ' + str(resultOscar))
    print('')
    for i in range(qt_candidates):
            if(resultOscar[i] == 1.0):
                candidate0[i] = candidate0[i] + 1
    if(result > 0.79):
        for i in range(qt_candidates):
            if(resultOscar[i] == 1.0):
                candidate[i] = candidate[i] + 1

#Predict Adaboost Grided
model = AdaBoostClassifier(n_estimators= 45, learning_rate= 0.01)
result = fit_and_predict("AdaBoostClassifier Grided", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Adaboost
model = AdaBoostClassifier()
result = fit_and_predict("AdaBoostClassifier", model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict SVC Grided
model = svm.SVC(C= 0.03, max_iter= -1, decision_function_shape= 'ovo', tol= 0.001, class_weight= None, gamma='scale')
result = fit_and_predict("SVC Grided", model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict SVC Grided with class_weight balanced
model = svm.SVC(C= 0.03, max_iter= -1, decision_function_shape= 'ovo', tol= 0.001, class_weight= 'balanced', gamma='auto')
result = fit_and_predict("SVC Grided with class_weight balanced", model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict SVC Grided with class_weight count
model = svm.SVC(C= 0.03, max_iter= -1, decision_function_shape= 'ovo', tol= 0.001, class_weight= class_weight_dict, gamma='auto')
result = fit_and_predict("SVC Grided with class_weight count", model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict SVC
model = svm.SVC(gamma='auto')
result = fit_and_predict("SVC Grided", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict KNN Grided
model = neighbors.KNeighborsClassifier(n_neighbors = 4, metric = 'euclidean', weights = 'uniform', algorithm = 'auto')
result = fit_and_predict("KNN Grided", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict KNN
model = neighbors.KNeighborsClassifier()
result = fit_and_predict("KNN", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict MLP
model = neural_network.MLPClassifier()
result = fit_and_predict("MLP", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Logistic Regression and class_weigh balanced
model = LogisticRegression(C= 0.01, fit_intercept= True, tol= 1e-05, max_iter= 100, class_weight= 'balanced', solver='lbfgs')
result = fit_and_predict("LogisticRegression Grided and class_weigh balanced", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Logistic Regression and class_weigh count
model = LogisticRegression(C= 0.01, fit_intercept= True, tol= 1e-05, max_iter= 100, class_weight= class_weight_dict, solver='lbfgs')
result = fit_and_predict("LogisticRegression Grided and class_weigh balanced", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Logistic Regression
model = LogisticRegression(solver='lbfgs')
result = fit_and_predict("LogisticRegression", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Bagging Grided
model = BaggingClassifier(n_estimators= 42, max_samples= 0.087)
result = fit_and_predict("BaggingClassifier Grided", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Bagging
model = BaggingClassifier()
result = fit_and_predict("BaggingClassifier", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Gradient Boosting
model = GradientBoostingClassifier()
result = fit_and_predict("GradientBoostingClassifier", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Random Forest Grided
model = RandomForestClassifier(n_estimators= 10, criterion= 'gini', max_depth= 3)
result = fit_and_predict("RandomForestClassifier Grided", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Random Forest Grided and class_weight count
model = RandomForestClassifier(n_estimators= 10000, criterion= 'gini', max_depth= 3, class_weight= class_weight_dict)
result = fit_and_predict("RandomForestClassifier Grided and class_weight count", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Random Forest Grided and class_weight balanced
model = RandomForestClassifier(n_estimators= 10000, criterion= 'gini', max_depth= 3, class_weight= 'balanced')
result = fit_and_predict("RandomForestClassifier Grided and class_weight balanced", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Random Forest
model = RandomForestClassifier(n_estimators=100)
result = fit_and_predict("RandomForestClassifier", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Extra Tress Grided
model = ExtraTreesClassifier(criterion = 'gini', max_depth = 3, n_estimators = 150)
result = fit_and_predict("ExtraTreesClassifier Grided", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Extra Tress Grided and class_weight balanced
model = ExtraTreesClassifier(criterion = 'gini', max_depth = 3, n_estimators = 150, class_weight= 'balanced')
result = fit_and_predict("ExtraTreesClassifier Grided and class_weight balanced", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict Extra Tress
model = ExtraTreesClassifier(n_estimators = 100)
result = fit_and_predict("ExtraTreesClassifier", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict DecisionTree Grided
model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
result = fit_and_predict("DecisionTree Grided", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict DecisionTree Grided and class_weight count
model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, class_weight= class_weight_dict)
result = fit_and_predict("DecisionTree Grided and class_weight count", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict DecisionTree Grided and class_weight balanced
model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, class_weight = 'balanced')
result = fit_and_predict("DecisionTree Grided and class_weight balanced", model, train_data_X, train_data_Y)
predict_results(model, result)

#Predict DecisionTree
model = tree.DecisionTreeClassifier()
result = fit_and_predict("DecisionTree", model, train_data_X, train_data_Y)
predict_results(model, result)

#The effectiveness of the algorithm that kicks everything 0 or 1 or a single value
base_hit = max(Counter(test_data_Y).values()) #Devolve a quantidade do maior elemento
base_one = list(Y_labels).count(1)
base_zero = list(Y_labels).count(0)
hit_rate_base = 100.0 * base_hit / len(test_data_Y)
print("Hit rate based on validation data: %f" %hit_rate_base)

maximum = max(results)
winner = results[maximum]
print('\n\n')
print(winner)
print('\n\n')
winner.fit(train_data_X, train_data_Y)
result = winner.predict(test_data_X)

len_to_predict = len(train_data_Y)
hit_rate = metrics.accuracy_score(test_data_Y, result)

print("Better algorithm hit rate in the real world " + "was: " 
      + str(hit_rate) + "% " + "from " + str(len_to_predict) + " elements\n\n")

name_and_films = df_to_predict[['film', 'name']]
print(name_and_films)
print('\n')
winner.fit(train_data_X, train_data_Y)
winner_result = winner.predict(X_to_predict)
print('\nBest model predict:')
print(winner_result)
print(name_and_films.iloc[[winner_result.tolist().index(max(winner_result))]])

print('\nWithout accuracy validation:')
print(candidate0)
print(name_and_films.iloc[[candidate0.index(max(candidate0))]])
print("\n")
print('\nOnly if accuracy > 89%:')
print(candidate)
print(name_and_films.iloc[[candidate.index(max(candidate))]])