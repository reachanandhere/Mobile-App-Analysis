import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
from sklearn import tree
from sklearn import naive_bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sn

# #Reading the CSV File
playStoreDataFrame = pd.read_csv('googleplaystore.csv')


playStoreDataFrame['Target'] = playStoreDataFrame.apply(lambda x: 1 if x["Type"] == "Paid" else 0, axis=1)
playStoreDataFrame.drop(columns=["Type"],axis=1, inplace=True)

playStoreDataFrame = playStoreDataFrame.dropna(subset=list(playStoreDataFrame.columns))

playStoreDataFrame.rename(columns = {'Content Rating' : 'Content_Rating' }, inplace = True)
playStoreDataFrame.rename(columns={"Size":"size_Mb"}, errors="raise", inplace=True)

playStoreDataFrame['Installs']=playStoreDataFrame['Installs'].apply(lambda x: x.replace('+', ""))
playStoreDataFrame['Installs']=playStoreDataFrame['Installs'].apply(lambda x: x.replace(',', ""))


playStoreDataFrame['Price']=playStoreDataFrame['Price'].apply(lambda x: x.replace('$', ''))
playStoreDataFrame['Price']=pd.to_numeric(playStoreDataFrame['Price'], errors='raise')


playStoreDataFrame['size_Mb']=playStoreDataFrame['size_Mb'].apply(lambda x: x.replace('M', ""))
playStoreDataFrame['size_Mb']=playStoreDataFrame['size_Mb'].apply(lambda x: x.replace('k', ""))

labelObject = preprocessing.LabelEncoder()
playStoreDataFrame['EncodedCategory'] = labelObject.fit_transform(playStoreDataFrame['Category'])
playStoreDataFrame['EncodedRating'] = labelObject.fit_transform(playStoreDataFrame['Content_Rating'])
playStoreDataFrame['EncodedGenres'] = labelObject.fit_transform(playStoreDataFrame['Genres'])
playStoreDataFrame.drop(columns=["Category","Content_Rating", "Genres"],axis=1, inplace=True)

unwanted_columns = ['Last Updated','Current Ver','Android Ver', 'Price','App']
playStoreDataFrame = playStoreDataFrame.drop(columns=unwanted_columns)

playStoreDataFrame = playStoreDataFrame.loc[playStoreDataFrame["size_Mb"]!="Varies with device"]


minorityClassList = playStoreDataFrame[playStoreDataFrame['Target']==1]
majorityClassList = playStoreDataFrame[playStoreDataFrame['Target']==0].sample(n=len(minorityClassList), random_state=1)

UndersampledDataframe = pd.concat([minorityClassList, majorityClassList])
# print(UndersampledDataframe.value_counts())

y = UndersampledDataframe['Target']
X = UndersampledDataframe.drop(columns=['Target'])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.33, random_state=1)


targets = UndersampledDataframe["Target"]
features = UndersampledDataframe.drop("Target", axis=1)


# Decision Tree
# param_grid = {'criterion': ["entropy","gini"], 'max_depth': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
# decisionTreeClassifier= GridSearchCV(tree.DecisionTreeClassifier(), param_grid, refit=True, verbose=1)
# decisionTreeClassifier.fit(Xtrain, Ytrain)
# testPrediction = decisionTreeClassifier.predict(Xtest)
# accuracy = accuracy_score(Ytest, testPrediction)*100
# accuracies = cross_val_score(decisionTreeClassifier, Xtrain, Ytrain, cv=5)
# std = np.std(accuracies)
# mean = np.mean(accuracies)
# print(f"DT k-fold accuracies: {accuracies}")
# print(f"DT mean: {mean*100}%")
# print(f"DT std: {std}")
# decisionTreeProba = decisionTreeClassifier.predict_proba(Xtest)[:,1]
# falsePositiveRate_DT, truePositiveRate_DT, threshold_DT = roc_curve(Ytest, decisionTreeProba)
# DecisionTreeAUC = auc(falsePositiveRate_DT, truePositiveRate_DT)
# print('Accuracy of Decision Tree model is equal ' + str(round(accuracy, 2)) + ' %.')
# cm_df = pd.DataFrame(confusion_matrix(Ytest, testPrediction))
# print(classification_report(testPrediction, Ytest), "\n\n")
# recallScore = recall_score(Ytest, testPrediction, average='weighted') * 100
# print('Recall Score of Decision Tree  is equal '+ str(round(recallScore, 2)) + " %.")
# precisionScore = precision_score(Ytest, testPrediction, average='weighted')* 100
# print('Precision Score of Decision Tree model is equal '+str(round(precisionScore, 2)) + " %.")

# fig, ax = plt.subplots()
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position("top")
# sn.heatmap(cm_df, annot=True, cmap="Purples", fmt="d")
# plt.title("Decision Tree model")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()
# #79




# #Logistic Regression

# param_grid = {'max_iter': [100,200, 300, 400, 500, 600, 700, 800, 900, 1000]}
# logisticRegressionclassifier= GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=1)
# logisticRegressionclassifier.fit(Xtrain, Ytrain)
# testPrediction = logisticRegressionclassifier.predict(Xtest)
# accuracy = accuracy_score(Ytest, testPrediction)*100

# accuracies = cross_val_score(logisticRegressionclassifier, Xtrain, Ytrain, cv=5)


# std = np.std(accuracies)
# mean = np.mean(accuracies)
# print(f"LR k-fold accuracies: {accuracies}")
# print(f"LR mean: {mean*100}%")
# print(f"LR std: {std}")

# logisticRegressionProba = logisticRegressionclassifier.predict_proba(Xtest)[::,1]
# falsePositiveRate_LR, truePositiveRate_LR, threshold_lr = roc_curve(Ytest, logisticRegressionProba)
# LogisticRegressionAUC = auc(falsePositiveRate_LR, truePositiveRate_LR)

# print('Accuracy of Logistic Regression model is equal ' + str(round(accuracy, 2)) + ' %.')
# cm_df = pd.DataFrame(confusion_matrix(Ytest, testPrediction))
# print(classification_report(testPrediction, Ytest), "\n\n")
# recallScore = recall_score(Ytest, testPrediction, average='weighted')
# print('Recall Score of LR is equal ', recallScore*100)
# precisionScore = precision_score(Ytest, testPrediction, average='weighted')
# print('Precision Score of LR is equal ', precisionScore*100)
# fig, ax = plt.subplots()
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position("top")
# sn.heatmap(cm_df, annot=True, cmap="Purples", fmt="d")
# plt.title("Logistic Regression model")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()
# #79


# # K-NN Classifier


# k_list = list(range(3, 17, 2))
# best_k, best_accuracy = 0,0

# for i in k_list:
#     knnclassifier = KNeighborsClassifier(n_neighbors=i, metric="euclidean")    
#     knnclassifier.fit(Xtrain, Ytrain)
#     testPrediction = knnclassifier.predict(Xtest)
#     accuracy = accuracy_score(Ytest, testPrediction)*100
#     if accuracy > best_accuracy:
#         print(accuracy)
#         best_accuracy =accuracy

#         best_k = i


# Training KNN with the best K received from above
knnclassifier = KNeighborsClassifier(n_neighbors=7, metric="euclidean")    
knnclassifier.fit(Xtrain, Ytrain)
testPrediction = knnclassifier.predict(Xtest)
accuracies = cross_val_score(knnclassifier, Xtrain, Ytrain, cv=5)
std = np.std(accuracies)
mean = np.mean(accuracies)
print(f"KNN k-fold accuracies: {accuracies}")
print(f"KNN mean: {mean*100}%")
print(f"KNN std: {std}")

knnclassifierProba = knnclassifier.predict_proba(Xtest)[::,1]
falsePositiveRate_KNN, truePositiveRate_KNN, threshold_knn = roc_curve(Ytest, knnclassifierProba)
kNNAUC = auc(falsePositiveRate_KNN, truePositiveRate_KNN)
accuracy = accuracy_score(Ytest, testPrediction) * 100
print("Accuracy of KNN model is equal " + str(round(accuracy, 2)) + " %.")
cm_df = pd.DataFrame(confusion_matrix(Ytest, testPrediction))
print(classification_report(testPrediction, Ytest), "\n\n")
recallScore = recall_score(Ytest, testPrediction, average='weighted') * 100
print('Recall Score of KNN Model is equal '+ str(round(recallScore, 2)) + " %.")
precisionScore = precision_score(Ytest, testPrediction, average='weighted')
print('Precision Score of KNN model is equal ', precisionScore)

# fig, ax = plt.subplots()
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position("top")
# sn.heatmap(cm_df, annot=True, cmap="Purples", fmt="d")
# plt.title("K Nearest Neighbor model")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()

# # Naive Bayes Classifier

# param_grid = {'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20]}
# naiveBayesClassifier= GridSearchCV(naive_bayes.GaussianNB(), param_grid, refit=True, verbose=1)


# naiveBayesClassifier = naive_bayes.GaussianNB()
# naiveBayesClassifier.fit(Xtrain, Ytrain)

# testPrediction = naiveBayesClassifier.predict(Xtest)
# accuracy = accuracy_score(Ytest, testPrediction)*100

# accuracies = cross_val_score(naiveBayesClassifier, Xtrain, Ytrain, cv=5)
# std = np.std(accuracies)
# mean = np.mean(accuracies)
# print(f"Naive Bayes k-fold accuracies: {accuracies}")
# print(f"Naive Bayes mean: {mean*100}%")
# print(f"Naive Bayes std: {std}")

# naiveBayesClassifierProba = naiveBayesClassifier.predict_proba(Xtest)[::,1]
# falsePositiveRate_NB, truePositiveRate_NB, threshold_NB = roc_curve(Ytest, naiveBayesClassifierProba)
# naiveBayesAUC = auc(falsePositiveRate_NB, truePositiveRate_NB)


# print('Accuracy of Naive Bayes Classifier is equal ' + str(round(accuracy, 2)) + ' %.')
# cm_df = pd.DataFrame(confusion_matrix(Ytest, testPrediction))
# print(classification_report(testPrediction, Ytest), "\n\n")
# recallScore = recall_score(Ytest, testPrediction, average='weighted')
# print('Recall Score of Naive Bayes Classifier Model is equal ', recallScore*100)
# precisionScore = precision_score(Ytest, testPrediction, average='weighted')
# print('Precision Score of Naive Bayes Classifier Model is equal ', precisionScore*100)
# fig, ax = plt.subplots()
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position("top")
# sn.heatmap(cm_df, annot=True, cmap="Purples", fmt="d")
# plt.title("Naive Bayes model")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()

# # # LDA Classifier

# param_grid = {'solver': ['svd','lsqr','eigen']}
# LinearDiscriminantAnalysisModel= GridSearchCV(LinearDiscriminantAnalysis(), param_grid, refit=True, verbose=1)
# LinearDiscriminantAnalysisModel.fit(Xtrain, Ytrain)
# testPrediction = LinearDiscriminantAnalysisModel.predict(Xtest)



# accuracies = cross_val_score(LinearDiscriminantAnalysisModel, Xtrain, Ytrain, cv=5)
# std = np.std(accuracies)
# mean = np.mean(accuracies)
# print(f"LDA k-fold accuracies: {accuracies}")
# print(f"LDA mean: {mean}")
# print(f"LDA std: {std}")

# LinearDiscriminantAnalysisModelProba = LinearDiscriminantAnalysisModel.predict_proba(Xtest)[::,1]
# falsePositiveRate_LDA, truePositiveRate_LDA, threshold_LDA = roc_curve(Ytest, LinearDiscriminantAnalysisModelProba)
# ldaAUC = auc(falsePositiveRate_LDA, truePositiveRate_LDA)
# accuracy = accuracy_score(Ytest, testPrediction)*100
# print('Accuracy of Linear Discriminant Analysis model is equal ' + str(round(accuracy, 2)) + ' %.')
# cm_df = pd.DataFrame(confusion_matrix(Ytest, testPrediction))
# print(classification_report(testPrediction, Ytest), "\n\n")
# recallScore = recall_score(Ytest, testPrediction, average='weighted') * 100
# print('Recall Score of Linear Discriminant Analysis  is equal '+ str(round(recallScore, 2)) + " %.")
# precisionScore = precision_score(Ytest, testPrediction, average='weighted')
# print('Precision Score of Linear Discriminant Analysis model is equal ', precisionScore)
# fig, ax = plt.subplots()
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position("top")
# sn.heatmap(cm_df, annot=True, cmap="Purples", fmt="d")
# plt.title("Linear Discriminant Analysis")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.plot(falsePositiveRate_DT, truePositiveRate_DT, label=f'Decision Tree (AUC = {DecisionTreeAUC:.2f})', color='green')
# plt.plot(falsePositiveRate_LR, truePositiveRate_LR, label=f'Logistic Regression (AUC = {LogisticRegressionAUC:.2f})', color='blue')
# plt.plot(falsePositiveRate_KNN, truePositiveRate_KNN, label=f'K-NN (AUC = {kNNAUC:.2f})', color='pink')
# plt.plot(falsePositiveRate_NB, truePositiveRate_NB, label=f'Naive Bayes (AUC = {naiveBayesAUC:.2f})', color='red')
# plt.plot(falsePositiveRate_LDA, truePositiveRate_LDA, label=f'Linear Discriminant Model (AUC = {ldaAUC:.2f})', color='brown')
# plt.plot([0, 1], [0, 1], linestyle='--', color='yellow', label='Random Guess (AUC = 0.5)')

# # Customize the plot
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()