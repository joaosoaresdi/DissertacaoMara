import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time

dir_data = 'add_classes/FRQ_ADFA-LD/W=10/TreinoMisto_labels_w=10/'
dir_data_teste2 = 'add_classes/FRQ_ADFA-LD/W=10/Teste 2/'
dir_data_teste3 = 'add_classes/FRQ_ADFA-LD/W=10/Teste 3/'


def FilesConcat(directory):
    for f in os.listdir(directory):
        if ".csv" not in f:
            continue
        df = pd.read_csv(directory + f)
        df = df.iloc[:, 1:]
        break

    # Concatenar ficheiros um 1 dataframe só
    i = 0
    for f in os.listdir(directory):
        if ".csv" not in f:
            continue
        #print(f)
        if i == 0:
            i = i + 1
            continue
        else:
            df1 = pd.read_csv(directory + f)
            df1 = df1.iloc[:, 1:]
            # print(df1.shape[0])
            df = pd.concat([df, df1])
    return df


df = FilesConcat(dir_data)
#df = df.loc[:, (df != 0).any(axis=0)]  # Apagar colunas com todos os valores = 0
print(df)

df_teste2 = FilesConcat(dir_data_teste2)
#df_teste1 = df_teste1.loc[:, (df_teste1 != 0).any(axis=0)]  # Apagar colunas com todos os valores = 0
print(df_teste2)

df_teste3 = FilesConcat(dir_data_teste3)
#df_teste1 = df_teste1.loc[:, (df_teste1 != 0).any(axis=0)]  # Apagar colunas com todos os valores = 0
print(df_teste3)



#---------------------------------Dados de treino----------------------------------------------------------------

target = df["Class"]
features = df.drop(["Class"], axis=1)

target = target.values
features = features.values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=10)
print('X_train')
print(X_train)
print('X_test')
print(X_test)
print('Y_train')
print(y_train)
print('Y_test')
print(y_test)

print("X_train, X_test, Y_train, Y_test")

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#-------------------------------------------------------------------------------------------

filename = 'W=10/RF_model_W=10.sav'

def trainingModel():
    # get the start time
    st = time.time()

    clf = RandomForestClassifier(n_estimators=100, bootstrap=False, max_features='sqrt', max_depth = 50)
    clf.fit(X_train, y_train)

    # save the model to disk
    pickle.dump(clf, open('models/' + filename, 'wb'))

    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Training runtime:', elapsed_time, 'seconds')

    return clf

# Treinar modelo novamente  (comentar caso não seja necessário novo treino)
clf = trainingModel()

# load the model from disk
clf = pickle.load(open('models/' + filename, 'rb'))

#---------------1ºteste com os dados do treino------------------------
result = clf.score(X_test, y_test)
print("Score")
print(result)


# get the start time
st = time.time()

y_pred = clf.predict(X_test)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Test runtime:', elapsed_time, 'seconds')

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Curva de ROC------------------------------------
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)

plt.title('1ºTeste')
#create ROC curve
plt.plot(fpr,tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#----------------------------DADOS 2ºTESTE-----------------------

target = df_teste2["Class"]
features = df_teste2.drop(["Class"], axis=1)

target = target.values
features = features.values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=10)

# get the start time
st = time.time()

y_pred = clf.predict(X_test)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Test runtime Teste 2:', elapsed_time, 'seconds')

print("Accuracy Teste 2:", metrics.accuracy_score(y_test, y_pred))

print("Confusion Matrix Teste 2:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Curva de ROC------------------------------------
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)

plt.title('2ºTeste')
plt.plot(fpr,tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#-------------------------------------------------------------------------------

#----------------------------DADOS 3ºTESTE-----------------------



target = df_teste3["Class"]
features = df_teste3.drop(["Class"], axis=1)

target = target.values
features = features.values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=10)

# get the start time
st = time.time()

y_pred = clf.predict(X_test)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Test runtime Teste 3:', elapsed_time, 'seconds')

print("Accuracy Teste 3:", metrics.accuracy_score(y_test, y_pred))

print("Confusion Matrix Teste 3:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Curva de ROC------------------------------------
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)

plt.title('3ºTeste')
#create ROC curve
plt.plot(fpr,tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#-------------------------------------------------------------------------------



#--------------------------------4ºteste--------------------------

target = df["Class"]
features = df.drop(["Class"], axis=1)

target = target.values
features = features.values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.35, random_state=30)

y_pred = clf.predict(X_test)

print("Accuracy 4ºteste:", metrics.accuracy_score(y_test, y_pred))

print("Confusion Matrix 4ºteste:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Curva de ROC------------------------------------
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)

plt.title('4ºTeste')
#create ROC curve
plt.plot(fpr,tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()



''''
thresholds = np.linspace(0, 1, 100)
precision_scores = []
recall_scores = []
for threshold in thresholds:
    adjusted_predictions = [1 if p > threshold else 0 for p in y_pred]
    precision_scores.append(precision_score(y_test, adjusted_predictions))
    recall_scores.append(recall_score(y_test, adjusted_predictions))
plt.plot(thresholds, precision_scores, label="precision")
plt.plot(thresholds, recall_scores, label="recall")
plt.show()'''