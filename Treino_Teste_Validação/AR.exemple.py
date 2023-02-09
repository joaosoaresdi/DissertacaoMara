import os
import pickle
import pandas as pd
from sklearn import metrics
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Conv2D, Embedding
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten,Dropout,Conv2D
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix

dir_data = 'add_classes/SEQ_ADFA-LD/W=10/Seq_TreinoMisto_labels_w=10/'
dir_data_teste2 = 'add_classes/SEQ_ADFA-LD/W=10/Teste 2/'
dir_data_teste3 = 'add_classes/SEQ_ADFA-LD/W=10/Teste 3/'


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
print(df)
print(df.max(axis=0)) #valor maximo

df_teste2 = FilesConcat(dir_data_teste2)
print(df_teste2)
print(df_teste2.max(axis=0)) #valor maximo


df_teste3 = FilesConcat(dir_data_teste3)
print(df_teste3)
print(df_teste3.max(axis=0)) #valor maximo


#----------DADOS DE TREINO------------------------------------------------------------------

target = df["Class"]
features = df.drop(["Class"], axis=1)

target = target.values
features = features.values

print("classe", target)
print("sequencias:")
print(features)

X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=.3)

print("X_test.shape")
print(X_test.shape)

print("X_train.shape")
print(X_train.shape)

print("y_train.shape")
print(y_train.shape)

print("y_test.shape")
print(y_test.shape)

#-----------------------------------------------------------------------------------------------
#-----------------------------------CRIAÇÃO DAS CAMADAS DO MODELO------------------------------------------
embedding_dim = 32
max_length=10  #tamanho da sequencia

# create the model
model = Sequential()
model.add(Embedding(341, embedding_dim, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
#-----------------------------------------------------------------------------------------------


filename = 'LSTM_model_W=10.sav'

def trainingModel():
    # get the start time
    st = time.time()

    model.fit(X_train, y_train, epochs=30, batch_size=1)

    # save the model to disk
    pickle.dump(model, open('models/' + filename, 'wb'))

    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Training runtime:', elapsed_time, 'seconds')

    return model

# Treinar modelo novamente  (comentar caso não seja necessário novo treino)
model = trainingModel()

# load the model from disk
#model = pickle.load(open('models/' + filename, 'rb'))

#---------------1ºTESTE COM DADOS DO TREINO------------------------
# get the start time
st = time.time()

pred = model.predict(X_test)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Test runtime:', elapsed_time, 'seconds')

prediction=[None]*len(pred)

for i in range(len(pred)):
    if pred[i] >= 0.5:
        prediction[i] = 1
    if pred[i] < 0.5:
        prediction[i] = 0

print("Confusion Matrix:\n", confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

#Curva de ROC------------------------------------
fpr, tpr, _ = metrics.roc_curve(y_test,  prediction)
auc = metrics.roc_auc_score(y_test, prediction)
plt.title('1ºTeste')
plt.plot(fpr,tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#----------------------------DADOS 2ºTESTE-------------------------------------------------------------------

target = df_teste2["Class"]
features = df_teste2.drop(["Class"], axis=1)

target = target.values
features = features.values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

st = time.time()

pred = model.predict(X_test)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Test runtime Teste 2:', elapsed_time, 'seconds')

prediction = [None] * len(pred)

for i in range(len(pred)):
    if pred[i] >= 0.5:
        prediction[i] = 1
    if pred[i] < 0.5:
        prediction[i] = 0

print("Confusion Matrix:\n", confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

# Curva de ROC------------------------------------
fpr, tpr, _ = metrics.roc_curve(y_test, prediction)
auc = metrics.roc_auc_score(y_test, prediction)
plt.title('2ºTeste')
plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#----------------------------DADOS 3ºTESTE-------------------------------------------------------------------


target = df_teste3["Class"]
features = df_teste3.drop(["Class"], axis=1)

target = target.values
features = features.values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

st = time.time()

pred = model.predict(X_test)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Test runtime Teste 3:', elapsed_time, 'seconds')

prediction = [None] * len(pred)

for i in range(len(pred)):
    if pred[i] >= 0.5:
        prediction[i] = 1
    if pred[i] < 0.5:
        prediction[i] = 0

print("Confusion Matrix:\n", confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

# Curva de ROC------------------------------------
fpr, tpr, _ = metrics.roc_curve(y_test, prediction)
auc = metrics.roc_auc_score(y_test, prediction)
plt.title('3ºTeste')
plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#--------------------------------4ºteste--------------------------

target = df["Class"]
features = df.drop(["Class"], axis=1)

target = target.values
features = features.values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.35, random_state=30)

st = time.time()

pred = model.predict(X_test)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Test runtime Teste 4:', elapsed_time, 'seconds')

prediction = [None] * len(pred)

for i in range(len(pred)):
    if pred[i] >= 0.5:
        prediction[i] = 1
    if pred[i] < 0.5:
        prediction[i] = 0

print("Confusion Matrix:\n", confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

# Curva de ROC------------------------------------
fpr, tpr, _ = metrics.roc_curve(y_test, prediction)
auc = metrics.roc_auc_score(y_test, prediction)
plt.title('4ºTeste')
plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
