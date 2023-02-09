import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

dir_data = 'add_classes/FRQ/W=2/treino_Misto_Label_W=2/'

for f in os.listdir(dir_data):
    if ".csv" not in f:
        continue
    df = pd.read_csv(dir_data + f)
    df = df.iloc[:, 1:]
    break

# Concatenar ficheiros um 1 dataframe só
i = 0
for f in os.listdir(dir_data):
    if ".csv" not in f:
        continue
    print(f)
    if i == 0:
        i = i + 1
        continue
    else:
        df1 = pd.read_csv(dir_data + f)
        df1 = df1.iloc[:, 1:]
        # print(df1.shape[0])
        df = pd.concat([df, df1])
print(df)

# To calculate the accuracy score of the model
df = df.loc[:, (df != 0).any(axis=0)]  #Apagar colunas com todos os valores = 0

print(df)

target = df["Class"]
features = df.drop(["Class"], axis=1)

target = target.values
features = features.values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)


print("X_train, X_test, Y_train, Y_test")

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


