import os
import pandas as pd
import csv
import time

dir_data = 'Validation_Data_Master/'
st = time.time()
for f in os.listdir(dir_data):
    print(f)
    seq = 7
    num_raw = 0
    df = pd.DataFrame({0: [0]})
    #print(df)

    if ".csv" not in f:
        continue

    for i in range(1, seq):
        df1 = pd.DataFrame({i: [0]})  # criar novo dataframe com a nova coluna
        df2 = df.join(df1)  # concatenar o df atual com o novo df1
        df = df2
    #print(df)

    with open(dir_data + f, "r") as f0:
        reader = csv.reader(f0)
        data = list(reader)
        data = data[0]
        #print(data)
        array = [int(numeric_string) for numeric_string in data]
        data = array
        #print(data)
        tam = len(data)
        #print(tam)

    for y in range(0, tam-7, 1):  #tam do ficheiro
        aux = []
        for x in range(y, y + seq, 1):

            aux.extend([data[x]])  # xtending list elements
        df.loc[num_raw] = aux
        num_raw += 1

        last_row = df.iloc[-1].tolist()  #get last raw in list format
        df.loc[num_raw] = last_row
        df.loc[num_raw] = 0  #colocar linhas a zeros

        df.iloc[:-1] #drop last row
        #print(df)
        df.to_csv('Seq_Validation_Data_Master_W=7/' + f)
et = time.time()
#get the execution time
elapsed_time = et - st
print('Tempo de Pré-processamento:', elapsed_time, 'seconds')






