import os
import pandas as pd
import csv
import time

dir_data = 'Attack_Data_Master/Web_Shell_6/'

for f in os.listdir(dir_data):
    print(f)
    window = 10
    first_carater = 0
    num_col = 1  # numero de colunas
    num_raw = 1  # numero de linhas
    df = pd.DataFrame({0: [0]})
    
    if ".csv" not in f:
        continue

    for i in range(1, 345):
        df1 = pd.DataFrame({i: [0]})  # criar novo dataframe com a nova coluna
        df2 = df.join(df1)  # concatenar o df atual com o novo df1
        num_col += 1
        df = df2

    with open(dir_data + f, "r") as f0:
        reader = csv.reader(f0)
        data = list(reader)
        data = data[0]
        tam = len(data)

    for y in range(0, tam-10, 1):  #tam do ficheiro

        if num_raw > 1:
            last_row = df.iloc[-1].tolist()  # get last raw in list format
            df.loc[num_raw-1] = last_row
            df.loc[len(df)-1] = 0  # colocar linhas a zeros

        with open(dir_data +f, "r") as file1:
            for line in file1:
                current = line.split(",")

            for x in range(0, window, 1):
                with open(dir_data+f, "r") as file2:
                    for line1 in file2:
                        current1 = line1.split(",")
                        if int(current1[first_carater+x]) in df.columns:
                            df.loc[num_raw-1, int(current1[first_carater + x])] += 1
                        else:
                            print("ALERTA")
            first_carater += 1
            num_raw += 1


    df.to_csv('Attack_Data_Master_PreProc_W=10/Web_Shell_6/' + f)



