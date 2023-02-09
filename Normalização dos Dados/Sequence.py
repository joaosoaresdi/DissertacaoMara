import csv
import os
import random
import matplotlib.pyplot as plt
import medium as medium
import numpy as np


dir_data = 'Attack_Data_Master/Hydra_FTP_9/'
for f in os.listdir(dir_data):
    if ".csv" not in f:
        continue

    file_name = f
    file_name = f
    index = file_name.index('.')
    file_name = file_name[:index]
    print(file_name)

    with open(dir_data + f, "r") as f0:
        reader = csv.reader(f0)
        data = list(reader)
        data = data[0]

    data1 = []
    for i in range(len(data)):
        t = int(data[i])
        data1.append(t)
    data = data1
    print(len(data))

    #print("data", data)
    tam = len(data)
    # print(tam)

    arrayNews = [(0, 0)] * 10000
    index = 0

    for i in range(len(data) - 10):
        array = [(0, 0)] * 9
        contador = 0
        #print("-.-------------------------------------")
        seq = 2
        aux = (data[i], data[i + 1])
        #print(aux)
        for j in range(1, 10):

            if seq == 2:

                flag = 0
                for p in range(len(arrayNews)):
                    if arrayNews[p][0] == aux:
                        #print("já existe")
                        #print("Array", array)
                        #print("Array das repetições", arrayNews)
                        flag = 1

                    else:
                        continue
                if flag == 0:

                    for w in range(len(data) - 1):

                        tuplo = (data[w], data[w + 1])
                        if aux == tuplo:
                            count = array[contador][1]
                            count = count + 1
                            array[contador] = (aux, count)  # colocar tupla e somar 1
                            arrayNews[index] = (aux, count)
                    #print("array", array)
                    #print("repetidos", arrayNews)
                            #print(array)
                    index +=1
                    contador += 1

            else:
                aux1 = (data[i + j],)
                aux = aux + aux1

                if seq == 3:
                    flag = 0
                    for p in range(len(arrayNews)):
                        if arrayNews[p][0] == aux:
                            flag = 1
                            #print("já existe")
                            #print("Array", array)
                            #print("Array das repetições", arrayNews)
                        else:
                            continue
                    if flag == 0:
                        for w in range(len(data) - seq):
                            tuplo = (data[w], data[w + 1])
                            secondt = (data[w + 2],)
                            tuplo = tuplo + secondt
                            if aux == tuplo:
                                count = array[contador][1]
                                count = count + 1
                                array[contador] = (aux, count)
                                arrayNews[index] = (aux, count)
                        #print("array", array)
                        #print("repetidos", arrayNews)
                        index +=1
                        contador += 1
                                #print("array", array)

                else:

                    flag = 0
                    for p in range(len(arrayNews)):
                        if arrayNews[p][0] == aux:
                            #print("já existe")
                            #print("Array", array)
                            #print("Array das repetições", arrayNews)
                            flag = 1
                        else:
                            continue
                    if flag == 0:
                        #print("aux", aux)
                        for w in range(len(data) - seq):
                            tuplo = ()
                            tuplo = (data[w],)
                            for l in range(w + 1, w + seq):
                                secondt = (data[l],)
                                tuplo = tuplo + secondt
                                #print("tuplo", tuplo)

                                if aux == tuplo:
                                    count = array[contador][1]
                                    count = count + 1
                                    array[contador] = (aux, count)
                                    arrayNews[index] = (aux, count)
                        #print("array", array)
                        #print("repetidos", arrayNews)
                                    #print("array", array)
                        index += 1
                        contador += 1
            seq += 1

        #print(array)
        #print("-----")
        #print(arrayNews)
        #print("-----------")


        plt.figure(figsize=(20, 10))
        plt.title("Frequência das Sequências de Syscalls do ficheiro " + f)
        plt.xlabel("Syscalls")
        plt.ylabel("Frequência")
        array_index = []
        new_array = []
        for p in range(0, len(array)):
            if array[p][1] > 0:
                array_index.append(array[p][0])
                new_array.append(array[p][1])
        #print(array_index)
        #print(new_array)
        pos = np.arange(len(array_index))
        #print(pos)
        plt.xticks(pos, array_index)
        plt.xticks(rotation=10, fontsize=8)
        plt.bar(pos, new_array, 0.2)
        N = random.random()
        for q in range(len(array)):
            plt.text(q, array[q][1], array[q][1], backgroundcolor="orange")
        plt.savefig('Sequence_DataAttack/Hydra_FTP_9/' + file_name + "/" + str(i) + ".png")
        plt.close('all')