import csv
import os
import matplotlib.pyplot as plt
import numpy as np

dir_data = 'Attack_Data_Master/Web_Shell_10/'
for f in os.listdir(dir_data):
    if ".csv" not in f:
        continue

    print(f)

    with open(dir_data + f, "r") as f0:
        reader = csv.reader(f0)
        data = list(reader)
        data = data[0]
        tam = len(data)
        print("tamanho:", tam)

    data1 = []
    for i in range(len(data)):
        t = int(data[i])
        data1.append(t)
    data = data1

    print("data", data)

    tam = max(data)
    print("nr máx:", tam)

    array = [0] * tam

    for i in range(0, len(data)):
        aux = data[i]

        array[aux-1] += 1

    print(array)
    plt.figure(figsize=(20, 10))
    plt.title("Frequência singular das syscalls do ficheiro " + f)
    plt.xlabel("Syscall")
    plt.ylabel("Frequência")

    array_index = []
    new_array = []
    for i in range(0, len(array)):
        if array[i] > 0:
            array_index.append(i+1)
            new_array.append(array[i])
    print(array_index)
    print(new_array)
    pos = np.arange(len(array_index))
    print(pos)
    plt.xticks(pos, array_index)
    plt.bar(pos, new_array, 0.2)
    plt.savefig('Frequence_DataAtack/Web_Shell_10/' + f + ".png")
