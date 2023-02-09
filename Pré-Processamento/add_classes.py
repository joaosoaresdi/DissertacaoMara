import os
import pandas as pd


dir_data = 'FRQ_ADFA-LD/W=10/Attack_Data_Master_PreProc_W=10/Web_Shell_10/'


for f in os.listdir(dir_data):

    if ".csv" not in f:
        continue
    
    print(f)
    df = pd.read_csv(dir_data + f)
    df = df.iloc[:, 1:]
    df.insert(345, "Class", 1)
    df.drop(df.tail(1).index,inplace=True)
    print(df)
    df.to_csv('FRQ_ADFA-LD/W=10/Attack_Data_Master_labels_W=10/Web_Shell_10/' + f)
