import pandas as pd
import os
root_path = r"./data/CulturaX"
parquet_lst = os.listdir(root_path)

f = open('data_clm.txt', 'w', encoding='utf-8')

for i in parquet_lst:
    df = pd.read_parquet(os.path.join(root_path,i))
    for index, row in df.iterrows():
        f.write(row["text"])
        f.write(row["\n"])
    
f.close()
