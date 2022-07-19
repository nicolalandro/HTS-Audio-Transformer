from curses import meta
import os
import pandas as pd

dataset_path = "/home/super/datasets-nas/ESC-50/ESC-50-master"
meta_path = os.path.join(dataset_path, "meta", "esc50.csv")

df = pd.read_csv(meta_path)

classes_df = df[['target', 'category']].drop_duplicates().sort_values(by=['target'])
print(classes_df)

l = []
for k,v in classes_df.iterrows():
    l.append(v['category'])
print(l)