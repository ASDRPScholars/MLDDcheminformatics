import pandas as pd

df = pd.read_csv("sampled_fully_desc.csv", low_memory=False, index_col=0)

print(df["Unnamed: 1877"].isnull().all())
