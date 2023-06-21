import pandas as pd
from tqdm import tqdm

master_df = pd.DataFrame()

for i in tqdm(range(400)):
    master_df = pd.concat([master_df, pd.read_csv(f"synthetic_descriptors{i}.csv", low_memory=False, index_col=0).sample(frac=1).reset_index(drop=True).head(125)])

master_df.to_csv("sampled_fully_desc.csv")
