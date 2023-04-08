import pandas as pd

nat_data = pd.read_csv('nat_comb.csv', low_memory=False)
syn_data = pd.read_csv('syn_comb.csv', low_memory=False)


for col in nat_data.columns:
    if "unnamed" in col.lower():
        nat_data.drop(col, inplace=True, axis=1)

for col in syn_data.columns:
    if "unnamed" in col.lower():
        syn_data.drop(col, inplace=True, axis=1)

nat_data['class_label'] = 0
syn_data['class_label'] = 1
all_data = pd.concat([nat_data, syn_data])
all_data = all_data.sample(frac=1).reset_index(drop=True)

nat_data.to_csv("nat_comb1.csv")
print(pd.read_csv("nat_comb1.csv"))
syn_data.to_csv("syn_comb1.csv")
print(pd.read_csv("syn_comb1.csv"))
all_data.to_csv("final_combed_set.csv")
print(pd.read_csv("final_combed_set.csv"))
