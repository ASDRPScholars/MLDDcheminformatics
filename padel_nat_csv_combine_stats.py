import pandas as pd
from tqdm import tqdm

"""
Authors: Spencer Ye

Last Updated: 10/1/2022 @12:28am

"""

df_list = []

num_frames = 400
correlation_max = 1

for i in tqdm(range(num_frames)):
	df_list.append(pd.read_csv("synthetic_descriptors%s.csv" % i, low_memory=False))

df_master = pd.concat(
    df_list,
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    copy=True,
)


df_master_numerical = df_master.select_dtypes(include='number')

""" This section is for dropping any r=1 columns, only found one, so commenting out for now (changed)

columns_to_drop = []
correlation_matrix = df_master_numerical.corr(numeric_only=True)
print(len(correlation_matrix.columns))
print(correlation_matrix)
num_columns = range(len(correlation_matrix.columns) - 1)


for column in tqdm(num_columns):
    print(column)
    for checking_column in range(column):
        temp = correlation_matrix.iloc[column:(column+1), (checking_column+1):(checking_column+2)]

        if temp.values >= correlation_max:
            columns_to_drop.append(temp.columns.values[0])




columns_to_drop_set = set(columns_to_drop)

print(len(columns_to_drop_set))

print(len(df_master_numerical.columns))

df_master_numerical = df_master_numerical.drop(columns=columns_to_drop_set)

print(len(df_master_numerical.columns))

"""

df_master_numerical.to_csv("padel_nat_combined_numerical.csv")

df_stats = df_master_numerical.describe()

df_stats.to_csv("padel_nat_combined_stats(1).csv")
