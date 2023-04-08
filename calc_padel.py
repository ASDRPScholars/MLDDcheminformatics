from padelpy import from_smiles
import pandas as pd
from joblib import Parallel, delayed
import numpy as np


def add_to_padel_list(l):
    l = l.strip()
    m = l
    if m is None:
        return
    try:
        d = from_smiles(m, timeout=90)
        d['smiles'] = m
        return d
    except:
        print("bad")
        return


if __name__ == '__main__':
    with open('natural_products.smi', 'r') as f:
        all_lines = f.readlines()
        all_lines = np.array(all_lines)
        chunked = np.array_split(all_lines, 400)

        count = 0
        for all_lines in chunked:
            r = Parallel(n_jobs=7, backend='threading', verbose=51)(delayed(add_to_padel_list)(line) for line in all_lines)
            r = [x for x in r if x is not None]
            df = pd.DataFrame(r)

            df.to_csv(f'synthetic_descriptors{count}.csv')
            count += 1
