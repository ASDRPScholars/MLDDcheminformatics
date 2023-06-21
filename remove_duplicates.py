import os
from io import StringIO
import sys
from rdkit import Chem, rdBase
from tqdm import tqdm

rdBase.LogToPythonStderr()


def remove_sampled_smiles_duplicates():
    print(f"Opening natural_products.smi")
    with open("../natural_products.smi", "r") as coconut_smiles_file:
        coconut_smiles_lines = coconut_smiles_file.readlines()
    coconut_smiles_lines = strip_list(coconut_smiles_lines)  # Removes trailing \n
    coconut_smiles_lines = format_smiles_list(coconut_smiles_lines)  # Conforms Smile Strings so comparisons are valid
    print(f"Finished Opening natural_products.smi\n")

    coconut_dict = {line: 1 for i, line in enumerate(coconut_smiles_lines)}

    print(f"Opening sampled_smiles.smi")
    with open("./sampled_smiles.smi", "r") as sampled_smiles_file:
        sampled_smiles_lines = sampled_smiles_file.readlines()
    sampled_smiles_lines = strip_list(sampled_smiles_lines)  # Removes trailing \n
    sampled_smiles_lines = format_smiles_list(sampled_smiles_lines)  # Conforms Smile Strings so comparisons are valid
    print(f"Finished Opening sampled_smiles.smi")

    smiles_duplicates = 0
    new_sampled_smiles_lines = ["smiles"]

    print("Starting filter")

    for sampled_smile_line in sampled_smiles_lines:
        if sampled_smile_line in coconut_dict:
            smiles_duplicates += 1
        else:
            new_sampled_smiles_lines.append(sampled_smile_line)

    with open(f"./sampled_smiles_new.smi", "w") as new_sampled_smiles_file:
        new_sampled_smiles_file.writelines(fatten_list(new_sampled_smiles_lines))

    print("Created Valid sampled_smiles.smi")

    print("\nCompleted!")
    print(f"Total Smiles Duplicates Found: {smiles_duplicates}")


def format_smiles_list(smiles: list):
    formatted_smiles = []

    invalid_molecules = 0
    for i, smile in enumerate(smiles):
        sio = sys.stderr = StringIO()
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            # print(i, molecule, sio.getvalue())
            invalid_molecules += 1
        else:
            formatted_smiles.append(Chem.MolToSmiles(molecule))

    return formatted_smiles


def strip_list(arr: list):
    for i, line in enumerate(arr):
        arr[i] = line.rstrip("\n")
    return arr


def fatten_list(arr: list):
    for i, line in enumerate(arr[:-1]):
        arr[i] += "\n"
    return arr
