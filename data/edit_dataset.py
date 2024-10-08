import pickle


file_path = "3D_molecular_representation/data/FG-enhanced-SMILES_20M.pkl"
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

