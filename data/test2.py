import json
import pandas as pd

json_file = 'qm9_test_noH.json'
with open(json_file, 'r') as file:
    data = json.load(file)

# Extract smiles and coordinates
rows = []
for entry in data:
    smiles = entry['SMILES']
    position = entry['coords']
    rows.append([smiles, position])

# Create DataFrame and save as CSV
df = pd.DataFrame(rows, columns=['smiles', 'position'])
csv_path = 'smiles_positions_qm9_test.csv'
df.to_csv(csv_path, index=False)


