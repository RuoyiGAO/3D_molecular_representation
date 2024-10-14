# Import necessary libraries
from rdkit import Chem
from rdkit.Chem import Descriptors  # For calculating molecular properties like weight
from rdkit.Chem import AllChem
from tqdm import tqdm
import json
from itertools import islice
# Function to get atom types and their 3D coordinates from a molecule
def get_atoms_and_coordinates(mol):
    """
    Extracts atom information (atom type and 3D coordinates) from a given molecule.
    
    Parameters:
    mol (Mol): RDKit molecule object.
    
    Returns:
    list: A list of dictionaries, each containing atom type, atom index, and 3D coordinates.
    """
    atoms_and_coords = []  # List to store atoms and their 3D coordinates
    
    # Check if the molecule has any 3D conformers (needed for coordinates)
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()  # Get the first conformer (set of 3D coordinates)
        
        # Iterate over all atoms in the molecule
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()  # Get the atom symbol (e.g., C, O, N)
            atom_idx = atom.GetIdx()  # Get the atom index
            
            # Get the 3D coordinates from the conformer
            pos = conf.GetAtomPosition(atom_idx)
            coords = (pos.x, pos.y, pos.z)  # Store coordinates as a tuple (x, y, z)
            
            # Append atom symbol, coordinates, and atom index to the list
            atoms_and_coords.append({
                "Atom": atom_symbol,
                "Coordinates": coords,
                "Atom_idx": atom_idx
            })

    
    return atoms_and_coords

# Function to read an SDF file and extract molecule properties
def read_sdf_file(sdf_file):
    """
    Reads an SDF file and extracts molecular information, such as name, molecular weight,
    and SMILES representation.
    
    Parameters:
    sdf_file (str): Path to the SDF file.
    
    Returns:
    list: A list of dictionaries containing molecule information.
    """
    # Load molecules from the SDF file
    supplier = Chem.SDMolSupplier(sdf_file)
    
    # Initialize an empty list to store molecule data
    mol_data = []
    supplier = islice(supplier, 10000)

    # Iterate through molecules in the file
    for mol in tqdm(supplier):
        if mol is None:
            continue  # Skip invalid molecules

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol) 

        # Get atom types and their 3D coordinates from the molecule
        atoms_and_coords = get_atoms_and_coordinates(mol)

        # Set atom index labels for SMILES representation for visualization purposes
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())  # Set the atom index to show in the SMILES string

        # Generate the SMILES string for the molecule with atom index labels
        smiles_with_indices = Chem.MolToSmiles(mol)

        # Output: SMILES string with atom index labels
        #print(smiles_with_indices)

        mol_info = {
            "smiles": smiles_with_indices,
            "coords": atoms_and_coords
        }
        
        # Extract properties of the molecule
        
        # Append extracted data (name, weight, SMILES, and molecule object) to list
        mol_data.append(mol_info)  # Store the molecule object for further processing)

    
    with open("qm9_test_noH.json", "w") as json_file:
        json.dump(mol_data, json_file, indent=4, ensure_ascii=False)
    
    return mol_data



# Specify the path to your SDF file
sdf_file = './gdb9.sdf'
# Read the SDF file and extract molecular data
molecules = read_sdf_file(sdf_file)
# Retrieve the molecule object from the extracted data (for the second molecule here)
mol = molecules[1]


