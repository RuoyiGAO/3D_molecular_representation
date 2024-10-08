from huggingface_hub import hf_hub_download
import pickle

# Download the .pkl file from your Hugging Face dataset repo
#RG: I add repo_type there, otherwise it cannot be download, there is an error
# requests.exceptions.HTTPError: 404 Client Error 
#RG: add an argument to change the data save path
file_path = hf_hub_download(repo_id="thaonguyen217/FG-enhanced-SMILES_20M", filename="FG-enhanced-SMILES_20M.pkl", 
                            repo_type="dataset")
                            #,local_dir="/scratch/eecs542f24_class_root/eecs542f24_class/gaoruoyi/data") 

# Load the dataset from the .pkl file
#RG: change the file_path
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

print(f'Number of FG-enhanced SMILES in the dataset: {len(dataset)}')
print(f'Some examples: {dataset[:10]}')
