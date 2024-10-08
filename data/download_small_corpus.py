from huggingface_hub import hf_hub_download
import pickle

# Download the .pkl file from your Hugging Face dataset repo
#RG: I add repo_type there, otherwise it cannot be download, there is an error
# requests.exceptions.HTTPError: 404 Client Error 
#RG: add an argument to change the data save path
file_path = hf_hub_download(repo_id="thaonguyen217/FG-enhanced-SMILES_20M", filename="FG-enhanced-SMILES_20M.pkl", 
                            repo_type="dataset")
                            #,local_dir="/scratch/eecs542f24_class_root/eecs542f24_class/gaoruoyi/data") 

print("------")
print(file_path)
print("------")
# Load the dataset from the .pkl file
#RG: change the file_path
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

print(f'Number of FG-enhanced SMILES in the dataset: {len(dataset)}')
print(f'Some examples: {dataset[:10]}')

dataset_new = dataset[:10000]
dataset_val = dataset[:100]

#RG: get a small set of data for train
save_path = 'FG-enhanced-SMILES_10K.pkl'
print(f'Number of FG-enhanced SMILES in the dataset_new: {len(dataset_new)}')
with open(save_path, 'wb') as f:
    pickle.dump(dataset_new, f)
txt_new = '\n'.join(dataset_new)
with open(save_path.replace('pkl', 'txt'), 'w') as f:
    f.write(txt_new)

#RG: get a small set of data for val
save_path_val = 'FG-enhanced-SMILES_10K_val.pkl'
print(f'Number of FG-enhanced SMILES in the dataset_new_val: {len(dataset_val)}')
with open(save_path_val, 'wb') as f:
    pickle.dump(dataset_val, f)
txt_val = '\n'.join(dataset_val)
with open(save_path_val.replace('pkl', 'txt'), 'w') as f:
    f.write(txt_val)
