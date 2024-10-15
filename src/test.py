import pickle
import torch

from transformers import (PreTrainedTokenizerFast)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, input_ids, positions):
        self.input_ids = input_ids
        self.positions = positions

    def __len__(self):
        assert len(self.input_ids) == len(self.positions)
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'positions': torch.tensor(self.positions[idx], dtype=torch.float)
        }
    
    
class DataCollatorForMLM:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, features):
        # 提取 input_ids 和 positions
        input_ids = [f['input_ids'] for f in features]
        positions = [f['positions'] for f in features]

        # padding for input ids
        input_ids = torch.stack(input_ids)
        #print(input_ids[0].shape)
        
        #print(batch_input_ids.shape)
        
        # padding for positions
        tensor_3d = torch.stack(positions)

        # 进行 MLM 掩码
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < self.mlm_probability) * (input_ids != self.tokenizer.pad_token_id)

        # 将需要掩码的位置替换为 [MASK] token
        input_ids[mask_arr] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'positions': tensor_3d,
            'labels': labels
        }
    
tokenizer = PreTrainedTokenizerFast.from_pretrained("output_1014")
max_length = 512
per_device_train_batch_size=64

with open("/home/gaoruoyi/3D_molecular_representation/src/corpus_position_1014_qm9_test/corpus_qm9_test.pkl", 'rb') as f:
    train_corpus = pickle.load(f) 


input_ids_train = []
for text in train_corpus:
    encoded = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=max_length)  
    input_ids_train.append(encoded['input_ids'][0])
    

with open("/home/gaoruoyi/3D_molecular_representation/src/corpus_position_1014_qm9_test/positions_qm9_test.pkl", 'rb') as f:
    train_positon = pickle.load(f) 

train_positon[0] = train_positon[0] + [[100.00,100.00,100.00]]*(max_length-len(train_positon[0]))
tensor_data_train = [torch.tensor(sublist) for sublist in train_positon]
train_positon = torch.nn.utils.rnn.pad_sequence(tensor_data_train, batch_first=True, padding_value=100.0)

dataset_train = CustomDataset(input_ids_train, train_positon)
data_collator = DataCollatorForMLM(tokenizer)

dataloader_train = DataLoader(dataset_train, batch_size=per_device_train_batch_size, shuffle=True, collate_fn=data_collator)

print(len(input_ids_train[0]))
print(len(train_positon[0]))

for batch in dataloader_train:
    input_ids = batch['input_ids']
    positions = batch['positions']
    labels = batch['labels']

print("end")