import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, input_ids, positions):
        self.input_ids = input_ids
        self.positions = positions

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        print(f"Index type: {type(idx)}, Index value: {idx}")  # 调试信息
        print(idx)
        input()
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'positions': torch.tensor(self.positions[idx], dtype=torch.float)
        }

def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    positions = [item['positions'] for item in batch]

    input_ids_tensor = torch.stack(input_ids, dim=0)
    positions_tensor = torch.stack(positions, dim=0)

    return {
        'input_ids': input_ids_tensor,
        'positions': positions_tensor
    }

# 假设 input_ids 和 positions 是已定义的列表
input_ids = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    # 添加更多数据
]

positions = [
    [[0.1, 0.2],[0.1, 0.2]],
    [[0.1, 0.2],[0.1, 0.2]],
    [[0.1, 0.2],[0.1, 0.2]],
    [[0.1, 0.2],[0.1, 0.2]]
    # 添加更多数据
]

dataset = CustomDataset(input_ids, positions)
dataloader = DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=False, 
    collate_fn=custom_collate_fn
)

for batch in dataloader:
    print(batch)
