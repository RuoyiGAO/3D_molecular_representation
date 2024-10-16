import numpy as np
import torch
import pickle
import math
import warnings
from tqdm import tqdm
from transformers import (PreTrainedTokenizerFast, #BertForMaskedLM, 
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from modeling_bert_rope import BertForMaskedLM
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import wandb


# Ignore warnings
warnings.filterwarnings("ignore")

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            'positions': torch.tensor(self.positions[idx], dtype=torch.long)
        }

class DataCollatorForMLM:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, features):
        input_ids = [f['input_ids'] for f in features]
        positions = [f['positions'] for f in features]

        input_ids = torch.stack(input_ids)
        positions = torch.stack(positions)

        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < self.mlm_probability) * (input_ids != self.tokenizer.pad_token_id)
        input_ids[mask_arr] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids.squeeze(),
            'positions': positions,
            'labels': labels
        }
    
def evaluate_model(model, dataloader, device):
    model.eval()
    losses = []
    for step, batch in enumerate(tqdm(dataloader, position=1, leave=False, desc="Step")):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            positions = batch['positions'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, position_3d=positions, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

    return sum(losses)


def main(
        train_corpus_path, 
        train_position_path,
        val_corpus_path, 
        val_position_path,
        max_length, 
        masked_percentage, 
        per_device_train_batch_size,
        per_device_eval_batch_size,
        numepoch,
        output_dir = None, 
        tokenizer_path = "thaonguyen217/farm_molecular_representation", 
        pretrained_model = "thaonguyen217/farm_molecular_representation", 
        logging_steps =100
        ):
    """
    Main function to train a BERT model for masked language modeling.

    Args:
        train_corpus_path (str): Path to the training corpus.
        val_corpus_path (str): Path to the validation corpus.
        tokenizer_path (str): Path to the pretrained tokenizer.
        pretrained_model (str): Path to the pretrained BERT model.
        output_dir (str): Directory to save model outputs.
        max_length (int): Maximum sequence length for tokenization.
        masked_percentage (float): Percentage of tokens to mask during training.
        numepoch (int): Number of training epoch.
        per_device_train_batch_size (int): Batch size for train.
        per_device_eval_batch_size (int): Batch size for validation.
    """

    # Load tokenizer and Data collator
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    data_collator = DataCollatorForMLM(tokenizer, mlm_probability = masked_percentage)
    
    # Load corpus
    with open(train_corpus_path, 'rb') as f:
        train_corpus = pickle.load(f) 
    with open(val_corpus_path, 'rb') as f:
        val_corpus = pickle.load(f)  
    max_length = 200 
    input_ids_train = []
    for text in train_corpus:
        #encoded = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=max_length)  
        #encoded = tokenizer.encode(text, padding='max_length', truncation=True, return_tensors='pt', max_length=max_length)
        input_ids = tokenizer.encode(text, padding='max_length',max_length=max_length, return_tensors='pt',truncation=True)
        input_ids_train.append(input_ids)
    
    input_ids_val = []
    for text in val_corpus:
        input_ids = tokenizer.encode(text, padding='max_length',max_length=max_length, return_tensors='pt',truncation=True)
        input_ids_val.append(input_ids)


    # Load positions and do padding for posiitons the shape for each molecule will be [max_length, 3]
    with open(train_position_path, 'rb') as f:
        train_positon = pickle.load(f) 
    with open(val_position_path, 'rb') as f:
        val_positon = pickle.load(f) 

   
    train_positon[0] = train_positon[0] + [[100,100,100]]*(max_length-len(train_positon[0]))
    tensor_data_train = [torch.tensor(sublist) for sublist in train_positon]
    tensor_data_train = [(item *100+1200).round().int() for item in tensor_data_train]
    

    train_positon = torch.nn.utils.rnn.pad_sequence(tensor_data_train, batch_first=True, padding_value=11200)
    train_positon = torch.where(train_positon == 11200, 0, train_positon)
    # print(train_positon.shape)
    # torch.save(train_positon, 'train_positon.pt')


    val_positon[0] = val_positon[0] + [[100,100,100]]*(max_length-len(val_positon[0]))
    tensor_data_val = [torch.tensor(sublist) for sublist in val_positon]
    tensor_data_val = [(item *100+1200).round().int() for item in tensor_data_val]
    

    val_positon = torch.nn.utils.rnn.pad_sequence(tensor_data_val, batch_first=True, padding_value=11200)
    val_positon = torch.where(val_positon == 11200, 0, val_positon)
    # torch.save(val_positon, 'val_positon.pt')
    

    # Create datasets for train and evaluation
    dataset_train = CustomDataset(input_ids_train, train_positon)
    # print(dataset_train[0])
    # print(dataset_train[0]["input_ids"].shape)
    # print(dataset_train[0]["positions"].shape)
    # print(dataset_train[1])
    # input()
    dataset_val = CustomDataset(input_ids_val, val_positon)
  
    # Create dataloader for train and evaluation
    dataloader_train = DataLoader(dataset_train, batch_size=per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
    dataloader_val = DataLoader(dataset_val, batch_size=per_device_eval_batch_size, shuffle=True, collate_fn=data_collator)

    # Load the pretrained BERT model
    model = BertForMaskedLM.from_pretrained(pretrained_model).to(device)
    model.resize_token_embeddings(len(tokenizer))
    
    # trian model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    run = wandb.init(project="Bert with RoPE training test",
                    config={"batch_size": per_device_train_batch_size,
                            "learning_rate": 1e-5,
                            "epochs": numepoch})
    
    
    
    print("Training Begin")
    for epoch in range(numepoch):
        losses = []
        # for batch in dataloader_train:
        for step, batch in enumerate(tqdm(dataloader_train, position=1, leave=False, desc="Step")):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            # print("input_ids.shape")
            # print(input_ids.shape)
            positions = batch['positions'].to(device)
            #print(positions)
            #input()
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, position_3d=positions, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss})

            losses.append(loss.item())

        print(f"Epoch: {epoch+1}, Loss: {sum(losses)}")
        # evaluation per epoch
        print(f"Evaluating for {epoch+1}")
        eval_loss_sum = evaluate_model(model, dataloader_val, device)
        wandb.log({"loss_eval": eval_loss_sum})
        print(f"Evaluation loss for {epoch+1}: {eval_loss_sum}")

    print("Training complete!")
    
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train BERT for Masked Language Modeling')
    parser.add_argument('--train_corpus_path', type=str, required=True, help='Path to the training corpus')
    parser.add_argument('--train_position_path', type=str, required=True, help='Path to the positions for training corpus')
    parser.add_argument('--val_corpus_path', type=str, required=True, help='Path to the validation corpus')
    parser.add_argument('--val_position_path', type=str, required=True, help='Path to the positions for validation corpus')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum length of input sequences')
    parser.add_argument('--masked_percentage', type=float, default=0.35, help='Percentage of tokens to mask')
    parser.add_argument('--per_device_train_batch_size', type=int, default=128, help='Batch Size for Train')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128, help='Batch Size for Validation')
    parser.add_argument('--numepoch', type=int, default=20, help='Number of Training Epoch')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save model outputs')
    parser.add_argument('--tokenizer_path', type=str, default='thaonguyen217/farm_molecular_representation', help='Path to the tokenizer')
    parser.add_argument('--pretrained_model', type=str, default='thaonguyen217/farm_molecular_representation', help='Path to the pretrained model')
    parser.add_argument('--logging_steps', type=int, default=10, help='logging_steps')
    
    args = parser.parse_args()
    
    # Call the main function with command-line arguments
    main(
         args.train_corpus_path, args.train_position_path, 
         args.train_corpus_path, args.val_position_path, 
         args.max_length, args.masked_percentage,
         args.per_device_train_batch_size,
         args.per_device_eval_batch_size,
         args.numepoch, args.output_dir,
         args.tokenizer_path, args.pretrained_model,
         args.logging_steps)
