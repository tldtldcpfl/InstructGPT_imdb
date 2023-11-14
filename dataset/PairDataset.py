# instruct_goose / dataset.py
from typing import Callable, Tuple, Iterable
from torch.utils.data import Dataset
from tqdm import tqdm

class PairDataset(Dataset):
    
    def __init__(self, dataset, tokenizer, max_length=1024):
        self.chosen = []
        self.rejected = []
        
        for data in tqdm(dataset):
            chosen, rejected = data['chosen'], data['rejected']
            chosen_encoding = tokenizer(
                chosen,
                max_length=max_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            rejected_encoding = tokenizer(
                rejected,
                max_length=max_length, padding='max_length',truncation=True,
                return_tensors='pt'
            )
            self.chosen.append({
                'input_ids': chosen_encoding['input_ids'],
                'attention_mask': chosen_encoding['attention_mask']
            })
            self.rejected.append({
                'input_ids': rejected_encoding['input_ids'],
                'attention_mask': rejected_encoding['attention_mask']
            })
    
    def __len__(self):
        return len(self.chosen)
    
    def __getitem__(self, idx):
        
        return self.chosen[idx]['input_ids'],\
                self.chosen[idx]['attention_mask'],\
                self.rejected[idx]['input_ids'],\
                self.rejected[idx]['attention_mask']
