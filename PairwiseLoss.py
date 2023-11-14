import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class PairwiseLoss(nn.Module):
    
    def forward(self, chosen_rewards, rejected_rewards):
        assert len(chosen_rewards) == len(rejected_rewards)
        batch_size = len(chosen_rewards)
        probs = torch.sigmoid(chosen_rewards - rejected_rewards).log
        
        return -probs.mean()/ batch_size 
