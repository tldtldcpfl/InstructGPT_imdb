import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    
    def __init__(self, model_name,dropout, device):
        super().__init__()
        
        model = AutoModel.from_pretrained(model_name)
        config = model.config
        n_embed = config.n_embd
        
        self.model = model
        # custom head
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_embd, 1),
            nn.Sigmoid()
            ).to(device)
        
    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ).last_hidden_state
        
        output = self.reward_head(last_hidden_state)
        reward_scalar = output[:, -1, 0]
        
        return reward_scalar
        
        # choose the hidden state of the last token as a reward
        
        
        
class PairwiseLoss(nn.Module):
    
    def forward(self, chosen_rewards, rejected_rewards):
        assert len(chosen_rewards) == len(rejected_rewards)
        batch_size = len(chosen_rewards)
        probs = torch.sigmoid(chosen_rewards - rejected_rewards).log
        
        return -probs.mean()/ batch_size 
