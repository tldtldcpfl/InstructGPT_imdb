class LightRewardModel(nn.Module):
    
    def __init__(self, model, loss_func, lr):
        super().__init__()
        self.model = model # model = reward_model
        self.loss_func = loss_func
        self.lr = lr
        
    def training_step(self, batch):
        chosen_input_ids, chosen_attention_mask, \
        rejected_input_ids, rejected_attention_mask = batch
        
        chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask )
        rejected_rewards = self.model(rejected_input_ids, rejected_attetnion_mask)
        
        loss = self.loss_func(chosen_rewards, rejected_rewards)
        
        print(f'loss={loss}')
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer