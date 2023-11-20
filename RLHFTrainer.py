# RLHF Trainer

class RLHFTrainer:
    def __init__(self, model, ref_model, config=config):
        self.model = model
        self.ref_model = ref_model
        self.epsilon = config.epsilon
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        
    def compute_advantage_and_return(self, rewards, values):
        rewards = rearrange(rewards, 'b -> 1 b')
        values = rearrange(values, 'b -> 1 b')
        
        lastgaelam = 0
        advantages_reversed = []
        gen_len = len(rewards)
        
        gamma =1
        lam =0.95
        
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t+1] if t < gen_len -1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:,t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0,1)
        returns = rearrange(returns, '1 b -> b')
        
        return advantages, returns 
    
    
    def compute_loss(self, query_ids, query_attention_mask,
                     response_ids, response_attention_mask,
                     rewards
    ):
        # calculate PPO's loss
        
        logprobs, values, entropies, ref_logprobs = self.forward(
            query_ids, query_ids,
            query_attention_mask = query_attention_mask,
            response_ids = response_ids,
            response_attention_mask = response_attention_mask
        )
        
        ratio = (logprobs - ref_logprobs).exp()
        clipped_ratio = torch.clamp(ratio, min=1-self.epsilon, max =1+self.epsilon)
        
        advantages, returns = self.compute_advantages_and_return(rewards, values)
        value_loss = (values - returns).pow(2).mean()
        
        pg_loss_1 = ratio * advantages
        pg_loss_2 = ratio * clipped_ratio
        pg_loss = torch.min(pg_loss_1, pg_loss_2).mean()
        loss = pg_loss - self.ent_coef * entropies.mean() +self.vf_coef * value_loss
        return loss
    
    def forward(self, query_ids, query_attention_mask, response_ids,
               response_attention_mask):
        input_ids = torch.cat([query_ids, response_ids],dim=1)
        attention_mask = torch.cat([query_attention_mask,reponse_attetntion_mask],dim=1)
        _, logprobs, entropy, value = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        _, ref_logprob,_,_ = self.ref_model(
            input_ids = input_ids,
            attention_mask = attetntion_mask
        )
        return logprobs, entropy, value, ref_logprob
