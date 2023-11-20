import utils
import Agent
import RLHFTrainer
from copy import deepcopy

config = utils.RLHFConfig()
model_base = AutoModelForCausalLM.from_pretrained("gpt2")
model = Agent(model_base)
ref_model = create_reference_model(model)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
max_new_tokens = 20
generation_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": max_new_tokens
}

N_EPOCH = 1
# RLHF Trainer
trainer = RLHFTrainer(model, ref_model, config)
optimizer = optim.SGD(model.parameters(), lr = 1e-3)

for epoch in range(N_EPOCH):
    for batch in train_dataloader:
        inputs= tokenizer(batch['text'], padding=True, truncation=True,return_tensors='pt')
        response_ids = model.generate(
            inputs['input_ids'], attention_mask = inputs['attention_mask'],
            **generation_kwargs
        )
        
        # extract generated text
        response_ids = response_ids[:, -max_new_tokens:]
        response_attention_mask = torch.ones_like(response_ids)
        
        with torch.no_grad():
            text_input_ids = torch.stack([torch.concat([q,r])for q,r in zip(inputs['input_ids'], response_ids)],dim=0)
            rewards = reward_model(text_inputs_ids)
            
        # calculate PPO loss
        loss = trainer.compute_loss(
            query_ids = inputs['input_ids'],
            query_attetnion_mask = inputs['attention_mask'],
            response_ids = response_ids,
            response_attention_mask = response_attention_mask,
            rewards = rewards
            
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'loss={loss}')
        
