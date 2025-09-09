#Imports
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel, get_linear_schedule_with_warmup
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset
from pickle import  load, dump
from encoder import CustomEncoder

#Classes
class video2reportDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data = self.data[idx]
        
        input = torch.stack(([data['icd_icpm']]+data['frames']))           
        
        opbericht = data['Ablauf']    
        opbericht = f"<|startoftext|>{opbericht.strip()}<|endoftext|>"
        # Tokenize target
        target = self.tokenizer.encode_plus(opbericht,
                                              truncation=True,
                                              padding='max_length',
                                              max_length = 1024,
                                              return_tensors='pt')

        return {"inputs": input, 
                "labels": torch.squeeze(target['input_ids'],0), 
                "attn_mask": torch.squeeze(target['attention_mask'],0)}

class video2report(nn.Module):
    def __init__(self, decoder_name):
        super().__init__()
        self.encoder = CustomEncoder(d_model=768, num_layers=12, num_heads=12, d_ff=1024)
        configuration = GPT2Config.from_pretrained(decoder_name, is_decoder = True, add_cross_attention = True, loss_type = 'ForCausalLMLoss')
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_name, config=configuration)
    
    def forward(self, x, inputs, attn_mask):
        x = self.encoder(x)

        out = self.decoder(input_ids = inputs, labels = inputs, attention_mask= attn_mask, encoder_hidden_states = x)
        return out
    
    def generate(self, x, inputs, attn_mask, generation_config):
        x = self.encoder(x)

        output = self.decoder.generate(inputs = inputs,
                                       generation_config =generation_config,
                                       attention_mask=attn_mask,
                                       encoder_hidden_states = x)
        
        return output

#Methods
def read_data():
    output = []
    dir_path = os.path.join(path,'Data', 'analyzed_imgs')
    dir = os.listdir(dir_path)
    for file in dir:
        if "p12" not in file:
            with open(os.path.join(dir_path, file), "rb") as input_file:
                dict = load(input_file)
            output.append(dict)
    return output 


#Parameters
path =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

decoder = os.path.join(path,'Code', 'Text_Generation', 'Models', 'pretrained_gpt2_v2')

tokenizer = GPT2TokenizerFast.from_pretrained(decoder,
                                            bos_token='<|startoftext|>',
                                            eos_token='<|endoftext|>',
                                            unk_token='<|unknown|>',
                                            pad_token='<|pad|>'
                                            )

data = read_data()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epochs = 50
learning_rate = 2e-5
warmup_steps = 1e2
epsilon = 1e-8

#Training

loo = LeaveOneOut()
results = {}
for fold, (train_idx, test_idx) in enumerate(loo.split(data)):
    print(f"Fold {fold + 1}/15")

    # Subsets
    train_val_subset = Subset(data, train_idx)
    test_subset = Subset(data, test_idx)

    train_subset, val_subset = train_test_split(train_val_subset, test_size=0.1, shuffle=True)

    train_dataset = video2reportDataset(train_subset, tokenizer)
    eval_dataset = video2reportDataset(val_subset, tokenizer)
    # Dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(eval_dataset, shuffle=False)


    # Initialize model, loss, optimizer
    model = video2report(decoder_name=decoder).to(device)
    optim = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
    total_steps = len(train_loader) * epochs  

    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    fold_eval_loss = {}    
    for epoch_i in range(0, epochs):
        model.train() 
        for step, batch in tqdm(enumerate(train_loader), total= len(train_loader), desc="Epoch: [{}]".format(epoch_i)): 
            b_input_ids = batch['inputs'].to(device) 
            b_labels    = batch['labels'].to(device) 
            b_attn_mask = batch['attn_mask'].to(device)

            model.zero_grad()
            outputs = model(x = b_input_ids, inputs = b_labels, attn_mask = b_attn_mask)

            loss = outputs[0]
            loss.backward()
            optim.step()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        # Evaluate data for one epoch
        for batch in val_loader:
            b_input_ids = batch['inputs'].to(device) 
            b_labels    = batch['labels'].to(device) 
            b_attn_mask = batch['attn_mask'].to(device)

            with torch.no_grad():
                outputs  = model(x = b_input_ids, inputs = b_labels, attn_mask = b_attn_mask)
                val_loss = outputs[0].item()

        print(f"Epoch {epoch_i + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}")
        fold_eval_loss[epoch_i] = val_loss / len(val_loader)

    # Save results for this fold
    results[fold] = val_loss / len(val_loader)
    torch.save(model.state_dict(),  os.path.join(path,'Code', 'Video_Analysis', 'Models', f'video2report_fold_{fold}.pth'))
    with open (os.path.join(path,'Code', 'Video_Analysis', 'Models', f'test_data_fold_{fold}.pkl'), 'wb') as file:
        dump(test_subset, file)

print(f"Cross-validation results: {results}")
  
tokenizer.save_pretrained(os.path.join(path,'Code', 'Video_Analysis', 'Models', 'tokenizer_v2'))
with open (os.path.join(path,'Code', 'Video_Analysis', 'Models', 'folds_eval_loss.pkl'), 'wb') as file:
    dump(results, file)