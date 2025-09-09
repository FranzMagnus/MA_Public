from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, GenerationConfig
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pickle import  load
from encoder import CustomEncoder
from pickle import load


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
                                       generation_config = generation_config,
                                       attention_mask = attn_mask,
                                       encoder_hidden_states = x)
        
        return output
    
def read_data():
    output = []
    dir_path = os.path.join(path,'Data', 'analyzed_imgs')
    dir = os.listdir(dir_path)
    for file in dir:
        with open(os.path.join(dir_path, file), "rb") as input_file:
            dict = load(input_file)
        output.append(dict)
    return output 

def infer(prompt):
    input = f"<|startoftext|>{prompt.strip()}"
    input = tokenizer(input, return_tensors='pt')
    input_ids      = input["input_ids"]
    attention_mask = input["attention_mask"]

    return input_ids, attention_mask



path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

data = read_data()

decoder = os.path.join(path, 'Code', 'Text_Generation', 'Models', 'pretrained_gpt2_v2')

tokenizer = GPT2TokenizerFast.from_pretrained(os.path.join(path, 'Code', 'Video_Analysis', 'Models', 'tokenizer'))

model = video2report(decoder_name=decoder)
model.load_state_dict(torch.load(os.path.join(path,'Code', 'Video_Analysis', 'Models', 'video2report_fold_1.pth'), weights_only=True))

data_train, data_test = train_test_split(data, test_size=0.05, shuffle=True, random_state=99)
#data_train, data_test = train_test_split(data, test_size=0.05, shuffle=True, random_state=13)

video2report_data_test = video2reportDataset(data=data_test, tokenizer=tokenizer)

test_dataloader = DataLoader(video2report_data_test)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

text_input = infer('Ablauf:')
batch = test_dataloader.dataset.__getitem__(0)
b_input_ids = batch['inputs'].to(device) 

greedy_config = GenerationConfig(max_new_tokens = 128,
                            do_sample = True, 
                            top_k = 50, 
                            top_p = 0.85,
                            renormalize_logits = True)

beam_config = GenerationConfig(max_new_tokens = 128,
                            early_stopping = True,
                            do_sample = True, 
                            top_k = 50, 
                            top_p = 0.85,
                            num_beams = 4,
                            renormalize_logits = True)


input_text = text_input[0].to(device)
input_attn = text_input[1].to(device)
for i in range(5):
    greedy_out = model.generate(x = b_input_ids.unsqueeze(0), inputs = input_text, attn_mask = input_attn, generation_config=greedy_config)
    input_text = greedy_out
    input_attn = torch.ones(greedy_out.size()[1], device=device).unsqueeze(0)

greedy_out = tokenizer.decode(greedy_out[0], skip_special_tokens=True)
print(greedy_out) 



input_text = text_input[0].to(device)
input_attn = text_input[1].to(device)
for i in range(5):
    beam_out = model.generate(x = b_input_ids.unsqueeze(0), inputs = text_input[0].to(device), attn_mask = text_input[1].to(device), generation_config=beam_config)
    input_text = beam_out
    input_attn = torch.ones(beam_out.size()[1], device=device).unsqueeze(0)
    
beam_out = tokenizer.decode(beam_out[0], skip_special_tokens=True)
print(beam_out)