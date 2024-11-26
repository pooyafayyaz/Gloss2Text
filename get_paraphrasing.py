import torch 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import NllbTokenizer
import gzip,pickle



tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-3.3B', src_lang="deu_Latn", tgt_lang="eng_Latn")
tokenizer_rev = NllbTokenizer.from_pretrained('facebook/nllb-200-3.3B', tgt_lang="deu_Latn", src_lang="eng_Latn")

model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-3.3B')
model.to("cuda")

with gzip.open("./data/phoenix-2014t/phoenix-2014t_cleaned."+'train','rb') as f:
    data = pickle.load(f)

gloss = []
translation = []

for i in range(len(data)):
    gloss.append(data[i]['gloss'].lower())
    translation.append(data[i]['text'])

back_trans = []
for i in tqdm(range(len(translation))):

    Y = tokenizer.encode(translation[i],  return_tensors="pt").to("cuda")  

    de2en = model.generate(input_ids = Y, num_beams=5, max_length=50, length_penalty=1,decoder_input_ids=torch.tensor([[256047]], device= 'cuda'))
    en2de = model.generate(input_ids = de2en, num_beams=5, max_length=50, length_penalty=1,decoder_input_ids=torch.tensor([[256042]], device= 'cuda'))
    back_trans.append(tokenizer_rev.decode(en2de[0][2:-2]).lower()+' .')
    

import csv
with open("res_nllb_text2text.csv", 'w') as csvfile:  
    csvwriter = csv.writer(csvfile)  

    # writing the fields  
    csvwriter.writerow(['translation', 'gloss', 'back_trans'])  

    for i in range(len(back_trans)):

        if translation[i] != back_trans[i]:
            csvwriter.writerow([translation[i],gloss[i],back_trans[i]]) 

