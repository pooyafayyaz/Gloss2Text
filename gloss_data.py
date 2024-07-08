from torch.utils.data import Dataset
import json
import pandas as pd 
import math
import torch
import random
import torch, os, gzip, pickle, json, numpy as np
from collections import OrderedDict
import re
# from charsplit import Splitter

import fasttext
from sklearn.metrics.pairwise import cosine_similarity

import itertools
import string

replacments = { "ueberschwemmung":"überschwemmung","sued":"süd","tagsueber":"tagsüber" , "fuenf":"fünf",
               "ruegen" :"rügen" , "fruehling": "frühling" , "bluete": "blüte" , "tagsueber":"tagsüber", 
               "begruessen":"begrüßen", "fuehlen":"fühlen", "duenn":"dünn","kuehl":"kühl", "fuenf":"fünf","kueste":"küste",
               "tschuess":"tschüss","trueb":"trüb","duesseldorf":"düsseldorf","ueberwiegen":"überwiegen","muessen":"müssen",
               "gemuetlich":"gemütlich", "ueber":"über", "wuenschen":"wünschen", "ueberall":"überall", "zurueck":"zurück", 
               "thueringe":"thüringe", "luecke":"lücken", "schwuel":"schwül","tuerkei":"türkei", "wuerttemberg":"württemberg","voruebergehend":"vorübergehend","fuer":"für",
               "thueringen":"thüringen", "schuetzen":"schützen","spueren":"spüren", "muenchen":"münchen","puenktlich":"pünktlich", "gluecken":"glücken",
               "wuerz":"würz", "merkwuerdig":"merkwürdig" , "rücken":"rücken", "muenster":"münster", "aufbluehen":"aufblühen", 
               "gruen":"grün", "glück":"glück"}

    

class GlossData(Dataset):
    def __init__(self, path:str ,subset , tokenizer, gloss_embeddings= None , map_ids = None, type="dec", use_embd = False, back_trans = False, model = None):
        with gzip.open("./data/phoenix-2014t/phoenix-2014t_cleaned."+subset,'rb') as f:
            data = pickle.load(f)
        
        with gzip.open("./data/phoenix-2014t/phoenix-2014t_cleaned."+'test','rb') as f:
            data_test = pickle.load(f)
        with gzip.open("./data/phoenix-2014t/phoenix-2014t_cleaned."+'dev','rb') as f:
            data_val = pickle.load(f)
        
        splitter = Splitter()

        self.list_of_tokens = []
        self.gloss = []
        self.translation = []
        self.use_embd = use_embd
        self.model = model        
                     
        annotations = pd.read_csv(path+"/PHOENIX-2014-T."+ subset+ ".corpus.csv", sep="|")
        self.gloss_f = annotations.orth.str.lower().tolist()
        
        # self.translation = annotations.translation.tolist()

        for i in range(len(data)):
            self.gloss.append(data[i]['gloss'].lower())
            self.translation.append(data[i]['text'])
            
        if back_trans and subset =='train' :
            back_trans_data = pd.read_csv("./res_nllb_text2gls.csv")

            for i in range(len(back_trans_data)):
                if isinstance(back_trans_data.iloc[i]['pred_gloss'] , str) :
                    self.gloss.append(back_trans_data.iloc[i]['pred_gloss'][:-1])
                    self.translation.append(back_trans_data.iloc[i]['translation'][:-1] + ' .')
                    
            back_trans_data_txt = pd.read_csv("./res_nllb_text2text.csv")
            for i in range(len(back_trans_data_txt)):
                self.gloss.append(back_trans_data_txt.iloc[i]['gloss'][:-1])
                self.translation.append(back_trans_data_txt.iloc[i]['back_trans'])
                            
                all_words = self.translation[-1].split()
                for sub_words in all_words:
                    if splitter.split_compound(sub_words)[0][0]> 0.90 :
                        self.translation[-1] = self.translation[-1].replace(sub_words ,splitter.split_compound(sub_words)[0][1].lower() + ' ' +  splitter.split_compound(sub_words)[0][2].lower() )

            
                    
        self.type = type
        self.gloss_embeddings = gloss_embeddings
        self.map_ids = map_ids
        
        self.tokenizer = tokenizer
        self.subset =subset
        self.X = []
        
        
        print(type)
        
        
        if "enc_dec" in type:
            
            self.Y = []
            
            self.ids_sim = set()
            
            for idx, i in enumerate(self.gloss):
                gl = i
                tr = self.translation[idx]
                

                self.X.append(gl)
                self.Y.append(tr)
                self.list_of_tokens.extend(self.tokenizer.encode(tr))


            self.list_of_tokens = set(self.list_of_tokens)
            self.list_of_tokens = np.array(list(self.list_of_tokens))
            
            swapped_dict = dict(map(lambda item: (item[1], item[0]), self.tokenizer.get_vocab().items()))
            
            numbers = 0

            self.list_of_texts= []
            
            # for tokens in self.list_of_tokens:
            self.list_of_texts = self.tokenizer.batch_decode(self.list_of_tokens.reshape(-1,1))

            model = fasttext.load_model('cc.de.300.bin')
            embeddings = [model.get_sentence_vector(tokens) for tokens in self.list_of_texts]

            self.similarity_matrix = cosine_similarity(embeddings)

        else:
            for idx, i in enumerate(self.gloss):
                self.X.append("<startofstring> "+ i +" <trans>: "+self.translation[idx]+" <endofstring>")

            print(self.X[0])
            
            #torch.where(torch.all(tokenizer(self.X,max_length=110, truncation=True, padding="max_length" , return_tensors="pt")['attention_mask'], axis=1))
            
            self.X_encoded = tokenizer(self.X, padding=True, return_tensors="pt")
            self.input_ids = self.X_encoded['input_ids']
            self.attention_mask = self.X_encoded['attention_mask']
        

    def build_tokenizer(self,translation, gloss):
        all_embds = {}
        import string
        train_vocabs = set()
        for i in translation+gloss:
            if i != i.translate(str.maketrans('', '', string.punctuation.replace("-", "").replace(".", ""))):
                print(i)

            train_vocabs.update(i.lower().split())
        
        train_vocabs = list(train_vocabs)
        initial_len = 0
        
        # import pdb; pdb.set_trace()
        with torch.no_grad():

            full_embedding_weight = self.model.model.shared.weight
            
            new_embeddings = torch.zeros((len(train_vocabs),full_embedding_weight.shape[1]), device=full_embedding_weight.device)
            
            for i in train_vocabs:
                gls_ids = self.tokenizer(i)['input_ids'][1:-1] # remove</s> <lang>
                emb = []
                            
                for j in gls_ids:
                    emb.append(full_embedding_weight[j,:])
                    
                emb = torch.mean(torch.stack(emb, dim=0), dim=0)
                new_embeddings[initial_len] = emb
                initial_len += 1 

            self.tokenizer.add_tokens(train_vocabs)
            self.model.model.resize_token_embeddings(len(self.tokenizer))
            
            self.model.model.shared.weight.data[-len(train_vocabs):].copy_(new_embeddings)
        
    def get_model(self):
        return self.model,self.tokenizer
    def __len__(self):
        return len(self.gloss)
    def return_sim(self):
        return self.list_of_tokens, self.similarity_matrix, self.list_of_texts
    
    def prepare_gloss_inputs(self, gloss):
        
        gloss = gloss.split()
        gloss = ["deu_Latn"] + gloss + ["</s>"]

        input_emb = torch.zeros(len(gloss),2048)
        
        for idx, gls in enumerate(gloss):
            try:
                input_emb[idx] = self.gloss_embeddings[gls]
            except: 
                print("unrec gloss :", gls)
                input_emb[idx] = self.gloss_embeddings['<unk>']            

        return input_emb*45.0 
    
    def prepare_txt_inputs(self, input_ids):        
        input_ids = input_ids.clone()
        for idx, txt_id in enumerate(input_ids):
            input_ids[idx] = self.map_ids[txt_id.item()]

        input_ids = torch.cat((torch.tensor([6]),input_ids[:-1],torch.tensor([6])))

        return input_ids.unsqueeze(0)
    
    def __getitem__(self, idx):
        
        self.Y_encoded = self.tokenizer(self.Y[idx], padding=True, return_tensors="pt")
        self.target_ids = self.Y_encoded['input_ids']
        self.target_attention_mask = self.Y_encoded['attention_mask']
        
        if self.type== "enc_dec_nllb_bio":
            
            self.X_encoded = self.tokenizer(self.X[idx], padding=True, return_tensors="pt")
            self.input_ids = self.X_encoded['input_ids']
            self.input_attention_mask = self.X_encoded['attention_mask']
            
            self.input_ids[0][0] = 256204
            self.target_ids[0][0] = 256042

            if self.subset == 'train':
                if not random.randrange(3):
                    return  self.input_ids, self.target_ids , len(self.input_ids[0]) , len(self.target_ids[0]) 
            return self.target_ids, self.input_ids , len(self.target_ids[0]) , len(self.input_ids[0]) 

        if self.type== "enc_dec_nllb":
            
            if self.use_embd:
                input_enc = self.prepare_gloss_inputs(self.X[idx])
                return self.target_ids, input_enc , len(self.target_ids[0]) , len(input_enc) 

            
            self.X_encoded = self.tokenizer(self.X[idx], padding=True, return_tensors="pt")
            self.input_ids = self.X_encoded['input_ids']
            self.input_attention_mask = self.X_encoded['attention_mask']

            return self.target_ids, self.input_ids , len(self.target_ids[0]) , len(self.input_ids[0]) 

        if self.type== "enc_dec_nllb_text_gls":
            
            if self.use_embd:
                input_enc = self.prepare_gloss_inputs(self.X[idx])
                return self.target_ids, input_enc , len(self.target_ids[0]) , len(input_enc) 

            
            self.X_encoded = self.tokenizer(self.X[idx], padding=True, return_tensors="pt")
            self.input_ids = self.X_encoded['input_ids']
            self.input_attention_mask = self.X_encoded['attention_mask']

            return  self.input_ids,  self.target_ids , len(self.input_ids[0]) , len(self.target_ids[0])
        
        if self.type== "enc_dec_all":
            self.X_encoded = self.tokenizer(self.X[idx], padding=True, return_tensors="pt")
            self.input_ids = self.X_encoded['input_ids']
            self.input_attention_mask = self.X_encoded['attention_mask']
            

            self.target_ids = torch.cat((torch.tensor([250003]),self.target_ids[0])).unsqueeze(0)

            return self.target_ids, self.input_ids , len(self.target_ids[0]) , len(self.input_ids[0]) 

        if self.type== "enc_dec":
            
            input_dec = self.prepare_txt_inputs(self.target_ids[0])
            input_enc = self.prepare_gloss_inputs(self.gloss[idx])
            
            return input_dec, input_enc , len(input_dec[0]) , len(input_enc) 
        else:
            return (self.input_ids[idx], self.attention_mask[idx])