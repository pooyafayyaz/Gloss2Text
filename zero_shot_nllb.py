import torch

from gloss_data import GlossData
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
import tqdm
import torch
from metrics import bleu,report_all 
import metrics

from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pickle
from torch import nn

from transformers import MBartForConditionalGeneration, MBartTokenizer
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from loss import XentLoss
from transformers import get_scheduler
from transformers import NllbTokenizer, SchedulerType
import csv  

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
use_embd = False 



batch_size = 1
annotation_dir = "/scratch/nnejatis/pooya/translation2/manual"
path = "/scratch/nnejatis/pooya/SLRT/TwoStreamNetwork/experiments/outputs/SingleStream/phoenix-2014t_g2t/ckpts/best.ckpt"
epochs = 150 
    
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")               

def load_pretrained_model(path, model):
    state_dict = torch.load(path, 'cuda:0')
    
    remove_prefix = 'translation_network.model.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict['model_state'].items()}
    state_dict.pop('translation_network.gloss_embedding.weight')

    model.load_state_dict(state_dict)
    return model 

def collate_fn(batch):
    input_ids, input_embds , lenghts_dec, lenghts_enc = zip(*batch)
    
    
    max_len_enc = max(lenghts_enc)
    max_len_dec = max(lenghts_dec)
    
    bs = len(batch)
    
    input_embd = torch.ones((bs, max_len_enc))
    input_mask = torch.zeros((bs, max_len_enc))
    
    input_id = torch.ones((bs, max_len_dec))
    labels = torch.ones((bs, max_len_dec))
    
    for i in range(len(input_embds)):
        j = len(input_embds[i][0])
        
        input_embd[i][:j] = input_embds[i][0]
        input_mask[i][:j] = 1

    for i in range(len(input_ids)):
        j = len(input_ids[i][0])
        
        input_id[i][:j] = input_ids[i][0]
        labels[i][:j-1] = input_ids[i][0][1:]
    
    
    return (input_id.long(),labels.long()), (input_embd.long() ,input_mask.long())


def infer(chatData,model,beam = 5, max_length = 100 ,length_penalty =1 ):
    
    preds = []
    gt_labels = []
    glosses = []
    
    for X,Y in chatData:            
        X, labels = X[0],X[1]
        X = X.to(device)
        labels = labels.to(device)
        
        att_mask = Y[1].to(device)
        Y = Y[0].to(device)
                        
        decoder_input_ids = torch.ones([Y.size(0),1],dtype=torch.long, device=Y.device) * tokenizer.lang_code_to_id["deu_Latn"]
        output = model.generate(input_ids = Y, attention_mask=att_mask, num_beams=beam, max_length=max_length, length_penalty=1,
                                decoder_input_ids=decoder_input_ids)


        for i in range(output.shape[0]):             
            for idx,el in enumerate(output[i]):
                if el.item() == 2 and idx != 0:
                    preds.append(tokenizer.decode(output[i][2:idx-1]) + ' .')
            # print(tokenizer.decode(output[i]))

        X = X.clone()                
        for i in range(X.shape[0]):
            for idx,el in enumerate(X[i]):
                if el.item() == 2 :
                    gt_labels.append(tokenizer.decode(X[i][1:idx-1]) + ' .')
        
        
        Y = Y .clone()                
        for i in range(Y.shape[0]):
            for idx,el in enumerate(Y[i]):
                if el.item() == 2 :
                    glosses.append(tokenizer.decode(Y[i][1:idx]))
        
    return report_all(gt_labels, preds)


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(device)

tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-3.3B', src_lang="deu_Latn", tgt_lang="deu_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-3.3B')

model = model.to(device)
model.train(False)

print_trainable_parameters(model)

chatData = GlossData(annotation_dir , "test", tokenizer, type = "enc_dec_nllb", use_embd = use_embd)
chatDataTest =  DataLoader(chatData, batch_size=16, collate_fn=collate_fn)

chatData = GlossData(annotation_dir , "dev", tokenizer, type = "enc_dec_nllb", use_embd = use_embd)
chatDataVal =  DataLoader(chatData, batch_size=16, collate_fn=collate_fn)

print("Zero Shot test .... ")

val_acc = infer(chatDataTest,model)

print(val_acc)
# print("BLEU-1: {}, BLEU-2: {}, BLEU-3: {}, BLEU-4: {}".format( val_acc["bleu1"],val_acc["bleu2"],val_acc["bleu3"],val_acc["bleu4"]))

print("Zero Shot val .... ")

val_acc = infer(chatDataVal,model)
print(val_acc)
    