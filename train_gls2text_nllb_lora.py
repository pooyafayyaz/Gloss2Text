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
from torch.nn import CrossEntropyLoss

from peft import LoraConfig, get_peft_model 
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftModel

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

batch_size = 8
annotation_dir = "./manual"
epochs = 50 
    
evauation = False
use_embd = False 

if evauation:
    epochs =1

len_list  = [50,60,70,80,90]
beam_list = [1,2,3,4,7,10]
length_penalty = [0, 1,2,5,7,10]

loss_encoder = nn.CTCLoss(blank=0,zero_infinity=True, reduction='none')

def save_list_to_file(text_list, filename):
    with open(filename, 'w') as file:
        for item in text_list:
            file.write("%s\n" % item)

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


def train(chatData, chatDataTest , model, optim,scheduler):
    translation_loss_fun = SALSLoss(
        pad_index=1, 
        smoothing=0.1, sim_ids = sim_ids, sim_matrix = sim_matrix,  sim_text=list_of_tokens, tokenizer= tokenizer)
    
    # translation_loss_fun = CrossEntropyLoss()
    
    best_ac = 0 

    for i in tqdm.tqdm(range(epochs)):
        
        lr = scheduler.optimizer.param_groups[0]["lr"]
        print("LR", lr)
        

        model.train(True)            
        total_loss = 0 

        for X,Y in chatData:
            if evauation:
                continue
            
            optim.zero_grad()
            
            X, labels = X[0],X[1]
            X = X.to(device)
            labels = labels.to(device)
            
            att_mask = Y[1].to(device)
            Y = Y[0].to(device)
               
            output  = model(input_ids = Y, attention_mask=att_mask , decoder_input_ids = X, labels = labels).logits
            
            log_prob = torch.nn.functional.log_softmax(output, dim=-1)  # B, T, L
                        
            batch_loss_sum = translation_loss_fun(log_prob[0],labels[0], decoded_texts = tokenizer.batch_decode(labels[0], skip_special_tokens=True) )
            loss = batch_loss_sum/log_prob.shape[0]

            loss.backward()
            optim.step()
            total_loss += loss
            scheduler.step()

        print("infer from model : ")
        model.train(False)
        
        val_acc = infer(chatDataVal,model)
        
        print("Epoch : {}".format( i+1 ))
        print(val_acc)
        print("Train Loss : " , total_loss/len(chatData) )            


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
        
    if evauation:
        with open("res_nllb.csv", 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)  
        
            # writing the fields  
            csvwriter.writerow(['gloss', 'translation', 'prediction'])  

        for i in range(len(gt_labels)):
            if bleu([gt_labels[i]], [preds[i]])['bleu1']< 30:
                print("###################################################################################")
                print(bleu([gt_labels[i]], [preds[i]]))
                print(glosses[i], '\n' , preds[i], '\n',  gt_labels[i] )
                print("###################################################################################")
    import pdb; pdb.set_trace()
    
    return report_all(gt_labels, preds)

    # return bleu(gt_labels, preds)


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(device)

tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-3.3B', src_lang="deu_Latn", tgt_lang="deu_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-3.3B')

model = model.to(device)

print("LORA: Freezing all parameters")
for param in model.parameters():
    param.requires_grad = False
    
config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling,
    target_modules=[
"q_proj",
"k_proj",
"v_proj",
"o_proj",
"fc1",
"fc2"
"out_proj",
"gate_proj",
"up_proj",
"down_proj",
"lm_head",
],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM 
)

chatDataTr = GlossData(annotation_dir , "train", tokenizer, type = "enc_dec_nllb", use_embd = use_embd, back_trans = False, model = model)
model,tokenizer = chatDataTr.get_model()

if evauation:
    model = PeftModel.from_pretrained(model,'pretrained/', is_trainable=False)
else:
    model = get_peft_model(model, config)

print_trainable_parameters(model)

chatDataTe = GlossData(annotation_dir , "test", tokenizer, type = "enc_dec_nllb", use_embd = use_embd)

chatDataDe = GlossData(annotation_dir , "dev", tokenizer, type = "enc_dec_nllb", use_embd = use_embd)

chatDataTrain =  DataLoader(chatDataTr, batch_size=batch_size,shuffle=True, collate_fn=collate_fn)
sim_ids, sim_matrix , list_of_tokens = chatDataTe.return_sim()

chatDataVal =  DataLoader(chatDataDe, batch_size=16, collate_fn=collate_fn)
chatDataTest =  DataLoader(chatDataTe, batch_size=16, collate_fn=collate_fn)

model.train()

optim = torch.optim.AdamW(
            params=model.parameters(),
            lr=5e-5,
            betas=(0.9,0.998),
            eps=1.0e-8,
            weight_decay=0.001,
            amsgrad=False,)

num_training_steps =  epochs * len(chatDataTrain)

scheduler = get_scheduler(
    name=SchedulerType.COSINE_WITH_RESTARTS, optimizer=optim, num_warmup_steps=0, num_training_steps=num_training_steps
) 

print("training .... ")
train(chatDataTrain, chatDataTest, model, optim,scheduler)
