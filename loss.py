import torch
from torch import nn, Tensor
from torch.autograd import Variable
import numpy as np
import pickle

class SALSLoss(nn.Module):

    def __init__(self, pad_index: int, smoothing: float = 0.0, sim_ids = None,sim_matrix =None, sim_text= None, tokenizer= None):
        super(SALSLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        self.sim_ids,self.sim_matrix= sim_ids,sim_matrix
        self.tokenizer = tokenizer
        self.sim_smoothing = 0.2
        with open('../sim_all.pkl', 'rb') as file:
            self.sim_all = pickle.load(file)
        
        
        with torch.no_grad():
            self.sim_matrix = (torch.from_numpy(self.sim_matrix)+1)/2
            self.sim_text = sim_text
                            
            self.sim_matrix =  torch.full(self.sim_matrix.shape, 0.2 / (len(self.sim_matrix) - 2) )          
            self.sim_matrix_expand = torch.zeros((256206,256206))

            for ii in range(len(self.sim_matrix)):
                self.sim_matrix_expand[ii][self.sim_ids] = self.sim_matrix[ii]
            
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")


    def _smooth_targets2(self, targets: Tensor, vocab_size: int, txt_target):     
        """
        Constructs similarity vectors for each token in the target text.
        
        Args:
        - targets (Tensor): The target tensor containing token indices.
        - vocab_size (int): The size of the vocabulary.
        - txt_target (str): The input text to tokenize.
        
        Returns:
        - Tensor: A tensor containing the similarity vectors for each token.
        """

        # Split the text into simple words 
        words = txt_target.split()
        
        # Initialize the similarity vectors
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        masks = torch.zeros_like(smooth_dist)
        smooth_dist.fill_(self.smoothing / (len(self.sim_ids) - 1))
        i = 0

        # Create the similarity vectors
        for word in words:
            tokenized_word = self.tokenizer(word, add_special_tokens=False)['input_ids']            
            smooth_dist[i] = (self.sim_smoothing / np.sum(self.sim_all[word])) * self.sim_all[word]
            i += len(tokenized_word)

        masks[:, self.sim_ids] = 1
        smooth_dist *= masks

        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - (self.smoothing + self.sim_smoothing ))
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0

        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Constructs targeted label smoothing where it only focuses on the target vocabulary.
        
        Args:
        - targets (Tensor): The target tensor containing token indices.
        - vocab_size (int): The size of the vocabulary.
        
        Returns:
        - Tensor: A tensor containing the similarity vectors for each token.
        """
        
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        masks = torch.zeros_like(smooth_dist)
        smooth_dist.fill_(self.smoothing / (len(self.sim_ids) - 1))

        masks[:, self.sim_ids] = 1
        smooth_dist *= masks

        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)

        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)


    def _smooth_targets3(self, targets: Tensor, vocab_size: int):
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        smooth_dist[:, self.pad_index] = 0
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)


    def forward(self, log_probs, targets, txt_target):
        
        if txt_target != None:
            targets = self._smooth_targets2(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1), txt_target
            )
        else:
            if self.sim_ids:
                targets = self._smooth_targets(
                    targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
                )
            else:
                targets = self._smooth_targets3(
                    targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
                )

            
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )

        return loss