#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        # there is two problems
        # 1. Now that embed_size is for output,
        # so why A4 code take embed_size as param for self.embeddings?
        # remember we take e_{char} = 50
        # 2. VocabEntry object doesn't own the attribute 'src'


        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size = embed_size
        char_embed_size = 50
        self.char_embedding = nn.Embedding(len(vocab.char2id),
                                           char_embed_size,
                                           pad_token_idx)
        self.convNN = CNN(f=self.embed_size)
        self.highway = Highway(embed_size=self.embed_size)
        self.dropout = nn.Dropout(p=0.3)

        ### END YOUR CODE

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        
        X_word_emb_list = []
        # divide input into sentence_length batchs
        for X_padded in input:
            X_emb = self.char_embedding(X_padded)
            X_reshaped = torch.transpose(X_emb, dim0=-1, dim1=-2)
            # conv1d can only take 3-dim mat as input
            # so it needs to concat/stack all the embeddings of word
            # after going through the network
            X_conv_out = self.convNN(X_reshaped)
            X_highway = self.highway(X_conv_out)
            X_word_emb = self.dropout(X_highway)
            X_word_emb_list.append(X_word_emb)

        X_word_emb = torch.stack(X_word_emb_list)
        return X_word_emb

        ### END YOUR CODE
