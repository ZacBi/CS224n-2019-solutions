#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, embed_size: int, m_word: int, k: int = 5, f: int = None):
        """ 
        Init CNN which is a 1-D cnn.

        @param embed_size (int): embedding size (dimensionality)
        @param k: kernel size, also called window size
        @param f: number of filters, should be embed_size fo application
        """

        super(CNN, self).__init__()
        if not f:
            f = embed_size
        self.conv1d = nn.Conv1d(in_channels=embed_size,
                                out_channels=f,
                                kernel_size=k)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=m_word-k+1)

    
    def forward(self, ):
        


### END YOUR CODE
