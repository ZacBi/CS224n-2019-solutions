#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()        
        self.charDecoder = nn.LSTM(char_embedding_size,hidden_size,batch_first=True) #bias = True
        self.char_output_projection = nn.Linear(hidden_size,len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id),char_embedding_size,padding_idx=target_vocab.char2id['<pad>']) 
        self.target_vocab = target_vocab

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        #print('size of input is',input.size())
        input = input.permute(1,0).contiguous()
        ip_embedding=self.decoderCharEmb(input)# F.embedding(source_padded, self.model_embeddings.source.weight)
        #X = nn.utils.rnn.pack_padded_sequence(src_padded_embedding,source_lengths)

        #ip_embedding = ip_embedding.permute(1,0,2).contiguous()

        output,(h_n,c_n) = self.charDecoder(ip_embedding,dec_hidden)
        #print('shape of hidden is',h_n.size())
        s_t = self.char_output_projection(output)
        #print('shape of logits is',s_t.size())
        s_t = s_t.permute(1,0,2).contiguous()
        
        return s_t,(h_n,c_n)
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        input = char_sequence[:-1,:]
        output = char_sequence[1:,:]
        #print(input)
        #print(output)
        target = output.reshape(-1)
        #print('shape of target',target.shape)
        s_t,(h_n,c_n) = self.forward(input,dec_hidden)
        #print('shape of s_t',s_t.shape) 
        s_t_shape = s_t.shape
        s_t_re = s_t.reshape(-1,s_t.shape[2])


        #print('shape of s_t_re',s_t_re.shape) 
        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'],reduction='sum') 

        return loss(s_t_re,target)
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        decodedWords = []
        current_char = self.target_vocab.start_of_word
        start_tensor = torch.tensor([current_char],device=device)
        #print('size of start_tensor is',start_tensor.shape)
        batch_size = initialStates[0].shape[1]
        start_batch = start_tensor.repeat(batch_size,1)
        #print('size of start_batch is',start_batch.shape)
        embed_current_char = self.decoderCharEmb(start_batch)
        #print('size of embed_current_char is',embed_current_char.shape)
        h_n,c_n = initialStates
        output_word = torch.zeros((batch_size,1),dtype=torch.long,device=device)      
        for t in range(0,max_length):
          #h_n,c_n = self.charDecoder(embed_current_char,(h_n,c_n))
          # s_t,(h_n,c_n) = self.forward(embed_current_char,(h_n,c_n))
          #print('shape of embed_current_char is',embed_current_char.shape)
          output,(h_n,c_n) = self.charDecoder(embed_current_char,(h_n,c_n))
          s_t = self.char_output_projection(output)
          #print(s_t.shape)
          st_smax = nn.Softmax(dim=2)(s_t) 
          p_next = st_smax.argmax(2)
          current_char = p_next
          embed_current_char = self.decoderCharEmb(current_char)
          #decodedWords.append(self.target_vocab.id2char[current_char])
          #print('*** size of current_char is',current_char.size())
          output_word = torch.cat((output_word,current_char),1)
        #Convert output_word tensor to list and each element to char and put together in decodedWords
        out_list = output_word.tolist()
        out_list = [[self.target_vocab.id2char[x] for x in ilist[1:]] for ilist in out_list]
        decodedWords = []
        for string_list in out_list:
           stringer = ''
           for char in string_list:
              if char!='}':
                 stringer = stringer+char
              else:
                 break
           decodedWords.append(stringer)
        return decodedWords
       ### END YOUR CODE

