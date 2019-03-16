#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
run.py: Run the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
"""
from datetime import datetime
import os
import pickle
import math
import time

from torch import nn, optim
import torch
from tqdm import tqdm

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

# -----------------
# Primary Functions
# -----------------
def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """
    best_dev_UAS = 0


    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###      1) Construct Adam Optimizer in variable `optimizer`
    ###      2) Construct the Cross Entropy Loss Function in variable `loss_func`
    ###
    ### Hint: Use `parser.model.parameters()` to pass optimizer
    ###       necessary parameters to tune.
    ### Please see the following docs for support:
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss


    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size
    @param lr (float): learning rate

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    parser.model.train() # Places model in "train" mode, i.e. apply dropout layer
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            ### YOUR CODE HERE (~5-10 lines)
            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step


            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    # Note: Set debug to False, when training on entire corpus
    debug = True
    # debug = False

    assert(torch.__version__ == "1.0.0"),  "Please install torch version 1.0.0"

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
