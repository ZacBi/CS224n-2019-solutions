# Final project

This is a note for CS224 final project.
For more solutions of CS224n assignments, you could access to my another repository: [CS224n-2019-solutions](https://github.com/ZacBi/CS224n-2019-solutions).
The final project is **under development**.

## To-Do list

**reading**:

- [x] final-project-practical-tips
- [x] default-final-project-handout
- [x] project-proposal-instructions
- [ ] project-milestone-instructions
- [x] Practical Methodology_Deep Learning book chapter
- [ ] Highway Networks
- [ ] Zero-Shot Relation Extraction via Reading Comprehension

### optim

- [x] Layer Normalization
- [x] Batch Norm
- [ ] Dropout
- [ ] Gradient flow in recurrent nets

### models

- [ ] Transformer-XL Language Modeling with Longer-Term Dependency
- [ ] BiDAF: Bidirectional Attention Flow for Machine Comprehension
- [x] Transformer: Attention is all you need

## Question & Answer

### Q: why we use single weight matrix for *similarity matrix* **S**?

A: recall three types of attention(just in computation way):

- dot product
- additive attentio
- multiplicative attention

these tree types of attention scoring functions could as below. for consistency, we use $c_i$ as context, $q_j$ as query.

$$
score(c, q) = c_i^T q_j \\ c_i^T W q_j \\ W[c_i; q_j]
$$
the third equation could also be written as $W_1c_i + W_2q_j$

### Q: What is logit?

A: separate the word 'logit': log it, 'it' refers the Odds:
$$Odds(a) = \cfrac{p}{1 - p}$$
However, in some code of NN, logit should refer to 'unnormalized probability', you can consider it like scores.

### Q: what's the difference between hidden state(h_n) and cell state (c_n)

A: Recall LSTM model.
&nbsp;

## concepts and terms

### Data shift

### (non) saturating nonlinearities

### Four types of Normalization

#### Batch Norm

#### Layer Norm

#### Weight Norm

#### Instance Norm
