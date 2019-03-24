# CS 224n: Assignment #4

## 1.  Neural Machine Translation with RNNs

### (g)

&emsp;&emsp;First, the mask operation set $e_t[src\_len:]$ to negative infinity, i.e., the bit of 'pad' in $e_t$ becomes negative infinity. So we know the effect of mask operatin is to make the prob of 'pad' in the attention vector($\alpha_t$ in the PDF) to be zero.
&emsp;&emsp;If we don't apply mask operation, the decode will use the information of hidden states of 'pad', and the $O_t$ may be predicted as 'pad', that's what we don't expect.