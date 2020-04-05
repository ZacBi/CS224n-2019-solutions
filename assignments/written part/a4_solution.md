# CS 224n: Assignment #4

## 1.  Neural Machine Translation with RNNs

### (g)

&emsp;&emsp;First, the mask operation sets $e_t[src\_len:]$ to negative infinity, i.e. the bit of 'pad' in $e_t$ becomes negative infinity. So we know the effect of mask operatin is to make the prob of 'pad' in the attention vector($\alpha_t$ in the PDF) to be zero.
&emsp;&emsp;If we don't apply mask operation, the decode will use the information  of 'pad' of hidden states, and $O_t$ maybe predicted as 'pad', that's what we don't expect.

### (j)

&emsp;&emsp;Dot product attention is computationally easy and directly, the disadvantage is too easy to get the true informations between $h^{dec}$ and $h^{enc}$.
&emsp;&emsp;Multiplicative attention seems like a transition between dot product and additive attention. It is similar in dimentionality as additive attention, and it run fater and is more space-efficient than the latter in small dimentionality. But it shows declining trend in large dim.
&emsp;&emsp;The disadvantage of additive attention is that we need more hyperparameters to be tuned such as $W_1, W_2$ and $V$ in the equation. Howerver, it can fit more complex situation and in experiment the additive attention always outperform the two others.

## 2. Analyzing NMT Systems

### (a)

Sorry for my knowledge level, I can't give the real reason behind the intuition.
$\mathrm{i.}$ speciﬁc linguistic construct
$\mathrm{ii.}$ speciﬁc linguistic construct
$\mathrm{iii.}$ speciﬁc model limitations(We can't know the unkonw word)
$\mathrm{iv.}$ speciﬁc model limitations
$\mathrm{v.}$ speciﬁc model limitations(the model may pay more attetion to the word 'she', so the result is "the women's")
$\mathrm{vi.}$ speciﬁc model limitations(the model can't distinguish the difference between French unit and American unit)

### (b)

### (c)
$\mathrm{i.}$ For $\mathbf{c}_1$:
$
p_1 = 0.6 \\
p_2 = 0.75 \\
r^* = 4 \\
BP = 1 (c_1 > r^*) \\
BLEU = 0.67 \\
$
for $\mathbf{c}_2$:
$
p_1 = 1 \\
p_2 = 0.5 \\
r^* = 4 \\
BP = 1 (c_1 > r^*) \\
BLEU = 0.70
$

So $\mathbf{c}_2$ get higher BLEU score. I agree with it.

$\mathrm{ii.}$ For $\mathbf{c}_1$:
$
p_1 = 0.6 \\
p_2 = 0.4 \\
r^* = 6 \\
BP = exp(-0.2) \\
BLEU = 0.401 \\
$
for $\mathbf{c}_2$:
$
p_1 = 0.4 \\
p_2 = 0.2 \\
r^* = 4 \\
BP = epx(-0.2) \\
BLEU = 0.23
$

So $\mathbf{c}_1$ get higher BLEU score. In reality, $\mathbf{c}_2$ is a better translation.
    
$\mathrm{iii.}$ We can get the reason from the question above. A single reference translation make our candidate sentence particular, somehow, it's overfitting.

$\mathrm{iv.}$
Advantages:
- It's convenient and fast, a good substitue of human evaluation.
- In the corpus level, the score of BLEU is close to human's score.
  
Disadvantages:
- It only works well on the corpus level because any zeros in precision scores will zero the entire BLEU score.
- BLEU score as presented suffers for only comparing a candidate translation against a single reference, which is surely a noisy representation of the relevant n-grams that need to be matched.