# CS 224n Assignment #3: Dependency Parsing

### 1. Machine Learning & Neural Networks

#### (a)

<!-- **i** We all konw that as our parameter $\theta$ closing to the convex minima, the gradient of loss function wrt $\theta$ will be smaller, when our loss func hits the best solution, $\nabla_\theta{J}$ may become 0. So in the update method,  -->
$\mathrm{i.}$&emsp;Image the shape of loss function $J$ is a bowl in 3-D space. We could consider the gradient of $J$ wrt $\theta$, is a 'force'(also could be acceleration) push the loss function downward the slope of the bowl, and the $m$ is a 'velocity' controls the speed and direction. With the help of acceleration, the loss function $J$ could be more faster to convergence, however, the factor(hyperparameter) $\beta_1$ could be considered as a friction force preventing $J$ from overshooting.
&emsp;&emsp;The low variance could help $J$ reduce the vibration on the verical direction while downward to the convergence point.

$\mathrm{ii.}$&emsp;

#### (b)

$\mathrm{i.}$&emsp;We know that for a matrix $A$, the expectation is:
$$\mathbb{E}[\mathbf{A}]_i = \mathbb{E}[\mathbf{A}_i]$$
&emsp;&emsp;So, the expected value of $\mathbf{h}_{drop}$ is:

$$\begin{aligned}
    \mathbb{E}_{p_{drop}}[\mathbf{h}_{drop}]_i &= \mathbb{E}_{p_{drop}}[\gamma \mathbf{d} \circ \mathbf{h}]_i \\
    &= \gamma \mathbb{E}_{p_{drop}}[d_i h_i] \\
    &= \gamma 0.8h_i
\end{aligned} $$

&emsp;&emsp;Now we make $\gamma 0.8h_i = h_i$, so $\gamma = 1.25$
    
$\mathrm{ii.}$&emsp;If we apply dropout during evaluation, we'll get a random(uncertain) result for the keep_prob, that's bad for evaluation.

### 2. Neural Transition-Based Dependency Parsing

#### (a)

<!-- $$
\begin{array}{|l|l|l|l} Stack & Buffer & New dependency & Transition \\
\hline
[Root] & [this, sentence, correctly] & ROOT \rightarrow pased & RIGHT-ARC \\
[Root, this] & [sentence, correctly] &  & SHIFT \\
[Root, this, sentence] & [correctly] &  & SHIFT \\
[Root, sentence] & [correctly] & sentence \rightarrow this & LEST-ARC \\SHIFT \\
\end{array}$$ -->

&emsp;&emsp;We can't remove parsed at the first step otherwise it wll disobey the condition.

$$
\begin{array}{|l|l|l|l} Stack & Buffer & New dependency & Transition \\
\hline
[Root, parsed] & [this, sentence, correctly] & parsed \rightarrow I & LEFT-ARC \\
[Root, parsed, this] & [sentence, correctly] &   & SHIFT \\
[Root, parsed, this, sentence] & [correctly] &   & SHIFT \\
[Root, parsed, sentence] & [correctly] & this \rightarrow sentence & LEFT-ARC \\
[Root, parsed] & [correctly] & parsed \rightarrow sentence & RIGHT-ARC \\
[Root, parsed, correctly] & [] & & SHIFT \\
[Root, parsed] & [] & parsed \rightarrow correctly & RIGHT-ARE \\
[Root] & [] & Root \rightarrow parsed & RIGHT-ARE
\end{array}$$

#### (b)

&emsp;&emsp;$n-1$ steps.
&emsp;&emsp;There is and only is a dependency between two words, and every word only relate to a dependency. So there is $n-1$ steps.(I don't consider ROOT, otherwise it's n steps)

#### (c)

&emsp;&emsp;

