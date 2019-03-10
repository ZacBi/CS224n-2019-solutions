# Answer to assignment #2

### 1 Understanding word2vec

 **(a)**  As described in the doc, $\boldsymbol{y}$ is a one-hot vector with a 1 for the true outside word $o$, that means $y_i$ is 1 if and only if $i == o$. so the proof could be below:
<!-- $ - \sum_{w\in Vocab}y_w\log(\hat{y}_o) = $ -->

$\begin{aligned}
    - \sum_{w\in Vocab}y_w\log(\hat{y}_w) &= - [y_1\log(\hat{y}_1) + \cdots + y_o\log(\hat{y}_o) + \cdots + y_w\log(\hat{y}_w)] \\
    & = - y_o\log(\hat{y}_o) \\
    & = -\log(\hat{y}_o) \\
    & = -\log \mathrm{P}(O = o | C = c)
\end{aligned}$

**(b)** $J_{naive-softmax}(v_c, o, U) = -y^T\log{\hat{y}}$
so the derivative wrt $y, \hat{y}, U$ is:

$\begin{aligned}
\cfrac{\partial J }{\partial \hat{y}} &= -\cfrac{\partial y^T\log{\hat{y}}}{\partial \hat{y}} \\ 
&= -\cfrac{\partial< y, \log\hat{y}>}{\partial \hat{y}} \\
&= -\cfrac{\partial<\log\hat{y}^T, y>}{\partial \hat{y}} \\
&= \log'\hat{y}^T \circ y
\end{aligned}$

