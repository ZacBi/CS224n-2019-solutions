# Answer to assignment #2

## 1 Understanding word2vec

### (a)

As described in the doc, $\boldsymbol{y}$ is a one-hot vector with a 1 for the true outside word $o$, so the proof could be below:
<!-- $ - \sum_{w\in Vocab}y_w\log(\hat{y}_o) = $ -->

$\begin{aligned}
    - \sum_{w\in Vocab}y_w\log(\hat{y}_w) &= - [y_1\log(\hat{y}_1) + \cdots + y_o\log(\hat{y}_o) + \cdots + y_w\log(\hat{y}_w)] \\
    & = - y_o\log(\hat{y}_o) \\
    & = -\log(\hat{y}_o) \\
    & = -\log \mathrm{P}(O = o | C = c)
\end{aligned}$

### (b)
