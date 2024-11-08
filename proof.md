# Polynomial Space Exponential Time SVP

## Introduction

Let $\mathcal{L}$ be a given lattice and $B\in\mathcal{M}_{n}(\mathbb{R})$ a full rank basis matrix for $\mathcal{L}$. We are interested in finding the vector $\lambda_1(\mathcal{L})$ which is defined as the shortest non-zero vector in $\mathcal{L}$:
$$
\lambda_1=B\cdot (\arg\min_{c\in\mathbb{Z}^n} \|Bc\|)
$$
To achieve this, we give a certain probability measure $\mu$ on $\mathbb{R}^n$ and sample accordingly then we can construct a vector in $\mathcal{L}$ by considering the sampled vector module $B$. Thus if we define $\mathcal{P}(B)$ as the fundamental parallel-piped induced by $B$, the probability of constructing $\lambda_1$ is equal to:
$$
\int_{\mathcal{P}(B)+\lambda_1}1d\mu
$$
which we show is equal to $O(\frac{1}{2^n})$, and yields a poly space, exponential time algorithm for SVP.

## Proof sketch

Let $P$ be the probability measure defined by the bounded measure $ f_\sigma^R(x) = e^{-\frac{\|x\|^2 - R^2}{\sigma}} $ on $\mathbb{R}^n$. The integral we want to compute is:
$$
\int_{\mathcal{P}(B) + s} 1 \, dP(x) = \int_{\mathcal{P}(B) + s} P(x) \, dx = \frac{1}{Z} \int_{\mathcal{P}(B) + s} f_\sigma^R(x) \, dx,
$$
where $ Z = \int_V f(x) \, dx $, and $ \mathcal{P}(B) + s $ is the parallelepiped induced by $B$ and shifted by $s$.

Since computing the value of $Z$ is challenging, we overcome this issue by computing the probability $P(x\in \mathcal{P}(B)+s\mid \|x\|\in[aR,bR])$, where $a\in[0,1]$ and $b>1$. Thus, if we choose $a$ and $b$ appropriately such that for every $x\in\mathcal{P}(B)+s$, we have $\|x\|\in[aR,bR]$, we can be sure that:
$$
P(x\in \mathcal{P}(B)+s)=\frac{P(x\in \mathcal{P}(B)+s\mid \|x\|\in[aR,bR])}{P(\|x\|\in[aR,bR])}=\frac{\int_{\mathcal{P}(B)+s}f(x)dx}{\int_{\|x\|\in[aR,bR]}f(x)dx}
$$
------

One can compute the denominator using the concentration of measure inequalities or rewriting this integral in spherical coordinates and derive incomplete gamma function, then use related bounds on such function [1]. Both these ideas imply that the value we are looking for is $O(\frac{1}{2^n})$.

------

To lower bound the enumerator, we can calculate the integral directly. Substituting $ y = x - s $ and the expanding $ \|y + s\|^2 $ gives:
$$
\int_{\mathcal{P}(B)} e^{-\frac{\|y\|^2 + 2 \langle y, s \rangle + \|s\|^2 - R^2}{\sigma}} \, dy.
$$
We now try to simplify the integral domain, expand the quadratic form $ \|Bv\|^2 $ and the linear term $ \langle Bv, s \rangle $, the integral becomes:
$$
I = |\det(B)| \int_{[0,1]^n} e^{-\frac{v^T B^T B v + 2 v^T B^T s}{\sigma}} \, dv.
$$
Simplifying the exponent, The integral can now be written as:
$$
I = |\det(B)| e^{\frac{s^T B^{-T} B^{-1} s}{\sigma}} \int_{[0,1]^n} e^{-\frac{(v + B^{-1} s)^T B^T B (v + B^{-1} s)}{\sigma}} \, dv.
$$
For simplicity, if we assume the region $[0,1]^n$ is small compared to the decay of the Gaussian, the term inside the integral varies slowly, and the integral over the parallelepiped can be approximated by the volume of the region times the value at the center.

Thus, we approximate the integral over $[0,1]^n$ by the value of the Gaussian term evaluated near the center of the parallelepiped. This gives us:

$$
I \approx |\det(B)| e^{\frac{s^T B^{-T} B^{-1} s}{\sigma}} \cdot \text{Vol}(\mathcal{P}(B)).
$$
Hence, let $s$ be the minimum eigenvector of $B^{-1}: B^{-1}s=\gamma_\min s$, then we have:
$$
\begin{split}
\begin{cases}
s^TB^{-T}=(B^{-1}s)^T=\gamma_\min s^T\\
B^{-1}s=\gamma_\min s
\end{cases}\Longrightarrow {s^T B^{-T} B^{-1} s}=\gamma_\min^2\langle s,s\rangle=\gamma_\min \|s\|^2=\gamma_\min R^2
\end{split}
$$

Which implies the intended result.

## References

1.  Borwein, J., and O-Yeat Chan. "Uniform bounds for the incomplete complementary gamma function." *Mathematical Inequalities and Applications* 12 (2009): 115-121.
