## Polynomial Space Exponential Time SVP Algorithm

**Overview**

This repository implements a polynomial space, exponential time algorithm for the Shortest Vector Problem in Euclidean lattices. The algorithm samples vectors from 1-dimensional Gaussian joints with uniform distribution on the $n$-dimensional unit sphere ($\sim e^{-\frac{\|x\|^2 - R^2}{\sigma}}$). By subtracting the modulo vector, the algorithm generates lattice vectors as potential SVP solutions.

<!--- **Usage**
Provide instructions on how to use the code, including any necessary dependencies and configuration options. --->

**TO DO**

- **(50%)** Rigorous analysis of the algorithm's time complexity.
- **(50%)** Determine the required sample size. (0.2)
- **(Done)** Experimenting on Higher Dimensional Hard Lattices. (80d now!)
- **(DONE)** Multi Thread Implementation.
- **(DONE)** *Preliminary Test.*
- ~~(Maybe) Use discrete sampling to improve the sampling process and reduce overhead.~~
