## Polynomial Space Exponential Time SVP Algorithm

**Overview**

This repository implements a polynomial space, exponential time algorithm for the Shortest Vector Problem in Euclidean lattices. The algorithm samples vectors from 1-dimensional Gaussian joints with uniform distribution on the $n$-dimensional unit sphere ($\sim e^{-\frac{\|x\|^2 - R^2}{\sigma}}$). By subtracting the modulo vector, the algorithm generates lattice vectors as potential SVP solutions.

<!--- **Usage**
Provide instructions on how to use the code, including any necessary dependencies and configuration options. --->

**TO DO**

- Determine the optimal sample size required for accurate regression analysis to assess the algorithm's performance and efficiency. 
- (Maybe) Use discrete sampling to improve the sampling process and reduce computational overhead.
- **(50%)** Rigorous analysis of the algorithm's error bounds.
- GPU Implementation (CuPy).
- Experimenting on Higher Dimensional Lattices.
- **(DONE)** *Multi Process Implementation.*
- **(DONE)** *Preliminary Test.*
