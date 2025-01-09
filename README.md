## Trying to achieve: Polynomial Space Exponential Time SVP Algorithm

**Overview**

It may be a polynomial space, exponential time algorithm for the Shortest Vector Problem in lattices. The algorithm samples vectors from 1-dimensional Gaussian joints with uniform distribution on the $n$-dimensional unit sphere ($\sim e^{-\frac{\|x\|^2 - R^2}{\sigma}}$). By subtracting the modulo vector, the algorithm generates lattice vectors as potential SVP solutions.

<!--- **Usage**
Provide instructions on how to use the code, including any necessary dependencies and configuration options. --->

**TO DO**

- **(0%)** Rigorous analysis of the algorithm's time complexity.
- **(50%)** Two stage algorithm:
	- **(80%)** Constructing short basis
 		- Multi Thread Implementation.
	- **(50%)** Trying on this basis
- **(50%)** Determine the required sample size. (guess: 2^cn, c>0.52)
- **(DONE)** Experimenting on Higher Dimensional Hard Lattices. (80d now!)
- **(DONE)** Multi Thread Implementation.
- **(DONE)** *Preliminary Test.*
- ~~(Maybe) Use discrete sampling to improve the sampling process and reduce overhead.~~
