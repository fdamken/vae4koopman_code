# Implementation of Paper "Parameter Estimation for Linear Dynamic Systems"

This is an implementation of the EM algorithm described in the paper [Parameter Estimation for Linear Dynamic Systems by Ghahramani and Hinton][GH96].

All formulas referenced only by a number reference to the above paper.




## Results

There were three experiments (multi-dimensional, one-dimensional state and one-dimensional state and observation) done, of which
the raw results can be found in in [results](results) directory.



### Multi-Dimensional

The states (and observations) were generated using the following parameters:

![multi-dimensional params](http://www.sciweavers.org/tex2img.php?eq=T%20%3D%205%20%5Cquad%0A%5Cpi_1%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%200%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AV_1%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AA%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AQ%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AC%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AR%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D&bc=Transparent&fc=Black&im=png&fs=18&ff=modern&edit=0)

The algorithm converged after 18739 iterations with a final x-loss of approx. 2.67. The following plot shows the behavior of the
loss vs. the iterations:

![multi-dimensional loss](results/multi-dimensional_loss.png)



### One-Dimensional State

The states (and observations) were generated using the following parameters:

![one-dimensional-state params](http://www.sciweavers.org/tex2img.php?eq=T%20%20%3D%20100%20%5Cquad%0A%5Cpi_1%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AV_1%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AA%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AQ%20%3D%20%5Cbegin%7Bbmatrix%7D%202%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AC%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5C%5C%202%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AR%20%3D%20%5Cbegin%7Bbmatrix%7D%202%20%26%200%20%5C%5C%200%20%26%202%20%5Cend%7Bbmatrix%7D&bc=Transparent&fc=Black&im=png&fs=18&ff=modern&edit=0)

The algorithm converged after 830 iterations with a final x-loss of approx. 5.55. The following plot shows the behavior of the
loss vs. the iterations:

![one-dimensional-state loss](results/one-dimensional-state_loss.png)

The following plot shows the true and the estimated state vs. the time steps:

![one-dimensional-state states](results/one-dimensional-state_states.png)



### One-Dimensional State and Observation

The states (and observations) were generated using the following parameters:

![one-dimensional-state-observation params](http://www.sciweavers.org/tex2img.php?eq=T%20%3D%20100%20%5Cquad%0A%5Cpi_1%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AV_1%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AA%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AQ%20%3D%20%5Cbegin%7Bbmatrix%7D%202%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AC%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5Cend%7Bbmatrix%7D%20%5Cquad%0AR%20%3D%20%5Cbegin%7Bbmatrix%7D%205%20%5Cend%7Bbmatrix%7D&bc=Transparent&fc=Black&im=png&fs=18&ff=modern&edit=0)

The algorithm converged after 572 iterations with a final x-loss of approx. 11.89. The following plot shows the behavior of the
loss vs. the iterations:

![one-dimensional-state-observation loss](results/one-dimensional-state-observation_loss.png)

The following plot shows the true and the estimated state vs. the time steps:

![one-dimensional-state-observation states](results/one-dimensional-state-observation_states.png)





[GH96]: https://pdfs.semanticscholar.org/2e31/70f91e1d8037f8ba03286fa5ddd347a0b88e.pdf
