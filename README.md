# Implementation of Paper "Parameter Estimation for Linear Dynamic Systems"

This is an implementation of the EM algorithm described in the paper [Parameter Estimation for Linear Dynamic Systems by Ghahramani and Hinton][GH96].

All formulas referenced only by a number reference to the above paper.




## Results

There were three experiments (multi-dimensional, one-dimensional state and one-dimensional state and observation) done, of which
the raw results can be found in in [results](results) directory.



### Multi-Dimensional

The states (and observations) were generated using the following parameters:

![multi-dimensional params](results/equations/multi-dimensional.png)

The algorithm converged after 18739 iterations with a final x-loss of approx. 2.67. The following plot shows the behavior of the
loss vs. the iterations:

![multi-dimensional loss](results/multi-dimensional_loss.png)



### One-Dimensional State

The states (and observations) were generated using the following parameters:

![one-dimensional-state params](results/equations/one-dimensional-state.png)

The algorithm converged after 830 iterations with a final x-loss of approx. 5.55. The following plot shows the behavior of the
loss vs. the iterations:

![one-dimensional-state loss](results/one-dimensional-state_loss.png)

The following plot shows the true and the estimated state vs. the time steps:

![one-dimensional-state states](results/one-dimensional-state_states.png)



### One-Dimensional State and Observation

The states (and observations) were generated using the following parameters:

![one-dimensional-state-observation params](results/equations/one-dimensional-state-observation.png)

The algorithm converged after 572 iterations with a final x-loss of approx. 11.89. The following plot shows the behavior of the
loss vs. the iterations:

![one-dimensional-state-observation loss](results/one-dimensional-state-observation_loss.png)

The following plot shows the true and the estimated state vs. the time steps:

![one-dimensional-state-observation states](results/one-dimensional-state-observation_states.png)





[GH96]: https://pdfs.semanticscholar.org/2e31/70f91e1d8037f8ba03286fa5ddd347a0b88e.pdf
