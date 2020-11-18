These results where generated using the original Code from Morton et al.,
modified to not have control inputs (to compare it with our results). See
[GitHub](https://github.com/fdamken/variational-koopman/tree/without-control)
for the code we used. We also added code to save the predictions into a JSON
file rather than only creating the plot.

The results where generated with the following CLI parameters:

| File | Parameters |
| --- | --- |
| `acrobot-64.json` | `--seq_length 64 --n_trials 20 --n_subseq 220 --kl_weight 0.1 --extractor_size 64 64 --inference_size 64 64 --prior_size 64 32 --domain_name ContinuousAcrobot-v0` |
| `cartpole-32.json` | `--seq_length 32 --n_trials 20 --n_subseq 220 --kl_weight 0.1 --extractor_size 64 64 --inference_size 64 64 --prior_size 64 32 --domain_name ContinuousCartPole-v0` |
| `cartpole-sine_cosine-32.json` | `--seq_length 32 --n_trials 20 --n_subseq 220 --kl_weight 0.1 --extractor_size 64 64 --inference_size 64 64 --prior_size 64 32 --domain_name ContinuousSineCosineCartPole-v0` |
| `pendulum-50.json` | `--seq_length 50 --n_trials 20 --n_subseq 220 --kl_weight 0.1 --extractor_size 64 64 --inference_size 64 64 --prior_size 64 32` |
