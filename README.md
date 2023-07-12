# deep-dynamics

## Getting Started

### Bayesrace

1. Data recorded from the [forked Bayesrace simulator](https://github.com/linklab-uva/bayesrace) can be converted to training format by running `python3 tools/bayesrace_parser.py {path to bayesrace data} {horizon}`, where horizon is the historical horizon used for training.

2. Training loops are then started by running `python3 model/train.py {path to model cfg} {path to dataset from 1} {experiment name}`. The model cfg specifices the structure of the neural network and the experiment name specifies where the trained models will be stored (output/{model name}/{experiment name}). The flag `--log_wandb` can also be added to log metrics.
