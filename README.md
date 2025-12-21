# Classification as Regression + Argmax (1D Visualization)
## Overview
This project visualizes multi-class classification as a set of competing regression functions in logit space, using a minimal 1D neural network. While logits are not probabilities, they play a role analogous to log-odds: classification depends on their relative magnitudes rather than their absolute values.

## Visualization
<img width="578" height="455" alt="regression_of_logits" src="https://github.com/user-attachments/assets/98e7501e-4f6a-4802-83ae-ed1e9df69dc1" />

## Interpretation
Although classification results are discrete, the underlying model is fully continuous. The three logit regression functions represent how strongly the model favors assigning an input to each class across the input range. The apparent decision boundaries arise solely from the argmax operation applied to continuous logit regression functions.

### Loss and Optimization Choices

Mean Squared Error (MSE) is used instead of cross-entropy to keep the training objective aligned with a regression perspective. Each class logit is trained
as a continuous function over the input space, and classification emerges only through comparison between these functions via an argmax operation.

SGD with Nesterov momentum is chosen to preserve a transparent relationship between gradients and parameter updates. Unlike adaptive optimizers, this
allows the learned regression functions and their evolution during training to remain easier to interpret.

## Scope
This repository is not intended as a performance benchmark or a training tutorial. Its sole purpose is to make the geometric structure of multi-class classifiers explicit.

## Usage
```
$ python3 -m src.trainer
```
| Argument                |
|-------------------------|
|`--lr`                   |
|`--momentum`             |
|`--data_seed`            |
|`--epoches`              |
|`--save_checkpoints`     |
|`--save_checkpoint_dir`  |
|`--save_logits`          |
|`--save_logit_dir`       |
