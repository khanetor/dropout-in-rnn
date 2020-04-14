# Dropout as Bayesian Approximation in RNN
##### Master thesis - Aalto University - Machine Learning and Data Science

## Dataset
- [Penn Tree Bank](https://www.kaggle.com/nltkdata/penn-tree-bank/data)
- Cornell film reviews: sentence polarity
- Apple Stock from Yahoo
- [Occupancy prediction](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)


## TODOs
- [x] Add biases
- [x] Make loss function
- [x] Stack multiple LSTM layers
- [x] Handle single dimension input
- [x] Iterate Stochastic pass once during training
- [x] Use parameter optimizer instead of writing custom loss functions
- [x] Make stochastic modules for regression
- [x] Make stochastic modules for classification
- [x] Compute predictive variance for classification
- [x] Compute predictive variance for regression
- [x] Optimize dropout rate (using Concrete Dropout)
- [x] Download clinical data
- [x] Early stop in training (use validation error)
- [x] Make equivalent in Tensorflow
