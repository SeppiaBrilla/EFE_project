# Regression Experiment

As suggested by Nguyen, I tried to train the neural network to predict the runtime of each algorithm in order to choose the best one. 
I trained a network similar to the one we have used for the CNN and BNN models: BERT -> feature layer (100 neurons) + tanh -> post features layer (200 neurons) + ReLu -> Output layer (one neuron per algorithm). The only big change is the final activation which is a ReLu instead of Softmax or Sigmoid. To avoid having huge values like 3600, I converted everything into log scale using the formula: $log(1 + x)$ ([nunpy reference](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html)).

### Loss function
The loss function I used is a modified version of the MSE loss to also take into account the difference between the predicted value and the virtual best. The final loss is computed as:
$$
\text{Loss}(y, \hat{y}) = \frac{1}{N} \sum (\hat{y} - y) ^ 2 \times (\hat{y} - \min(y))
$$
where $y$ and $\hat{y}$ are. respectively, the true and predicted values, $\min(y)$ is the virtual best of a given instance and $\times$ is the dot product between two vectors.

## Results
I have trained 4 folds of the covering array problem class. I've used a learning rate of $10^{-6}$ and trained the networks for 2300 epochs.
The final loss scores were extremely good.
![loss](loss.pdf)
Most of the folds reach a validation loss of below 1: (0: 0.482, 1: 1.093, 2: 0.557, 3: 0.675). We also see some kind of [Grokking](https://arxiv.org/abs/2405.19454) and the best results on the validation set are reached well after the trailng loss has stabilized.

Even though the training seems to have produced very good results, while using the models to predict the algorithm, the results are less than ideal. I have tried compute the PAR10 scores on the regression models (RNN) and the regression models used as feature extractor paired with Kmeans (K, rNN). 
![results](./results.pdf)