# Overfitting and Underfitting

Overfitting happens when a model learns the training data too well,
including its noise and random fluctuations, so it performs great on
training data but poorly on new, unseen data. A classic sign of
overfitting is a large gap between training accuracy and validation
accuracy — for example, 99% training accuracy but 70% validation
accuracy.

Underfitting is the opposite problem: the model is too simple to
capture the underlying pattern in the data, so it performs poorly on
both training and validation data.

Common ways to reduce overfitting:
- Get more training data
- Use regularization (L1/L2 penalties, dropout in neural networks)
- Reduce model complexity (fewer features, shallower trees, fewer layers)
- Use cross-validation to tune hyperparameters instead of the test set
- Use early stopping during training

Common ways to reduce underfitting:
- Use a more complex model
- Add more relevant features
- Reduce regularization strength
- Train for longer
