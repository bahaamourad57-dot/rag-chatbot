# Cross-Validation

Cross-validation is a technique for estimating how well a model will
generalize to unseen data, without needing a separate holdout set for
every experiment.

The most common form is k-fold cross-validation:
1. Split the training data into k equal-sized folds (commonly k=5 or k=10)
2. For each fold, train the model on the other k-1 folds and evaluate
   on the held-out fold
3. Average the k evaluation scores to get a single performance estimate

Stratified k-fold cross-validation preserves the class distribution in
each fold, which matters for classification tasks with imbalanced
classes — without stratification, a fold might end up with very few
examples of the minority class.

Why use cross-validation instead of a single train/test split:
- A single split gives one noisy estimate; averaging over k folds
  gives a more reliable estimate and also an estimate of variance
  (the standard deviation across folds)
- It makes better use of limited data, since every example is used for
  both training and validation at some point

Cross-validation is typically used during model selection and
hyperparameter tuning. The final model is usually retrained on the
full training set once the best configuration is chosen, and a
separate untouched test set is reserved for final evaluation.
