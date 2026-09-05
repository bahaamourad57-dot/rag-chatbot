# Precision, Recall, and F1 Score

Precision and recall are metrics used to evaluate classification
models, especially when classes are imbalanced (accuracy alone can be
misleading in that case).

Precision answers: "Of everything the model predicted as positive,
how much was actually positive?"
Precision = True Positives / (True Positives + False Positives)

Recall answers: "Of everything that was actually positive, how much
did the model correctly identify?"
Recall = True Positives / (True Positives + False Negatives)

There is usually a trade-off between precision and recall: raising the
decision threshold tends to increase precision but decrease recall,
and lowering it does the opposite.

F1 score is the harmonic mean of precision and recall, useful when you
want a single number that balances both:
F1 = 2 * (Precision * Recall) / (Precision + Recall)

When to prioritize recall: situations where missing a positive case is
costly (e.g. disease screening, fraud detection).

When to prioritize precision: situations where a false alarm is costly
and positive predictions trigger significant action (e.g. spam filters
that delete email automatically).
