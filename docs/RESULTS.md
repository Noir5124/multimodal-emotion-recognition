# Results

## Shared Label Space

```text
angry, fear, happy, sad, surprise
```

## High-Level Metrics

| Model / Split | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FER Validation | 0.9111 | 0.8953 | 0.8428 | 0.8953 | 0.8632 | 0.9178 | 0.9111 | 0.9133 |
| FER Test | 0.8896 | 0.8592 | 0.8199 | 0.8592 | 0.8367 | 0.8975 | 0.8896 | 0.8922 |
| Text Validation | 0.9374 | 0.9268 | 0.8937 | 0.9268 | 0.9057 | 0.9414 | 0.9374 | 0.9379 |
| Text Test | 0.9321 | 0.9242 | 0.8748 | 0.9242 | 0.8893 | 0.9413 | 0.9321 | 0.9339 |

## Visual Evaluation

### Snapshot

![Evaluation Snapshot](../evaluation_graphs/evaluation_snapshot.png)

### Summary Metrics

![Summary Metrics Grouped](../evaluation_graphs/summary_metrics_grouped.png)

![Summary Metrics Heatmap](../evaluation_graphs/summary_metrics_heatmap.png)

### Per-Class Metrics

![Per Class Metrics](../evaluation_graphs/per_class_metrics.png)

### Confusion Matrices

![Confusion Matrices](../evaluation_graphs/confusion_matrices.png)

### Training Curves

![Training Curves](../evaluation_graphs/training_curves.png)

## Quick Reading

- text model performs stronger than facial model on both validation and test splits
- FER model weakest class is `fear`
- text model shows best overall balance across 5 classes
- both models maintain strong weighted F1, meaning common classes are predicted well
- FER confusion matrix shows overlap between `fear`, `sad`, and `surprise`

## Source Files

Metrics came from exported artifact folder:

```text
D:\emotion_model_artifacts
```

Examples:

- `fer_validation_summary_metrics.csv`
- `fer_test_summary_metrics.csv`
- `text_validation_summary_metrics.csv`
- `text_test_summary_metrics.csv`
- `fer_training_history.csv`
- `text_training_history.csv`
