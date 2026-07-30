import numpy as np
import pytest

from location_classifier import evaluate

TRUE = ["a", "a", "b", "b", "c", "c"]
PRED = ["a", "b", "b", "b", "c", "a"]


def test_perfect_predictions_score_one():
    metrics = evaluate.evaluate_predictions(TRUE, TRUE)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["cohen_kappa"] == pytest.approx(1.0)


def test_metrics_report_counts_and_labels():
    metrics = evaluate.evaluate_predictions(TRUE, PRED)
    assert metrics["n_samples"] == 6
    assert metrics["n_classes"] == 3
    assert metrics["labels"] == ["a", "b", "c"]
    assert metrics["accuracy"] == pytest.approx(4 / 6)


def test_confusion_matrix_rows_sum_to_support():
    metrics = evaluate.evaluate_predictions(TRUE, PRED)
    frame = evaluate.confusion_frame(metrics)
    assert frame.shape == (3, 3)
    assert list(frame.sum(axis=1)) == [2, 2, 2]


def test_confusion_frame_requires_the_matrix():
    with pytest.raises(KeyError):
        evaluate.confusion_frame({"labels": ["a"]})


def test_per_class_frame_is_sorted_by_f1():
    metrics = evaluate.evaluate_predictions(TRUE, PRED)
    frame = evaluate.per_class_frame(metrics)
    assert list(frame.index) == sorted(frame.index, key=lambda n: frame.loc[n, "f1-score"])
    assert set(frame.index) == {"a", "b", "c"}


def test_worst_confusions_are_ranked():
    true = ["a"] * 10 + ["b"] * 4 + ["c"] * 2
    pred = ["b"] * 6 + ["a"] * 4 + ["c"] * 2 + ["b"] * 2 + ["c"] * 2
    pairs = evaluate.worst_confusions(evaluate.evaluate_predictions(true, pred))
    assert pairs[0]["true"] == "a" and pairs[0]["predicted"] == "b"
    assert pairs[0]["count"] >= pairs[-1]["count"]


def test_top_k_accuracy_is_added_when_probabilities_are_given():
    classes = ["a", "b", "c"]
    proba = np.array([[0.4, 0.5, 0.1], [0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
    metrics = evaluate.evaluate_predictions(
        ["a", "c", "b"], ["b", "c", "b"], y_proba=proba, classes=classes
    )
    assert metrics["top_2_accuracy"] == pytest.approx(1.0)


def test_probability_shape_is_checked():
    with pytest.raises(ValueError, match="y_proba shape"):
        evaluate.evaluate_predictions(
            ["a", "b"], ["a", "b"], y_proba=np.zeros((2, 5)), classes=["a", "b", "c"]
        )


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        evaluate.evaluate_predictions(["a", "b"], ["a"])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        evaluate.evaluate_predictions([], [])


def test_format_metrics_contains_headline_numbers():
    text = evaluate.format_metrics(evaluate.evaluate_predictions(TRUE, PRED), title="Test run")
    assert "Test run" in text
    assert "accuracy" in text
    assert "per class" in text
    assert "most confused pairs" in text
