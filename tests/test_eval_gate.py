import json
import os

MIN_ACCURACY = 0.65
MIN_F1 = 0.63

def test_model_meets_quality_bar():
    assert os.path.exists("metrics.json"), "metrics.json not found — run training first"
    with open("metrics.json") as f:
        metrics = json.load(f)

    acc_key = next(k for k in metrics if "accuracy" in k)
    f1_key = next(k for k in metrics if "f1" in k)

    assert metrics[acc_key] >= MIN_ACCURACY, (
        f"Accuracy {metrics[acc_key]:.3f} below threshold {MIN_ACCURACY}"
    )
    assert metrics[f1_key] >= MIN_F1, (
        f"F1 {metrics[f1_key]:.3f} below threshold {MIN_F1}"
    )