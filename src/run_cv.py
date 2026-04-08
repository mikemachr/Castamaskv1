from statistics import mean, stdev

from family_map import FAMILY_NAMES
from train_one_fold import train_one_fold
from config import EXPERIMENTS_ROOT
from utils import save_json


def aggregate_fold_results(results):
    metric_names = [
        "loss",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "static_false_positive_rate",
        "dynamic_false_negative_rate",
    ]

    summary = {}

    for metric in metric_names:
        values = [float(r["test_metrics"][metric]) for r in results]
        summary[metric] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }

    return summary


def main():
    all_results = []
    num_folds = len(FAMILY_NAMES)

    print(f"Running grouped cross-validation over {num_folds} folds...\n")

    for fold_id in range(num_folds):
        result = train_one_fold(fold_id=fold_id, device="cpu")
        all_results.append(result)

    summary = aggregate_fold_results(all_results)

    cv_dir = EXPERIMENTS_ROOT / "cross_validation"
    cv_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "num_folds": num_folds,
            "fold_results": all_results,
            "summary": summary,
        },
        cv_dir / "cv_results.json",
    )

    print("\n=== Cross-validation summary ===")
    for metric, stats in summary.items():
        print(f"{metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")


if __name__ == "__main__":
    main()