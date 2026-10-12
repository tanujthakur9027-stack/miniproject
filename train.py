import csv
import json
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


DATASET_FILES = (
    "Bitext_Sample_Customer_Service_Training_Dataset.csv",
    "Bitext_Sample_Customer_Service_Validation_Dataset.csv",
    "Bitext_Sample_Customer_Service_Testing_Dataset.csv",
)
REQUIRED_COLUMNS = ("utterance", "intent", "category", "tags")
RANDOM_SEED = 42
TEST_SIZE = 0.2


def load_datasets(project_dir: Path) -> list[dict[str, str]]:
    rows = []
    for filename in DATASET_FILES:
        path = project_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
            reader = csv.DictReader(dataset_file)
            if reader.fieldnames is None or not set(REQUIRED_COLUMNS).issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"{filename} must contain these columns: "
                    f"{', '.join(REQUIRED_COLUMNS)}"
                )

            for row_number, row in enumerate(reader, start=2):
                utterance = (row["utterance"] or "").strip()
                intent = (row["intent"] or "").strip()
                if not utterance or not intent:
                    raise ValueError(
                        f"{filename} has a blank utterance or intent "
                        f"at row {row_number}"
                    )
                rows.append(
                    {
                        "utterance": utterance,
                        "intent": intent,
                        "category": (row["category"] or "").strip(),
                        "tags": (row["tags"] or "").strip(),
                    }
                )

    if not rows:
        raise ValueError("The input datasets contain no examples.")
    return rows


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    rows = load_datasets(project_dir)
    random.Random(RANDOM_SEED).shuffle(rows)

    combined_path = project_dir / "combined_dataset.csv"
    with combined_path.open("w", encoding="utf-8", newline="") as combined_file:
        writer = csv.DictWriter(combined_file, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    texts = [row["utterance"] for row in rows]
    labels = [row["intent"] for row in rows]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train_texts, train_labels)

    predictions = model.predict(test_texts)
    metrics = {
        "total_examples": len(rows),
        "training_examples": len(train_texts),
        "test_examples": len(test_texts),
        "intent_count": len(set(labels)),
        "accuracy": accuracy_score(test_labels, predictions),
        "classification_report": classification_report(
            test_labels, predictions, output_dict=True, zero_division=0
        ),
    }

    model_path = project_dir / "customer_service_intent_model.joblib"
    metrics_path = project_dir / "evaluation_metrics.json"
    joblib.dump(model, model_path)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Combined dataset: {combined_path} ({len(rows)} examples)")
    print(
        f"Training examples: {len(train_texts)} | "
        f"Test examples: {len(test_texts)}"
    )
    print(f"Intent classes: {len(set(labels))}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Trained model: {model_path}")
    print(f"Evaluation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
