import cv2
import os
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

DATASET_PATH = "dataset"


def load_dataset():
    faces = []
    labels = []

    label_id = 0
    label_dict = {}

    for user in os.listdir(DATASET_PATH):

        user_path = os.path.join(DATASET_PATH, user)

        if not os.path.isdir(user_path):
            continue

        label_dict[label_id] = user

        for img_name in os.listdir(user_path):

            img_path = os.path.join(user_path, img_name)

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            img = cv2.resize(img, (200, 200))

            faces.append(img)
            labels.append(label_id)

        label_id += 1

    return faces, np.array(labels)


def evaluate_5fold():

    faces, labels = load_dataset()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    fold = 1

    for train_idx, test_idx in skf.split(faces, labels):

        X_train = [faces[i] for i in train_idx]
        X_test = [faces[i] for i in test_idx]

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.train(
            X_train,
            np.array(y_train)
        )

        y_pred = []

        for img in X_test:

            pred_label, confidence = recognizer.predict(img)

            y_pred.append(pred_label)

        acc = accuracy_score(y_test, y_pred)

        prec = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        rec = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        print(f"\n===== Fold {fold} =====")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

        fold += 1

    print("\n==============================")
    print("5-FOLD CROSS VALIDATION RESULT")
    print("==============================")

    print(f"Average Accuracy : {np.mean(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall   : {np.mean(recalls):.4f}")
    print(f"Average F1 Score : {np.mean(f1_scores):.4f}")


if __name__ == "__main__":
    evaluate_5fold()