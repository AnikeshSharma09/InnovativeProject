"""
Diagnostic script to inspect current classification errors, confusion matrix,
and pinpoint exact reasons for misclassifications across RF and LSTM models.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    rf = joblib.load('models/sign_model.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    X_test = pd.read_csv('data/X_test.csv').values
    y_test = pd.read_csv('data/y_test.csv')['label'].astype(str).values
    preds = encoder.inverse_transform(rf.predict(X_test))

    report = classification_report(y_test, preds, output_dict=True)
    print("=" * 60)
    print(f"RANDOM FOREST OVERALL TEST ACCURACY: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"MACRO F1-SCORE: {report['macro avg']['f1-score']:.4f}")
    print("=" * 60)

    # Classes with F1 < 0.95
    weak = []
    for k, v in report.items():
        if isinstance(v, dict) and 'f1-score' in v and k not in ('macro avg', 'weighted avg'):
            if v['f1-score'] < 0.95:
                weak.append((k, v['f1-score'], v['precision'], v['recall'], v['support']))
    weak.sort(key=lambda x: x[1])
    print("\nClasses needing accuracy boost (F1 < 0.95):")
    for w in weak:
        print(f"  Class '{w[0]}': F1={w[1]:.3f}, Precision={w[2]:.3f}, Recall={w[3]:.3f}, Support={w[4]}")

    # Top confusion pairs
    classes = list(encoder.classes_)
    cm = confusion_matrix(y_test, preds, labels=classes)
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusions.append((classes[i], classes[j], cm[i, j]))
    confusions.sort(key=lambda x: x[2], reverse=True)
    print("\nTop Misclassification Confusion Pairs (Actual -> Predicted):")
    for c in confusions[:20]:
        print(f"  Actual '{c[0]}' was misclassified as '{c[1]}': {c[2]} times")

if __name__ == '__main__':
    main()
