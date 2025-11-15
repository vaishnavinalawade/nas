"""
train_alert_priority.py
Baseline classifier to predict alert_priority from structured_anonymous_tips_dataset.csv
"""
import argparse, os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_and_prepare(path):
    df = pd.read_csv(path)
    df['text_field'] = (df.get('tip_summary_hints','').fillna('') + ' ' +
                        df.get('incident_context','').fillna('') + ' ' +
                        df.get('alleged_abuser_relation','').fillna(''))
    df['alert_priority'] = df['alert_priority'].fillna('low').str.lower()
    numeric_cols = [c for c in [
        'timeline_consistency_score','isolation_index','financial_control_index',
        'digital_intrusion_index','coercive_control_indicators','gaslighting_phrases_detected',
        'threat_implicitness_score','frequency_per_week','duration_months','prior_reports_count'
    ] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].fillna(0.0).astype(float)
    df['numeric_array'] = df[numeric_cols].values.tolist()
    return df, numeric_cols

def main(args):
    df, numeric_cols = load_and_prepare(args.data)
    df = df[df['alert_priority'].notnull()].copy()
    X = df[['text_field','numeric_array']]; y = df['alert_priority']

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    text_pipe = Pipeline([
        ('extract', FunctionTransformer(lambda d: d['text_field'].astype(str), validate=False)),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)))
    ])
    num_pipe = Pipeline([
        ('extract', FunctionTransformer(lambda d: np.vstack(d['numeric_array']).astype(float), validate=False))
    ])
    feats = FeatureUnion([('text', text_pipe), ('num', num_pipe)])
    model = Pipeline([('features', feats), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])

    print("Training..."); model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    print("Accuracy:", accuracy_score(yte, preds))
    print(classification_report(yte, preds))
    print("Confusion matrix:\n", confusion_matrix(yte, preds, labels=['low','medium','high','critical']))

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(model, os.path.join(args.outdir, 'alert_priority_model.joblib'))
    with open(os.path.join(args.outdir, 'metadata.json'), 'w') as f:
        json.dump({'numeric_cols': numeric_cols}, f)
    print("Saved model in", args.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', default='./model_out')
    args = ap.parse_args()
    main(args)
