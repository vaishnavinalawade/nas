# National Alert System — Deliverables Bundle

This bundle contains:
- Core analysis module: `sentiment_risk_alert_pipeline.py`
- Classifier training: `train_alert_priority.py`
- Seq2Seq dataset prep: `make_s2s_dataset.py`
- T5 fine-tuning script: `finetune_t5_fir.py`
- FastAPI service: `main.py`
- Jupyter notebook: `train_classifier_notebook.ipynb`
- requirements.txt

## Quick Start
1) `pip install -r requirements.txt`
2) Train classifier:  
   `python train_alert_priority.py --data structured_anonymous_tips_dataset.csv --outdir model_out`
3) Start API:  
   `export ALERT_MODEL_PATH=model_out/alert_priority_model.joblib`  
   `uvicorn main:app --reload --port 8080`
4) Prepare T5 training pairs:  
   `python make_s2s_dataset.py --data structured_anonymous_tips_dataset.csv --out train.jsonl`
5) Fine-tune T5 (optional):  
   `python finetune_t5_fir.py --train train.jsonl --outdir t5_fir_model`

## Notebook
Open `train_classifier_notebook.ipynb` and run all cells to train/evaluate quickly.
