# 🛡️ National Alert System (NAS)
### **AI-Assisted Early Warning System for Abuse, Harassment & Digital Intrusion**

The **National Alert System** is an offline-first, privacy-centric AI pipeline designed to help NGOs, psychologists, and legal responders detect patterns of *psychological abuse*, *coercive control*, *gaslighting*, *workplace harassment*, and *digital stalking* from anonymous citizen tips.

It converts unstructured messages into:
- 📊 Behavioral risk indicators  
- 🧠 Sentiment & coercive-control signals  
- 🔥 Composite risk scores (0–100)  
- 🛎️ Immediate safety alerts (SMS/Email/WhatsApp)  
- 📝 FIR-ready summaries and psychologist-ready assessments (LLM-driven)  
- 🔗 Ledger-ready case records for evidence chains  

This project is built to empower victims, assist human experts, and strengthen trust in justice systems — while maintaining strict ethical and privacy guarantees.

---

## 🚀 Key Features
- **Offline deterministic NLP engine** (`sentiment_risk_alert_pipeline.py`)  
- **Structured tip analysis** (`AlertEngine`)  
- **Alert priority classifier** (TF-IDF + numeric signals → ML model)  
- **Seq2Seq FIR/Psych generator** (T5 fine-tuning)  
- **FastAPI microservice** (deployable to cloud or NGO infrastructure)  
- **Jupyter notebook for training & evaluation**  
- **Evidence-safe outputs** (non-accusatory, factual, human-supervised)  

---

## 🔐 Ethics & Safety
NAS is designed with strict safeguards:
- Human-in-the-loop REQUIRED  
- No automatic accusations; no auto-filing FIRs  
- Privacy-first architecture  
- Ledger-ready case trail for accountability  
- Cross-verification, red-teaming, bias audits  

**This system does *not* replace police or psychologists — it amplifies them.**

---

## 📦 Deliverables in This Repository
- Core NLP engine  
- ML training scripts  
- Seq2Seq dataset builder  
- T5 fine-tuning script  
- FastAPI server (`main.py`)  
- Example datasets  

---

## 🤝 Supported By
Community volunteers, data scientists, trauma-response NGOs, justice advocates, and ethical technologists committed to building a safer society.

> _“Justice begins the moment a cry for help is truly heard.”_


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
