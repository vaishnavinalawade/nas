"""
FastAPI service for:
 - /analyze (AlertEngine analysis)
 - /classify (predict alert_priority using saved model)
 - /generate_reports (LLM/T5 text generation via template; placeholder)
 - /log (append to CSV log)

Run:
    uvicorn main:app --reload --port 8080
"""
import os, json, joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentiment_risk_alert_pipeline import AlertEngine

MODEL_PATH = os.environ.get("ALERT_MODEL_PATH", "./model_out/alert_priority_model.joblib")
LOG_PATH = os.environ.get("ALERT_LOG_PATH", "./service_log.csv")

app = FastAPI(title="National Alert System API", version="1.0.0")
engine = AlertEngine()

class AnalyzeReq(BaseModel):
    text: str
    hints: str
    victim_name: str = "Friend"
    channel: str = "sms"

class ClassifyReq(BaseModel):
    text_field: str
    numeric_array: list

class ReportReq(BaseModel):
    case_record: dict  # structured case dict

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    record = engine.analyze_tip(req.text, req.hints)
    msg = engine.generate_alert(record, req.channel, req.victim_name)
    return {"record": record, "alert_message": msg}

@app.post("/classify")
def classify(req: ClassifyReq):
    if not os.path.exists(MODEL_PATH):
        return {"error": f"Model not found at {MODEL_PATH}"}
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([{"text_field": req.text_field, "numeric_array": req.numeric_array}])
    pred = model.predict(X)[0]
    proba = getattr(model.named_steps['clf'], "predict_proba", lambda z: None)
    return {"prediction": pred}

@app.post("/generate_reports")
def generate_reports(req: ReportReq):
    # Placeholder template generator (replace with T5 or hosted LLM)
    case = req.case_record
    fir = (f"On {case.get('timestamp','date-unknown')}, victim {case.get('victim_name','Anonymous')} "
           f"reported concerns in context '{case.get('incident_context','unknown')}'. "
           f"Alleged relation: {case.get('alleged_abuser_relation','unknown')}. "
           f"Hints: {case.get('tip_summary_hints','')}. This is a draft for human review.")
    psy = ("Initial psychological summary: reported stress indicators and possible coercive control. "
           "Recommend safety planning, counseling, and evidence preservation.")
    return {"FIR_summary": fir, "Psych_report": psy, "recommended_actions": ["safety-plan","counseling","forensic-review"]}

@app.post("/log")
def log_json(payload: dict):
    df = pd.DataFrame([payload])
    if os.path.exists(LOG_PATH):
        df0 = pd.read_csv(LOG_PATH); df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return {"ok": True, "rows": len(df)}
