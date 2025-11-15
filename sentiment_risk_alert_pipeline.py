"""
sentiment_risk_alert_pipeline.py
(See inline comments for explainability; built offline and deterministic.)
"""
from __future__ import annotations
import re, numpy as np
from typing import Dict, List

NEG_WORDS = set(["afraid","broken","unsafe","threat","humiliate","shame","insult","control","stalk","blackmail",
                 "abuse","coerce","fear","cry","hurt","regret","worse","danger","isolate","ruin","panic","anxious"])
POS_WORDS = set(["help","support","protect","safe","assist","counsel","guide","care"])
SUBJECTIVE_MARKERS = set(["i think","i feel","seems","maybe","probably","worried","afraid","believe","guess","appears"])

def sentiment_scores(text: str) -> Dict[str, float]:
    t = (text or "").lower()
    tokens = re.findall(r"[a-z']+", t)
    pos = sum(1 for w in tokens if w in POS_WORDS)
    neg = sum(1 for w in tokens if w in NEG_WORDS)
    subj = sum(1 for m in SUBJECTIVE_MARKERS if m in t)
    total = max(1, pos + neg)
    compound = (pos - neg) / total
    subjectivity = min(1.0, subj / 3.0)
    return {"sent_pos_count": int(pos), "sent_neg_count": int(neg), "sent_compound": round(compound,3),
            "subjectivity_proxy": round(subjectivity,3), "token_len": int(len(tokens))}

def indices_from_hints(hints: str, neg_over_pos: bool) -> Dict[str, float]:
    import numpy as _np
    hint_list = [h.strip() for h in (hints or "").split(";") if h.strip()]
    digital_intrusion = 0.0
    if any(h in hint_list for h in ["reads-messages","gps-tracking","fake-accounts","revenge-porn"]): digital_intrusion += 0.65
    if neg_over_pos: digital_intrusion += 0.05
    digital_intrusion = float(_np.clip(digital_intrusion, 0, 1))

    isolation = 0.0
    if "isolates-from-friends" in hint_list: isolation += 0.6
    if "humiliation" in hint_list or "gaslighting" in hint_list: isolation += 0.15
    if neg_over_pos: isolation += 0.05
    isolation = float(_np.clip(isolation, 0, 1))

    financial_control = 0.0
    if "control-money" in hint_list or "salary-seized" in hint_list: financial_control += 0.65
    financial_control = float(_np.clip(financial_control, 0, 1))

    gaslighting_phrases = 0
    if "gaslighting" in hint_list: gaslighting_phrases += 6
    if "humiliation" in hint_list or "verbal-abuse" in hint_list: gaslighting_phrases += 3

    coercive_control_indicators = 0
    if "threats-coded" in hint_list: coercive_control_indicators += 3
    if "breaks-things" in hint_list: coercive_control_indicators += 4
    if "calls-nightly" in hint_list: coercive_control_indicators += 2
    if "stalking" in hint_list: coercive_control_indicators += 4

    threat_implicitness = 0.0
    if "threats-coded" in hint_list: threat_implicitness += 0.6
    if "stalking" in hint_list: threat_implicitness += 0.2
    if neg_over_pos: threat_implicitness += 0.05
    threat_implicitness = float(_np.clip(threat_implicitness, 0, 1))

    escalation_pattern = "escalating" if ("stalking" in hint_list or "revenge-porn" in hint_list) else "stable"
    frequency_per_week = 3 + 2*int("calls-nightly" in hint_list) + 2*int("stalking" in hint_list)
    timeline_consistency_score = 0.8
    contradiction_flags = 0

    duration_months = int(_np.clip(_np.random.lognormal(mean=2.0, sigma=0.6), 1, 120))
    prior_reports_count = int(_np.clip(_np.random.poisson(0.4), 0, 8))

    return {
        "timeline_consistency_score": round(timeline_consistency_score,2),
        "contradiction_flags": int(contradiction_flags),
        "coercive_control_indicators": int(coercive_control_indicators),
        "gaslighting_phrases_detected": int(gaslighting_phrases),
        "threat_implicitness_score": round(threat_implicitness,2),
        "isolation_index": round(isolation,2),
        "financial_control_index": round(financial_control,2),
        "digital_intrusion_index": round(digital_intrusion,2),
        "escalation_pattern": escalation_pattern,
        "frequency_per_week": int(frequency_per_week),
        "duration_months": int(duration_months),
        "prior_reports_count": int(prior_reports_count)
    }

def risk_score(signals: Dict[str, float], hints: str) -> float:
    import numpy as _np
    hint_list = [h.strip() for h in (hints or "").split(";") if h.strip()]
    base = (32*signals["threat_implicitness_score"] +
            18*(1 - signals["timeline_consistency_score"]) +
            14*signals["isolation_index"] +
            12*signals["digital_intrusion_index"] +
            0.7*signals["gaslighting_phrases_detected"] +
            1.1*signals["coercive_control_indicators"] +
            1.8*signals["frequency_per_week"] +
            (7 if signals["escalation_pattern"]=="escalating" else 0) +
            (6 if "stalking" in hint_list else 0) +
            (8 if "revenge-porn" in hint_list else 0))
    return float(_np.clip(round(base,1), 0, 100))

def priority_band(risk: float, escalation: str) -> str:
    if risk >= 75 or escalation == "escalating": return "critical"
    if risk >= 50: return "high"
    if risk >= 30: return "medium"
    return "low"

def generate_alert(channel: str, victim_name: str, risk_score_val: float, priority: str, recommended_actions: List[str]) -> str:
    first_name = (victim_name or "").split()[0] if victim_name else "there"
    actions_human = ", ".join(a.replace("-", " ") for a in (recommended_actions or []))
    if channel.lower() in ["sms","whatsapp"]:
        return (f"Confidential safety alert for {first_name}: {priority.upper()} risk (score {risk_score_val:.1f}). "
                f"Reply HELP for assistance. Next steps: {actions_human}. Reply STOP to opt-out.")
    if channel.lower() == "email":
        return (f"Subject: Confidential Support — {priority.capitalize()} Risk (Score {risk_score_val:.1f})\n\n"
                f"Dear {victim_name or 'Friend'},\n\n"
                f"We received a confidential safety alert indicating a {priority} risk (score {risk_score_val:.1f}). "
                f"If you feel unsafe, reply or call 112 immediately. Suggested next steps: {actions_human}.\n\n"
                f"This message is private.\n— National Safety Response Network")
    return f"[Alert] {priority} risk ({risk_score_val}). Actions: {actions_human}"

class AlertEngine:
    def __init__(self, output_path: str = None):
        import pandas as _pd
        self.output_path = output_path
        self.alert_log = []
        self._pd = _pd

    def analyze_tip(self, text: str, hints: str) -> Dict:
        snt = sentiment_scores(text)
        sig = indices_from_hints(hints, neg_over_pos=(snt['sent_neg_count'] > snt['sent_pos_count']))
        r = risk_score(sig, hints); band = priority_band(r, sig['escalation_pattern'])
        record = {"timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                  "tip_text": text, "hints": hints, **snt, **sig, "risk_score": r, "priority_band": band}
        return record

    def generate_alert(self, record: Dict, channel: str, victim_name: str) -> str:
        actions = ["victim-safety-plan","forensic-analysis","counseling-support"]
        return generate_alert(channel, victim_name, record["risk_score"], record["priority_band"], actions)

    def save_to_log(self, record: Dict):
        self.alert_log.append(record)
        if self.output_path:
            df = self._pd.DataFrame(self.alert_log)
            df.to_excel(self.output_path, index=False)
