"""
make_s2s_dataset.py
Create seq2seq JSONL pairs for FIR + Psych report generation from structured CSV.
"""
import json, pandas as pd, argparse

def build_input(row):
    signals = {k: row.get(k) for k in [
        'timeline_consistency_score','isolation_index','financial_control_index','digital_intrusion_index',
        'coercive_control_indicators','gaslighting_phrases_detected','threat_implicitness_score',
        'frequency_per_week','duration_months','prior_reports_count'
    ] if k in row}
    txt = (row.get('tip_summary_hints','') or '')[:400]
    return ("TIP_ID: {tip_id}\nVICTIM: {victim}\nCONTEXT: {ctx}\nRELATION: {rel}\n"
            "HINTS: {hints}\nTEXT: {txt}\nSIGNALS: {signals}".format(
                tip_id=row.get('tip_id','UNK'),
                victim=row.get('victim_name_synth','Anonymous'),
                ctx=row.get('incident_context','unknown'),
                rel=row.get('alleged_abuser_relation','unknown'),
                hints=row.get('tip_summary_hints',''),
                txt=txt,
                signals=str(signals)))

def build_target(row):
    return ("FIR: On {ts}, victim {victim} reported issues in context '{ctx}' involving relation '{rel}'. "
            "Allegations include: {hints}. The report indicates indicators of coercive control, digital intrusion, "
            "and isolation as applicable.\n"
            "PSYCH_REPORT: The account suggests elevated stress, potential anxiety from surveillance/financial control. "
            "Recommend immediate safety planning, device forensics, and counseling support.").format(
                ts=row.get('tip_time','date-unknown'), victim=row.get('victim_name_synth','Anonymous'),
                ctx=row.get('incident_context','unknown'), rel=row.get('alleged_abuser_relation','unknown'),
                hints=row.get('tip_summary_hints',''))

def main(args):
    df = pd.read_csv(args.data)
    recs = []
    for _, r in df.iterrows():
        inp = build_input(r)
        tgt = build_target(r)
        recs.append({'input': inp, 'target': tgt})
    with open(args.out, 'w') as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Wrote", len(recs), "records to", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='structured_anonymous_tips_dataset.csv')
    ap.add_argument('--out', required=True, help='output jsonl')
    args = ap.parse_args()
    main(args)
