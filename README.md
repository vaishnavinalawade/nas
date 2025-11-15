# nas
National Alert System — AI for Justice, Safety & Early Intervention

A privacy-first, offline-capable AI pipeline designed to detect psychological abuse, coercive control, digital intrusion, workplace harassment, and high-risk patterns from anonymous citizen tips.
This system transforms unstructured messages into structured signals, computes risk scores, generates alerts, and drafts FIR-style summaries and psychologist-ready reports — always with a human-in-the-loop.

🔍 Features

Deterministic NLP Engine: Sentiment analysis, behavioral indices, and composite risk scoring

Structured Tip Analyzer (AlertEngine): Converts any tip into a full risk + evidence profile

Priority Classifier: Machine learning model to predict alert levels (low → critical)

Seq2Seq FIR Generator: T5-based template model to create FIR drafts + psych summaries

FastAPI Microservice: /analyze, /classify, /generate_reports, /log endpoints

Evidence Governance: Ledger-ready outputs, safe templates, and audit-friendly structures

NGO & Police Friendly: Simple workflow, explainable outputs, and safety-aligned design

📦 Included Components

Reusable Python modules

Training scripts & dataset builders

FastAPI server for deployment

Jupyter notebook (training & evaluation)

Slide deck for non-technical stakeholders

ZIP bundle with all deliverables

🧠 Purpose

To support NGOs, psychologists, legal advocates, and safety responders by identifying hidden abuse patterns early — especially psychological abuse, gaslighting, stalking, digital intrusion, and workplace harassment — and ensuring that victims are never left unheard.

⚖️ Ethics & Safety

This project is strictly assistive, non-accusatory, and requires mandatory human review.
All data handling follows privacy, consent, and evidence-chain principles.
