---
title: DocuMind AI API
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.11"
app_file: space_app.py
pinned: false
short_description: RAG backend for cited PDF question answering
preload_from_hub:
  - sentence-transformers/all-MiniLM-L6-v2
---

# DocuMind AI API

Backend for [DocuMind AI](https://github.com/mahak-bit/ai-pdf). This Space is assembled from
`backend/` plus the files in `deploy/hf-space/` by `deploy/hf-space/build.sh`; edit the
source repository, not this Space.

Required Space settings: secret `GEMINI_API_KEY`; variables `GEMINI_MODEL` and `CORS_ORIGINS`.
