# Smart Job Match Agent 🎯

An intelligent Job Recommendation API powered by **TF-IDF semantic ranking** + **Gemini 2.0 Flash** agentic tool-calling.

**Live Demo:** `https://your-project.vercel.app`

---

## Architecture

```
Resume Text
    │
    ├─► TF-IDF Cosine Similarity  →  Top-5 Jobs  (Classical ML)
    │
    └─► Gemini 2.0 Flash Agent
            │
            ├─ Tool Call 1: parse_resume         → structured candidate profile
            ├─ Tool Call 2: generate_match_explanations → per-job reasoning + question
            └─ /refine endpoint  → re-ranks based on candidate's clarifying answer
```

---

## Setup (5 commands)

```bash
git clone https://github.com/YOUR_USERNAME/smart-job-match.git
cd smart-job-match
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then add your GEMINI_API_KEY
```

Get a **free** Gemini API key at: https://aistudio.google.com/apikey

Run locally:
```bash
uvicorn api.index:app --reload
```

Visit: http://localhost:8000

---

## API Endpoints

### `POST /recommend`
```json
{
  "resume_text": "Python developer with 2 years NLP experience..."
}
```

### `POST /refine`
```json
{
  "resume_text": "...",
  "clarifying_question": "Are you open to remote work?",
  "candidate_answer": "Yes, I prefer fully remote roles"
}
```

### `GET /health` — system status
### `GET /jobs` — browse dataset (filter by `?domain=Tech&remote=true`)
### `GET /stats` — dataset statistics
### `GET /docs` — interactive Swagger UI

---

## Deploy to Vercel

```bash
npm i -g vercel
vercel login
vercel --prod
```

Add `GEMINI_API_KEY` in Vercel dashboard → Project → Settings → Environment Variables.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Embedding | TF-IDF (custom) | Zero latency, no API cost, good spread on 50 jobs |
| LLM | Gemini 2.0 Flash | Free tier, fast, native tool-calling |
| API | FastAPI | Async, auto-docs, production-ready |
| Deploy | Vercel | Free, zero-config Python deployment |

---

## Project Structure

```
smart-job-match/
├── api/
│   ├── __init__.py
│   └── index.py        ← FastAPI app (all logic)
├── jobs.json           ← 50-job dataset
├── index.html          ← Web UI (served at /)
├── main.py             ← uvicorn entry point
├── requirements.txt
├── vercel.json
├── .env.example
└── WRITEUP.md
```
