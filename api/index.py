"""
Smart Job Match Agent  ·  api/index.py
=======================================
Pipeline:
  1. TF-IDF cosine similarity  →  top-5 job candidates  (Classical ML)
  2. Gemini tool-call Step 1   →  parse_resume           (structured extraction)
  3. Gemini tool-call Step 2   →  generate_match_explanations (reasoning + question)

LLM  : Gemini 2.0 Flash  (free tier, via REST)
Embed: TF-IDF (zero latency, no external API, good spread on 50 jobs)
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

# Load .env from project root (works both locally and is safely ignored on Vercel)
load_dotenv(Path(__file__).parent.parent / ".env")

# ─── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Job Match Agent",
    description="TF-IDF ranking + Gemini 2.0 Flash agentic tool-calling",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ─── Config ─────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
TOP_N          = 5

# ─── Load jobs at startup ────────────────────────────────────────────────────

BASE       = Path(__file__).parent.parent
_jobs_path = BASE / "jobs.json"
if not _jobs_path.exists():
    _jobs_path = BASE / "job_dataset.json"

with open(_jobs_path, encoding="utf-8") as _f:
    JOBS: list[dict] = json.load(_f)

# ─── TF-IDF engine ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    tokens = text.split()
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def _job_text(job: dict) -> str:
    skills = " ".join(job.get("skills", []))
    # repeat skills 3x — strongest signal
    return (
        f"{job['title']} {job['title']} "
        f"{job['description']} "
        f"{skills} {skills} {skills} "
        f"{job['domain']} {job.get('location', '')}"
    )


def _build_index(jobs: list[dict]):
    docs = [_tokenize(_job_text(j)) for j in jobs]
    N    = len(docs)
    df: Counter = Counter()
    for doc in docs:
        df.update(set(doc))
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    vecs = []
    for doc in docs:
        tf   = Counter(doc)
        vec  = {t: (tf[t] / len(doc)) * idf[t] for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vecs.append({t: v / norm for t, v in vec.items()})
    return vecs, idf


JOB_VECTORS, IDF_MAP = _build_index(JOBS)


def _query_vec(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    tf     = Counter(tokens)
    vec    = {t: (tf[t] / len(tokens)) * IDF_MAP.get(t, 0.0) for t in tf}
    norm   = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


def _cosine(a: dict, b: dict) -> float:
    return sum(a[t] * b[t] for t in set(a) & set(b))


def rank_jobs(resume_text: str, n: int = TOP_N) -> list[dict]:
    qvec   = _query_vec(resume_text)
    scored = [(i, _cosine(qvec, jv)) for i, jv in enumerate(JOB_VECTORS)]
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for i, score in scored[:n]:
        job = JOBS[i].copy()
        job["similarity_score"] = round(score, 4)
        results.append(job)
    return results


# ─── Gemini helpers ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "parse_resume",
        "description": (
            "Extract a structured candidate profile from raw resume text. "
            "Return only the specified fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name":             {"type": "string",  "description": "Full name of the candidate"},
                "skills":           {"type": "array", "items": {"type": "string"}, "description": "Technical and domain skills"},
                "experience_years": {"type": "number",  "description": "Years of professional experience (0 for freshers)"},
                "preferred_roles":  {"type": "array", "items": {"type": "string"}, "description": "Target job roles or domains"},
                "education":        {"type": "string",  "description": "Highest degree and institution"},
            },
            "required": ["name", "skills", "experience_years", "preferred_roles", "education"],
        },
    },
    {
        "name": "generate_match_explanations",
        "description": (
            "Given a candidate profile and top-5 jobs, produce a 2-3 sentence explanation "
            "for each job and one smart clarifying question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "explanations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "job_id":      {"type": "integer"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["job_id", "explanation"],
                    },
                },
                "clarifying_question": {
                    "type": "string",
                    "description": "One specific question to resolve an ambiguity in the resume-job match",
                },
            },
            "required": ["explanations", "clarifying_question"],
        },
    },
]


async def _gemini(messages: list[dict], tools: list[dict] | None = None) -> dict:
    payload: dict[str, Any] = {
        "contents":         messages,
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
    }
    if tools:
        payload["tools"]      = [{"functionDeclarations": tools}]
        payload["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, json=payload)

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error {r.status_code}: {r.text[:400]}"
        )
    return r.json()


def _tool_call(resp: dict) -> tuple[str, dict] | None:
    try:
        for part in resp["candidates"][0]["content"]["parts"]:
            if "functionCall" in part:
                fc = part["functionCall"]
                return fc["name"], fc.get("args", {})
    except (KeyError, IndexError):
        pass
    return None


def _text(resp: dict) -> str:
    try:
        return " ".join(
            p.get("text", "")
            for p in resp["candidates"][0]["content"]["parts"]
        ).strip()
    except (KeyError, IndexError):
        return ""


# ─── Two-step agent ──────────────────────────────────────────────────────────

async def run_agent(
    resume_text: str, top_jobs: list[dict]
) -> tuple[dict, list[dict], str]:

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your .env file."
        )

    # ── STEP 1 — parse_resume ────────────────────────────────────────────────
    r1 = await _gemini(
        messages=[{
            "role": "user",
            "parts": [{"text": (
                "You are an expert resume parser.\n\n"
                f"Resume:\n{resume_text}\n\n"
                "Call the parse_resume tool with the extracted candidate profile."
            )}],
        }],
        tools=[TOOLS[0]],
    )
    tc1 = _tool_call(r1)
    if not tc1 or tc1[0] != "parse_resume":
        raise HTTPException(
            status_code=502,
            detail="Agent did not invoke parse_resume tool. Please try again."
        )
    candidate = tc1[1]

    # ── STEP 2 — generate_match_explanations ─────────────────────────────────
    jobs_summary = json.dumps(
        [{
            "id":               j["id"],
            "title":            j["title"],
            "company":          j["company"],
            "domain":           j["domain"],
            "skills":           j.get("skills", []),
            "experience_years": j.get("experience_years"),
            "remote":           j.get("remote"),
            "salary_lpa":       j.get("salary_lpa"),
            "similarity_score": j["similarity_score"],
            "description":      j["description"][:220],
        } for j in top_jobs],
        indent=2,
    )

    r2 = await _gemini(
        messages=[{
            "role": "user",
            "parts": [{"text": (
                "You are an expert job match analyst.\n\n"
                f"Candidate profile:\n{json.dumps(candidate, indent=2)}\n\n"
                f"Top-5 matched jobs (TF-IDF cosine scores):\n{jobs_summary}\n\n"
                "Call generate_match_explanations. For each job write 2-3 sentences covering: "
                "(a) strength of skill alignment, (b) any experience-level gap, "
                "(c) domain fit. Then generate ONE smart clarifying question that resolves a "
                "specific ambiguity — never ask something generic."
            )}],
        }],
        tools=[TOOLS[1]],
    )
    tc2 = _tool_call(r2)
    if not tc2 or tc2[0] != "generate_match_explanations":
        raise HTTPException(
            status_code=502,
            detail="Agent did not invoke generate_match_explanations tool. Please try again."
        )
    expl_data = tc2[1]

    expl_map = {e["job_id"]: e["explanation"] for e in expl_data.get("explanations", [])}
    for job in top_jobs:
        job["explanation"] = expl_map.get(job["id"], "Strong technical alignment detected.")

    clarifying_q = expl_data.get(
        "clarifying_question",
        "What type of work environment do you prefer — remote, hybrid, or on-site?"
    )
    return candidate, top_jobs, clarifying_q


# ─── Refine agent ────────────────────────────────────────────────────────────

async def run_refine_agent(
    resume_text: str, clarifying_question: str, candidate_answer: str
) -> tuple[list[dict], str]:

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set.")

    base_jobs = rank_jobs(resume_text, n=10)

    prompt = (
        "You are a smart job match re-ranker.\n\n"
        f"Resume (first 500 chars): {resume_text[:500]}\n\n"
        f"Clarifying question: {clarifying_question}\n"
        f"Candidate answer: {candidate_answer}\n\n"
        f"Top-10 TF-IDF matched jobs:\n"
        + json.dumps(
            [{k: v for k, v in j.items() if k not in ("description",)} for j in base_jobs],
            indent=2,
        )
        + "\n\nBased on the candidate's answer, return the top 5 job IDs re-ranked and a short "
          "2-3 sentence reasoning for why the order changed.\n"
          "Respond ONLY with valid JSON — no markdown fences:\n"
          '{"ranked_ids": [id1, id2, id3, id4, id5], "reasoning": "..."}'
    )

    resp = await _gemini([{"role": "user", "parts": [{"text": prompt}]}])
    raw  = re.sub(r"```json|```", "", _text(resp)).strip()

    try:
        parsed     = json.loads(raw)
        ranked_ids = parsed["ranked_ids"]
        reasoning  = parsed["reasoning"]
    except Exception:
        ranked_ids = [j["id"] for j in base_jobs[:5]]
        reasoning  = "Re-ranking applied based on your stated preference."

    id_map      = {j["id"]: j for j in base_jobs}
    ranked_jobs = [id_map[rid] for rid in ranked_ids if rid in id_map][:5]

    # Mini explanation pass for refined list
    expl_prompt = (
        f"Candidate said: '{candidate_answer}'\n"
        "Write one sentence per job explaining fit after re-ranking:\n"
        + json.dumps([{"id": j["id"], "title": j["title"], "domain": j["domain"],
                       "remote": j.get("remote")} for j in ranked_jobs])
        + '\nRespond ONLY valid JSON: {"explanations": [{"job_id": N, "explanation": "..."}]}'
    )
    er   = await _gemini([{"role": "user", "parts": [{"text": expl_prompt}]}])
    eraw = re.sub(r"```json|```", "", _text(er)).strip()
    try:
        emap = {e["job_id"]: e["explanation"]
                for e in json.loads(eraw).get("explanations", [])}
    except Exception:
        emap = {}

    for job in ranked_jobs:
        job["explanation"] = emap.get(job["id"], "Matched based on your stated preference.")

    return ranked_jobs, reasoning


# ─── Pydantic models ─────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    resume_text: str

    @field_validator("resume_text")
    @classmethod
    def check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("resume_text cannot be empty")
        if len(v) < 30:
            raise ValueError("resume_text is too short — paste a complete resume")
        return v


class RefineRequest(BaseModel):
    resume_text:         str
    clarifying_question: str
    candidate_answer:    str

    @field_validator("resume_text", "clarifying_question", "candidate_answer")
    @classmethod
    def check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty")
        return v


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui = Path(__file__).parent.parent / "index.html"
    if ui.exists():
        return HTMLResponse(content=ui.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Smart Job Match Agent — visit <a href='/docs'>/docs</a></h2>")


@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "jobs_loaded":    len(JOBS),
        "model":          GEMINI_MODEL,
        "api_key_set":    bool(GEMINI_API_KEY),
        "tfidf_vocab":    len(IDF_MAP),
    }


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    t0       = time.time()
    top_jobs = rank_jobs(req.resume_text, n=TOP_N)
    candidate, ranked_jobs, clarifying_q = await run_agent(req.resume_text, top_jobs)

    return {
        "candidate": {
            "name":             candidate.get("name", "Unknown"),
            "skills":           candidate.get("skills", []),
            "experience_years": candidate.get("experience_years", 0),
            "preferred_roles":  candidate.get("preferred_roles", []),
            "education":        candidate.get("education", ""),
        },
        "ranked_jobs":         ranked_jobs,
        "clarifying_question": clarifying_q,
        "meta": {
            "model":          GEMINI_MODEL,
            "ranking_method": "TF-IDF cosine similarity",
            "jobs_scored":    len(JOBS),
            "top_n":          TOP_N,
            "latency_s":      round(time.time() - t0, 2),
        },
    }


@app.post("/refine")
async def refine(req: RefineRequest):
    t0 = time.time()
    ranked_jobs, reasoning = await run_refine_agent(
        req.resume_text, req.clarifying_question, req.candidate_answer
    )
    return {
        "ranked_jobs": ranked_jobs,
        "reasoning":   reasoning,
        "meta":        {"latency_s": round(time.time() - t0, 2)},
    }


@app.get("/jobs")
async def list_jobs(domain: str | None = None, remote: bool | None = None):
    jobs = JOBS
    if domain:
        jobs = [j for j in jobs if j.get("domain", "").lower() == domain.lower()]
    if remote is not None:
        jobs = [j for j in jobs if j.get("remote") == remote]
    return {"count": len(jobs), "jobs": jobs}


@app.get("/stats")
async def stats():
    domains  = Counter(j.get("domain", "Unknown") for j in JOBS)
    salaries = [j.get("salary_lpa", 0) for j in JOBS]
    return {
        "total_jobs":       len(JOBS),
        "remote_count":     sum(1 for j in JOBS if j.get("remote")),
        "domain_breakdown": dict(domains.most_common()),
        "salary":           {"min": min(salaries), "max": max(salaries),
                             "avg": round(sum(salaries) / len(salaries), 1)},
        "tfidf_vocab_size": len(IDF_MAP),
    }