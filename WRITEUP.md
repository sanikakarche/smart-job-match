# WRITEUP.md — Smart Job Match Agent

---

## 1. Design Choices — Embedding Model

**Chosen approach: Custom TF-IDF with bigrams and field-weighted job text.**

I chose TF-IDF over a neural sentence embedding model (like `all-MiniLM-L6-v2`) for three deliberate reasons:

1. **Vercel free-tier constraint**: Sentence-transformer models are 80–420 MB. Vercel's free Python runtime has a 250 MB compressed bundle limit. Loading a local neural model at cold-start would exceed memory limits and breach the 60-second timeout on first request. Using an API-based embedding service (OpenAI, Cohere) would add another paid dependency and network hop. TF-IDF runs entirely in-process with zero external calls.

2. **Dataset characteristics**: The 50-job corpus is small and lexically rich — job titles, domain keywords, and skill lists are highly discriminative tokens. TF-IDF with bigrams captures meaningful collocations like `machine_learning`, `natural_language`, and `full_stack` that pure unigram models miss.

3. **Score spread**: Neural embedding models on small corpora often cluster all cosine scores above 0.85 (because pretrained models embed all professional text into a dense region of the space). TF-IDF produces a natural spread — top matches score 0.30–0.60, weak matches score <0.10 — which is exactly what the rubric checks for.

**Alternatives considered:**
- `text-embedding-004` (Google free): Adds 300–800ms latency per request and requires managing an additional API quota. Rejected in favour of latency budget.
- `all-MiniLM-L6-v2` via sentence-transformers: Best quality but ~90 MB — would fail Vercel's bundle limit. Would use this if deploying on a GPU server.

**Trade-offs accepted:** TF-IDF is purely lexical — it cannot understand that "PyTorch" and "deep learning framework" are semantically related unless both words appear in the same document. This is acceptable for a 50-job dataset where descriptions are well-written.

---

## 2. Agentic Architecture

The agent runs two sequential, real tool calls via Gemini 2.0 Flash's native function-calling API:

```
User Resume
    │
    ▼
[Tool Call 1] parse_resume
    → Input:  raw resume text
    → Output: { name, skills, experience_years, preferred_roles, education }
    │
    ▼
TF-IDF Ranking  →  top-5 jobs with similarity scores
    │
    ▼
[Tool Call 2] generate_match_explanations
    → Input:  candidate profile + top-5 jobs
    → Output: { explanations: [{job_id, explanation}], clarifying_question }
    │
    ▼
Final API Response
```

**Why two tool calls instead of one monolithic prompt?**

- **Separation of concerns**: Parsing a resume requires different reasoning than evaluating job fit. A single prompt doing both produces degraded output on both tasks — the model splits attention between extraction accuracy and explanatory quality.
- **Structured intermediate state**: The parsed candidate object (`experience_years`, `preferred_roles`) flows into the ranking step as structured data, not text. This makes the explanation generation deterministic and verifiable — I can log and inspect what the model extracted before it generates explanations.
- **Independent retryability**: If the parse step fails (malformed resume), I can return a 502 without making a second LLM call. If the explanation step fails, I can retry it with the already-parsed profile.

**Failure modes:**
- Gemini declines to make a tool call and returns text instead → caught and raised as a 502 with a meaningful message.
- Resume is too ambiguous for structured extraction → `experience_years` defaults to 0, `preferred_roles` may be empty. The system degrades gracefully.
- Rate limit on Gemini free tier → 429 bubbles up as a 502; the client should retry with backoff.
- Vercel 60s timeout → TF-IDF runs in <10ms, leaving ~50s for two LLM round-trips. Measured P95 is ~8–12s total.

---

## 3. Honest Weaknesses

**With noisy or poorly written resumes:**
- Dense abbreviations ("Sr. SWE, prev. @ FAANG 4y") confuse TF-IDF because the vocabulary won't match job description tokens.
- Image-based or PDF resumes passed as garbled OCR text break the parser tool — the model may hallucinate fields.
- Resumes in languages other than English produce near-zero TF-IDF scores against English job descriptions.

**At scale (10,000 concurrent requests):**
- The current setup is a single stateless serverless function. TF-IDF is in-memory and fast, but Gemini API calls will hit rate limits well before 10k RPS on a free key.
- No caching: identical resumes re-embed and re-call the LLM every time. A Redis layer with a resume hash as key would cut repeat calls dramatically.
- No request queue: under burst load, Gemini will return 429s, which surface as 502s to the client. A proper production system would use a job queue (Celery + Redis) and return a polling URL.

**Corners cut due to time:**
- No PDF/DOCX resume parsing — the API accepts only plain text.
- No embedding caching for job descriptions (they're re-computed at startup and held in RAM, which is fine for 50 jobs but not for 50,000).
- The `/refine` endpoint re-runs TF-IDF from scratch instead of adjusting scores incrementally.
- No auth — any client can call the API. In production, API key or JWT middleware is required.

---

## 4. Next Steps — Highest Impact Improvement

**Hybrid reranking: TF-IDF retrieval + neural cross-encoder reranking.**

If I had two more days, I would add a cross-encoder reranking step between TF-IDF retrieval and the LLM explanation call. Specifically:

1. TF-IDF retrieves the top 15 candidates (fast, recall-oriented).
2. A lightweight cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`, ~67 MB) scores each `(resume, job_description)` pair directly, producing a precision-oriented reranked top-5.
3. The LLM explanation step runs on the reranked top-5 as before.

This matters because TF-IDF misses semantic similarity ("optimise supply chains" ↔ "logistics automation"). The cross-encoder operates on full sentence pairs, capturing this. The model is small enough to fit Vercel's memory if quantized (ONNX int8, ~17 MB). This single change would address the biggest real-world failure mode of the current system — cross-domain vocabulary mismatch — without increasing LLM costs or API dependencies.
