# ConvoEval — Production-Ready Conversation Benchmark

> **Ocean Across AI/ML Engineer Assignment**
> Score every conversation turn across **300+ distinct facets** covering linguistic quality, pragmatics, safety, and emotion.

[![API](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## Live Demo

| | URL |
|---|---|
| **UI** | [convoeval.vercel.app](https://convo-eval.vercel.app/) *(deploy instructions below)* |
| **API** | [convoeval-api.railway.app](https://web-production-b8690.up.railway.app) *(deploy instructions below)* |

---

## What This Does

ConvoEval evaluates every **assistant turn** in a conversation across up to **369 facets** (scalable to 5000+) and returns:

- A **score 1–5** per facet with a one-sentence rationale
- A **confidence value 0.0–1.0** per score
- A **summary** with mean score, safety flags, and low-confidence flags for human review

**Example output:**
```json
{
  "turn_index": 1,
  "role": "assistant",
  "scores": {
    "compassion":       { "score": 5, "confidence": 0.91, "rationale": "Response shows deep empathetic understanding." },
    "dishonesty":       { "score": 1, "confidence": 0.97, "rationale": "No deceptive elements present." },
    "critical_reasoning":{ "score": 4, "confidence": 0.78, "rationale": "Differentiates possible causes logically." }
  }
}
```

---

## Assignment Requirements

| Requirement | Status | How |
|---|---|---|
| 300 facets | ✅ 369 facets | Cleaned from `Facets_Assignment.csv` |
| Scales to 5000 | ✅ | Chunk-based registry, zero code changes needed |
| 5-point ordinal scale | ✅ | Scores 1–5 with anchored rubrics |
| No one-shot prompts | ✅ | 6-turn chain-of-thought per facet chunk |
| Open-weight ≤16B model | ✅ | Qwen2.5-3B (3B params, open licence) |
| Confidence outputs | ✅ | Self-critique pass after scoring |
| Dockerised baseline | ✅ | `docker/docker-compose.yml` |
| Sample UI | ✅ | React UI (`ui/index.html`) |
| GitHub repo + docs | ✅ | This README |
| 50 scored conversations | ✅ | `data/sample_conversations/` |

---

## Architecture

```
Conversation JSON
      │
      ▼
 TurnSplitter          ← splits messages into Turn objects with context windows
      │
      ▼
 FacetRegistry         ← loads all 369 facets from CSV, O(1) lookup
      │
      ▼
 ScoringOrchestrator   ← batches facets into chunks of 20
      │
      ├─ Chunk 1 ──► OllamaClient.chat() ──► 6-turn CoT ──► scores + confidence
      ├─ Chunk 2 ──► OllamaClient.chat() ──► 6-turn CoT ──► scores + confidence
      └─ Chunk N ──► ...
      │
      ▼
 Aggregator            ← merges results, computes summary, flags safety issues
      │
      ▼
  JSON results
```

### Scoring Flow (no one-shot)

For each chunk of 20 facets × each assistant turn:

```
Turn 1  [user]       Present the conversation turn for analysis
Turn 2  [assistant]  Chain-of-thought: grammar, tone, safety, emotion...
Turn 3  [user]       "Score these 20 facets. Return JSON."
Turn 4  [assistant]  {"scores": {"compassion": {"score": 5, "rationale": "..."}}}
Turn 5  [user]       "How confident are you in each score? Return 0.0-1.0"
Turn 6  [assistant]  {"confidence": {"compassion": 0.91, ...}}
```

### Scaling to 5000 Facets

The architecture handles this without any code changes:

1. **Chunked batching** — facets scored in groups of 20. 5000 facets = 250 chunks, same code.
2. **Registry-driven** — add facets by adding CSV rows, then `GET /facets/reload`
3. **Async orchestration** — `asyncio` + semaphore for parallel chunk processing
4. **`evaluable_from_text` flag** — 52 biometric/lab facets are auto-skipped

---

## Project Structure

```
convo-eval/
├── data/
│   ├── facets_processed.csv          # 369 cleaned facets, 17 columns
│   └── sample_conversations/         # 50 scored conversation JSONs
├── src/
│   ├── pipeline/
│   │   ├── preprocessor.py           # Cleans raw CSV, adds 16 columns
│   │   ├── facet_registry.py         # Thread-safe in-memory facet store
│   │   └── turn_splitter.py          # Conversation → Turn objects
│   ├── models/
│   │   ├── ollama_client.py          # Unified client: Ollama + HF Inference API
│   │   └── prompt_builder.py         # Multi-turn prompt construction
│   ├── scoring/
│   │   ├── orchestrator.py           # Async chunked scoring (6-turn CoT)
│   │   ├── confidence.py             # Confidence estimation
│   │   └── aggregator.py             # Result aggregation + summaries
│   └── api/main.py                   # FastAPI REST API
├── scripts/
│   ├── preprocess_facets.py          # CLI: clean raw CSV
│   ├── score_conversations.py        # CLI: batch score conversation files
│   └── generate_samples.py          # CLI: generate sample conversations
├── ui/index.html                     # React UI (single file, no build step)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── railway.json                      # Railway.app deploy config
├── Procfile                          # Railway/Render process config
├── requirements.txt
└── .env.example
```

---

## Facet Dataset

The raw `Facets_Assignment.csv` (399 rows, 1 column) was cleaned and enriched to 369 facets × 17 columns:

| Added Column | Description |
|---|---|
| `facet_id` | Clean slug for code use (`common_sense`) |
| `domain` | Auto-classified into 8 domains |
| `description` | What the facet measures |
| `rubric_1` to `rubric_5` | Anchored score descriptions per level |
| `polarity` | `higher_is_better` or `lower_is_better` |
| `requires_context` | Needs preceding turns to score |
| `evaluable_from_text` | False for biometric/lab facets (52 skipped) |
| `score_direction` | Human-readable `high=good` / `low=good` |
| `prompt_hint` | Injected into LLM prompt for this facet |
| `chunk_group` | Batch assignment (facets / 20) |

**Domain distribution:**

| Domain | Count |
|---|---|
| personality | 104 |
| health_lifestyle | 51 |
| spiritual_values | 49 |
| emotion | 46 |
| cognitive | 36 |
| pragmatics | 34 |
| safety | 30 |
| linguistic | 19 |

---

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/convo-eval.git
cd convo-eval

# 2. Install
pip install -r requirements.txt

# 3. Install Ollama  →  https://ollama.com/download
ollama pull qwen2.5:3b

# 4. Config
cp .env.example .env        # default settings work out of the box

# 5. Run
ollama serve                                          # terminal 1
uvicorn src.api.main:app --reload --port 8000        # terminal 2
open ui/index.html                                   # browser
```

---

## Deployment

### API → Railway.app (free)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo
4. Add environment variables:
   ```
   BACKEND=huggingface
   HF_TOKEN=hf_your_token_here
   HF_MODEL=Qwen/Qwen2.5-3B-Instruct
   FACETS_PATH=data/facets_processed.csv
   ```
5. Railway auto-deploys. Copy the public URL.

### UI → Vercel (free)

1. Go to [vercel.com](https://vercel.com) → New Project → Import your GitHub repo
2. Set **Root Directory** to `ui`
3. Set **Framework** to `Other`
4. Add environment variable: `VITE_API_URL=https://your-railway-url.railway.app`
5. Deploy → get public URL

> Or just open `ui/index.html` directly in a browser for local use — no build step needed.

### HuggingFace Token (free)

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens → New Token (read permission is enough)
3. Copy token → paste as `HF_TOKEN` in Railway env vars

---

## API Reference

### `POST /score`
Score a full conversation.
```json
{
  "conversation": [
    {"role": "user",      "content": "I feel really anxious lately."},
    {"role": "assistant", "content": "That sounds really difficult..."}
  ],
  "facet_ids": ["compassion", "dishonesty"],
  "model": "qwen2.5:3b"
}
```

### `POST /score/turn`
Score a single turn.
```json
{ "content": "That sounds really difficult.", "role": "assistant",
  "facet_ids": ["compassion", "frankness"] }
```

### `GET /facets?domain=emotion&limit=100`
List facets with optional domain filter.

### `GET /facets/domains`
List all domains and facet counts.

### `GET /facets/reload`
Hot-reload the facets CSV without restarting the API.

### `GET /health`
Check API + Ollama/HF connectivity.

---

## Score Scale

| Score | Label | Meaning |
|---|---|---|
| 1 | Poor | Fails the facet entirely |
| 2 | Below Average | Partial failure, noticeable issues |
| 3 | Average | Meets minimum expectations |
| 4 | Good | Clearly above average |
| 5 | Excellent | Exemplary, near-perfect |

---

## Sample Conversations

50 pre-scored conversations are in `data/sample_conversations/`. They cover:

- Customer service (angry, grateful, confused)
- Mental health support
- Harmful/unsafe requests
- Factual Q&A
- Creative writing
- Technical help (Python, CORS, SQL)
- Bias and misinformation handling
- Multi-language
- Ethical dilemmas
- Philosophical discussions

---

## License

MIT
