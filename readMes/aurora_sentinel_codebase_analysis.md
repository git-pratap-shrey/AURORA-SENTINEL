# AURORA-SENTINEL — AI/ML System Briefing

> **For:** Incoming AI/ML Engineer  
> **Date:** March 2026  
> **Scope:** Everything you need to know about the AI brains of this system

---

## 1. Architecture Overview

The system has **three independent execution environments** that talk to each other via HTTP and WebSocket:

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (React/Next.js)                                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │ REST + WebSocket
┌─────────────────▼───────────────────────────────────────────────┐
│  Backend (FastAPI)  – port varies                               │
│  ├─ api/routers/stream.py        (ML-only live feed)            │
│  ├─ api/routers/stream_vlm.py    (ML+VLM live feed)             │
│  ├─ api/routers/intelligence.py  (chat + search + upload)       │
│  ├─ api/routers/video.py         (video management)             │
│  ├─ api/routers/alerts.py        (alert CRUD)                   │
│  ├─ services/ml_service.py       (YOLOv8 singleton)             │
│  ├─ services/vlm_service.py      (VLM router + providers)       │
│  ├─ services/scoring_service.py  (two-tier scorer)              │
│  ├─ services/offline_processor.py(uploaded video pipeline)      │
│  └─ services/search_service.py   (ChromaDB RAG)                 │
└─────────────────┬───────────────────────────────────────────────┘
                  │ HTTP (port 3001)
┌─────────────────▼───────────────────────────────────────────────┐
│  AI Intelligence Layer (Node.js)  – port 3001                   │
│  └─ aiRouter_enhanced.py  (Qwen2-VL → Ollama → Nemotron chain)  │
│     Also calls back into backend's vlm_service.NemotronProvider │
└─────────────────────────────────────────────────────────────────┘
```

**Key config file:** [config.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/config.py)

| Setting | Default | Notes |
|---|---|---|
| `PRIMARY_VLM_PROVIDER` | `ollama_cloud` | Switch to `qwen2vl_local` for GPU |
| `OLLAMA_CLOUD_MODEL` | `qwen3-vl:235b-cloud` | The Ollama model tag |
| `QWEN2VL_MODEL_ID` | `Qwen/Qwen2-VL-2B-Instruct` | Local PyTorch model |
| `PRELOAD_LOCAL_MODELS` | `False` | Set `True` on a GPU machine |
| `EMBEDDING_MODEL_ID` | `all-MiniLM-L6-v2` | For RAG search |
| `NEMOTRON_MODEL_ID` | `nvidia/nemotron-colembed-vl-4b-v2` | Verification embedder |

---

## 2. The ML Layer (YOLOv8)

**File:** [ml_service.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/ml_service.py)

[MLService](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/ml_service.py#19-72) is a thread-safe singleton. It loads three sub-models in a **background thread** (so the API startup is never blocked):

| Sub-model | Class | Role |
|---|---|---|
| `UnifiedDetector` | [models/detection/detector.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/models/detection/detector.py) | YOLOv8 object/pose/weapon detection |
| `RiskScoringEngine` | [models/scoring/risk_engine.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/models/scoring/risk_engine.py) | Converts detections → ML risk score 0-100 |
| `PrivacyAnonymizer` | [models/privacy/anonymizer.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/models/privacy/anonymizer.py) | Face blurring / GDPR compliance |

**What `UnifiedDetector.process_frame()` returns:**
```python
{
  'poses':   [...],   # YOLO-pose keypoints per person
  'objects': [...],   # Bounding boxes + class + track_id
  'weapons': [...],   # Dedicated weapon sub-model hits
  'patterns': [...]   # High-level behavioural patterns
}
```

**Risk Engine output:** [(risk_score: float 0-100, risk_factors: dict)](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/search_service.py#37-56)  
Alert threshold in live mode: **>65**. Recording auto-starts at **>80**.

---

## 3. The Two-Tier Scoring Service

**File:** [scoring_service.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/scoring_service.py)

This is the **canonical scoring formula** for _uploaded video analysis_ and the AI Intelligence layer:

```
Final_Score = 0.3 × ML_Score + 0.7 × AI_Score   (if both available)
Final_Score = ML_Score                             (if AI unavailable)
Final_Score = AI_Score                             (if ML unavailable)
```

**Optimization:** If `ML_Score < 20`, AI analysis is skipped entirely (resource saving).  
**Alert threshold:** `final_score > 60`.

The `ai_score` fed into this formula can be **Nemotron-adjusted** if that verification model is available and agrees with Qwen2-VL.

---

## 4. The VLM Layer — Provider Chain

**File:** [vlm_service.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/vlm_service.py)

[VLMService](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/vlm_service.py#686-1068) supports **6 providers** with a sequential fallback strategy:

```
LocalAIProvider (port 3001)   ← Primary (calls the AI Intelligence Layer)
 ↓ fails
Qwen2VLProvider               ← Local GPU model (ENABLE_HEAVY_MODELS=true)
 ↓ fails
OllamaProvider                ← Local Ollama (qwen3-vl:235b-cloud)
 ↓ fails
GeminiProvider                ← Cloud (gemini-1.5-flash, rotates on 429)
 ↓ fails
QwenProvider / HuggingFaceProvider  ← HF API (last resort)
```

The **[_fusion_engine()](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/vlm_service.py#719-786)** method aggregates results from multiple providers using weighted averaging:

| Provider | Weight |
|---|---|
| Qwen2-VL (local GPU) | 0.30 |
| Gemini (cloud) | 0.30 |
| Ollama | 0.15 |
| Qwen (HF API) | 0.10 |
| Nemotron | 0.10 |
| HuggingFace | 0.05 |

**Key safety rule:** If `boxing` / `sparring` / `referee` keywords appear in descriptions and no `street fight` / `assault` keywords are present, the final risk is capped at **15**.

### Nemotron Verification (nvidia/nemotron-colembed-vl-4b-v2)

This is an **embedding model**, NOT a generative model. It's used to verify Qwen2-VL's output:
1. Encodes the frame as an image embedding
2. Encodes Qwen's text summary as a text embedding
3. Computes cosine similarity — if similarity > 0.6, Qwen's analysis is "verified"
4. Also classifies into 4 threat categories via embedding similarity: `real_fight`, `organized_sport`, `normal`, `suspicious`
5. Produces a `recommended_score` that overrides or blends with Qwen's score

**Timeout:** 3 seconds. On timeout, falls back to Qwen's raw score.

---

## 5. The AI Intelligence Layer (Node.js / Python hybrid)

**File:** [aiRouter_enhanced.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/ai-intelligence-layer/aiRouter_enhanced.py)

This is a Python module served by the Node.js layer ([server_local.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/ai-intelligence-layer/server_local.py)) on **port 3001**. The backend calls it via HTTP.

**Main entry point:** [analyze_image(image_data, ml_score, ml_factors, camera_id)](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/ai-intelligence-layer/aiRouter_enhanced.py#512-790)

**Decision tree (5-second hard timeout):**

```
ml_score < 20?  →  Return early (low risk, no deep analysis)
    ↓
PRIMARY_PROVIDER == "ollama_cloud"?
    → Try Ollama first, then fallback to Qwen2-VL
    ↓ (default: qwen2vl_local)
Try Qwen2-VL
    ↓ success → Try Nemotron verification (3s timeout)
                 → Use recommended_score
    ↓ fail
Try Ollama fallback
    ↓ fail
Use ML score directly (ml_fallback, confidence=0.3)
```

**Model availability tracker** ([model_availability.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/ai-intelligence-layer/model_availability.py)): Tracks failure counts with a **cooldown period** — if a model fails repeatedly, it's skipped for a while to prevent cascading timeouts.

**Prompt strategy:** Two different prompts based on `ml_score`:
- `ml_score > 70` → Detailed structured prompt asking for JSON output with `aiScore`, `sceneType`, `explanation`, `confidence`
- `ml_score ≤ 70` → General description prompt

---

## 6. Live Video Pipelines

Two WebSocket endpoints handle live camera feeds:

### `/stream/live-feed` (ML-only)
**File:** [stream.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/api/routers/stream.py)

- Receives raw JPEG frames over WebSocket
- Runs YOLO detection + risk scoring every frame (with adaptive skipping if >50ms)
- Face anonymization via `PrivacyAnonymizer`
- Saves alert to SQLite DB (10s cooldown)
- Auto-starts clip recording at risk >80
- **Does NOT call VLM in real-time**

### `/stream/vlm-feed` (ML + VLM)
**File:** [stream_vlm.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/api/routers/stream_vlm.py)

Same as above **plus:**
- **Motion change-point detection** (EMA-based MAD on downscaled frames) — independently detects sudden motion even if ML misses it
- **Async VLM analysis:** Triggers `vlm_service.analyze_scene()` in a background executor:
  - Every 10 seconds (periodic)
  - If `risk_score > 60` and last VLM was >3s ago
  - If motion spike or scene change detected
- While VLM is running, YOLO is paused (reuses cached detection) to prevent video freeze
- Adds `vlm_narrative` field to WebSocket JSON response

> [!IMPORTANT]
> The VLM in live mode only gets **a single frame** extracted from the stream. It has no temporal context about what happened before/after that frame.

---

## 7. Uploaded Video Pipeline

**File:** [offline_processor.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/offline_processor.py)

When a video is uploaded through the intelligence tab:

1. **Scan:** Finds all `.mp4/.avi/.mkv` files in `storage/clips/`
2. **ML Fast Filter:** Runs YOLO every **2 seconds** of video
3. **VLM Trigger:** Fires VLM analysis if:
   - YOLO detected a weapon, OR
   - YOLO detected people, OR
   - Periodic check every 6 seconds
4. **Audio Analysis:** [audio_service.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/audio_service.py) analyzes the audio track for threat sounds
5. **Metadata Store:** Saves events to `storage/metadata.json` (thread-safe with lock)
6. **Vector Index:** Auto-indexes all events into ChromaDB for search
7. **Parallel Processing:** Up to 4 videos processed concurrently (`ThreadPoolExecutor`)

**Event structure stored per timestamp:**
```python
{
  "timestamp": 12.50,
  "description": "...",    # VLM-generated text
  "threats": ["fight"],    # Keyword-extracted tags
  "severity": "high",      # low/medium/high
  "provider": "CORTEX-VLM",
  "confidence": 0.85
}
```

---

## 8. The Chatbot (Intelligence Chat)

**File:** [intelligence.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/api/routers/intelligence.py)  
**Endpoint:** `POST /intelligence/chat`

**Request:** `{ "question": "...", "filename": "optional.mp4" }`

**Current flow (4-level waterfall):**

```
1. Find video file (or use latest if none specified)
   ↓
2. Extract ONE frame from the MIDDLE of the video
   ↓
3. Call vlm_service.answer_question(image_data, question)
   ↓ fails or no frame
4. Fallback: Pattern-match question against metadata.json descriptions
   (keyword matching: "fight", "boxing", "how many", "describe", etc.)
   ↓ no metadata
5. Return: "Please upload a video first"
```

> [!WARNING]
> **Critical limitation:** The chatbot extracts only the **middle frame** of the video, regardless of the question. If you ask "what happened at 2 minutes?", it still looks at the middle frame. There is **no timeline awareness** and **no use of the stored per-second event data** in the primary VLM path.

---

## 9. Semantic Search (RAG)

**File:** [search_service.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/services/search_service.py)

- **Vector DB:** ChromaDB (persistent at `storage/vectordb`)
- **Embedding model:** `all-MiniLM-L6-v2` via sentence-transformers (lazy-loaded on first search)
- **Collection:** `video_events_v2`
- **Fallback:** Keyword linear scan over `metadata.json` if ChromaDB/sentence-transformers not installed

**Stored per event:** filename, timestamp, severity, threats (comma-joined), provider, confidence  
**Endpoint:** `GET /intelligence/search?q=person+with+knife&limit=5&filename=clip.mp4`

---

## 10. Database (SQLite/SQLAlchemy)

**File:** [models.py](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/backend/db/models.py)

Two tables:

**`alerts`** — Real-time threat alerts from live stream:
- `ml_score`, `ai_score`, `final_score`, `detection_source`
- `ai_explanation`, `ai_scene_type`, `ai_confidence`
- `status` (pending → acknowledged → resolved)
- `resolution_type`, `resolution_notes` (operator accountability)

**`system_settings`** — Key-value store for runtime config

> [!NOTE]
> The `metadata.json` file (flat JSON) and `chromadb` (vector DB) are **separate** from SQLite. SQLite stores live-stream alerts; JSON/ChromaDB stores uploaded-video analysis. These are never merged/cross-queried at present.

---

## 11. Improvement Recommendations

### 🔴 High Priority (Chatbot / Your Focus)

1. **Timeline-aware chatbot:** Instead of extracting one middle frame, the chatbot should:
   - Parse the stored `events[]` array from `metadata.json`
   - Find timestamp-relevant events based on the question ("what happened at 2 min" → events near 120s)
   - Feed the matching event **descriptions** (already generated by VLM) as context to an LLM
   - Use RAG: embed the question, query ChromaDB, return top-k event descriptions as context
   - This gives rich, timeline-accurate answers **without re-running vision models**

2. **Multi-turn conversation context:** The current `/chat` endpoint is stateless. A session ID + conversation history (stored server-side or in a lightweight KV) would enable follow-up questions like "who else was there?"

3. **Smarter frame selection for visual Q&A:** When a frame is needed for visual analysis:
   - Parse timestamps from the question (regex for "2 minutes", "at :30", "beginning", "end")
   - Query ChromaDB for relevant event timestamps
   - Extract frame at the relevant timestamp instead of always using `total_frames // 2`

4. **Return structured timeline in chat responses:** The chatbot response should include a `timeline` field listing which events (timestamp + description + severity) were used to answer.

### 🟡 Medium Priority (ML ↔ AI Integration)

5. **Live feed VLM is stateless:** The `stream_vlm.py` VLM trigger sends a single frame with no history. Pass the last N VLM narratives as context in the prompt to enable temporal reasoning ("the fight that started 30s ago is ongoing").

6. **Two-tier scoring not wired into live feed:** `stream.py` and `stream_vlm.py` use the `RiskScoringEngine` directly but **don't use `TwoTierScoringService`**. That means the 0.3×ML + 0.7×AI weighted formula is only applied in uploaded video processing, not live surveillance. Wire `scoring_service.py` into the VLM live feed loop.

7. **Alert DB doesn't store VLM narrative for live alerts:** When a VLM analysis escalates risk during live feed, the `save_alert_sync()` function doesn't write the `ai_explanation` or `ai_scene_type` to the `alerts` table (those columns exist but aren't populated from the stream router). Add this to fully utilize the schema.

8. **Nemotron disabled by default:** `PRELOAD_LOCAL_MODELS = False` means Nemotron never loads. Since it's the verification layer, consider loading it lazily on first use, or document a GPU deployment mode clearly.

### 🟢 Natural Feature Upgrades

9. **Video-level summary:** After `offline_processor` finishes, generate a single LLM-synthesized paragraph summarizing the entire video (feeds all event descriptions into an LLM). This is the "executive summary" the chatbot should lead with.

10. **Cross-video search:** The search endpoint supports `filename` filter but doesn't have a "search across all videos" summary view. A `/intelligence/cross-search` endpoint returning events grouped by video filename would be very useful for operators.

11. **Agent-style chatbot loop:** Rather than the waterfall fallback approach, use a ReAct-style agent: the LLM decides whether to call `visual_qa`, `metadata_lookup`, or `timeline_search` as tools based on the question type.

12. **Real-time VLM frequency control:** Let the frontend set `VLM_INTERVAL` via a WebSocket message. Operators could request "analyze every 3 seconds" for high-priority cameras.
