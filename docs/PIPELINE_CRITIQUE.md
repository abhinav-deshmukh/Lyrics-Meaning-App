# Pipeline critique: Download → Transcribe → Interpret

Critical examination of the current workflow and concrete suggestions to **reduce latency** and **improve interpretation quality**.

---

## Current flow (summary)

1. **Download** — Cobalt API (or yt-dlp fallback) → full audio file to disk.
2. **Transcribe** — `transcribe_with_timestamps()`:
   - Split audio into 4‑min chunks (if >4 min).
   - **Sequential** Whisper API call per chunk; retries per chunk; optional full retry if result too short.
3. **Interpret** — `interpret_segments()`:
   - Dedupe by line text.
   - **Sequential** batches of 25 lines to Claude; gap-fill retry for missing; full retry if >50% missing.
4. **Post-process** — `merge_instrumental_segments()` (fast).
5. **Metadata** — `get_language_and_mood()`: one Claude call on first 8 lines.
6. **Suggestions** — `get_suggested_songs()` (cache + curated + optional YouTube search).
7. **Encode & cache** — `get_audio_base64()`, `save_to_cache()` (includes `_get_youtube_metadata()`).

---

## Latency: where time goes

| Step | Bottleneck | Typical impact |
|------|------------|----------------|
| Download | Network + Cobalt/yt-dlp; single-threaded. | 5–30 s |
| Transcribe | **Strictly sequential chunks**; no language hint so Whisper does auto-detect every chunk. | 30–90+ s for long songs |
| Interpret | **Strictly sequential batches**; 25 lines/batch → many round-trips for long songs. | 20–60+ s |
| Language/mood | One Claude call after interpret; could be done earlier. | ~2–5 s |
| save_to_cache | `_get_youtube_metadata(url)` **blocking** at end (yt-dlp again). | 2–10 s |

**Main latency levers:** parallelize transcribe chunks, parallelize interpret batches, avoid redundant work (e.g. metadata at end), and optionally hint language to Whisper.

---

## Quality: risks and gaps

- **No language hint to Whisper** — Auto-detect per chunk can mis-detect or be slower; first chunk could drive a single hint for the rest.
- **Interpretation prompt** — Good (original, romanized, translation, meaning). Missing: explicit “preserve line order”, “one array element per line”, and “no extra commentary”.
- **Dedupe by exact text** — Normalized to lowercase; repeated lines (e.g. chorus) get one interpretation. Fine for display; if you ever want per-segment nuance, this loses it.
- **Gap-fill / full retry** — Sensible; no structured fallback (e.g. “translate only” for lines that failed meaning).
- **Language/mood** — Done after full interpret; if interpret is slow, this doesn’t help earlier steps (e.g. no language→Whisper hint).

---

## Recommendations

### A. Reduce latency

1. **Parallelize transcribe chunks**
   - Chunks are independent. Use `concurrent.futures.ThreadPoolExecutor` (or `ProcessPoolExecutor` if you prefer) to call Whisper for all chunks in parallel, then sort segments by `start`.
   - **Effect:** For a 10 min song (3 chunks), transcribe time ~max(chunk times) instead of sum (~3× faster in the ideal case).

2. **Parallelize interpretation batches**
   - Batches are independent. Run all `_interpret_batch()` calls in parallel (same executor pattern), then merge `interp_lookup`.
   - **Effect:** For 80 unique lines (4 batches), interpret time ~one batch time instead of 4× (~4× faster).

3. **Don’t block on YouTube metadata at save time**
   - `save_to_cache()` calls `_get_youtube_metadata(url)` synchronously. Either:
     - Move metadata fetch to a background thread/queue and write cache when metadata returns, or
     - Save cache immediately with minimal meta (url, title, cached_at); backfill channel/duration in `get_cached_songs()` (you already do backfill for missing fields).
   - **Effect:** User sees “Ready” sooner; no 2–10 s stall at the end.

4. **Optional: hint Whisper language from a fast first chunk**
   - Transcribe first chunk only (or first 30 s), get Whisper’s detected language from the response, then transcribe remaining chunks with `language=detected`. Cuts redundant auto-detect and can improve accuracy.
   - **Effect:** Slightly faster and more consistent transcribe, better for mixed or rare languages.

5. **Optional: stream “ready to play” before suggestions**
   - Build karaoke payload (audio_base64, segments, language, mood, summary) and put it in session state + cache as soon as interpret and language/mood are done; show the player. Load “You might also like” asynchronously or in a second request and patch session state when ready.
   - **Effect:** Time-to-first-play is lower; suggestions appear when ready without blocking play.

### B. Improve interpretation quality

1. **Tighten the interpretation prompt**
   - Add: “Output exactly one JSON object per line, in the same order as the input. No extra keys, no explanation outside the array.”
   - Optionally: “For idioms or wordplay, translation should be the natural English meaning; put the literal meaning or cultural note in **meaning**.”
   - **Effect:** Fewer malformed outputs, clearer separation of translation vs meaning.

2. **Pass language and/or mood into the interpretation prompt**
   - Once you have `get_language_and_mood()` (or a cheap “language only” from first chunk), pass “Language: Spanish” (etc.) into `INTERPRETATION_PROMPT`. Reduces ambiguity and improves idiom handling.
   - **Effect:** More consistent translations and better meaning for idioms.

3. **Optional: two-phase interpret (translate first, then meaning)**
   - Phase 1: One batch call that only asks for translation (and romanization). Fast, fewer tokens.
   - Phase 2: One or more calls that ask only for “meaning” for each line (can batch), given the original + translation. Lets the model focus and avoids truncation on long songs.
   - **Effect:** Better meaning quality and more robust to long songs; slightly more calls but can be parallelized.

4. **Validate and repair JSON per batch**
   - You already have `_extract_json_array`. Add: if the array length doesn’t match input length, try to fix (e.g. merge/split by line boundaries) or re-request that batch with “output exactly N objects.”
   - **Effect:** Fewer silent drops and gap-fill retries.

### C. Order of implementation (impact vs effort)

| Priority | Change | Latency impact | Quality impact | Effort |
|----------|--------|----------------|----------------|--------|
| 1 | Parallelize transcribe chunks | High | — | Medium |
| 2 | Parallelize interpret batches | High | — | Low |
| 3 | Defer or background YouTube metadata in save_to_cache | Medium | — | Low |
| 4 | Add language hint to Whisper (from first chunk) | Medium | Medium | Medium |
| 5 | Add language to interpretation prompt | — | Medium | Low |
| 6 | Tighten interpretation prompt (order, one object per line) | — | Medium | Low |
| 7 | Stream “ready” before suggestions | Medium | — | Medium |
| 8 | Two-phase interpret (translate then meaning) | — | High | Medium |

---

## Summary

- **Latency:** The pipeline is mostly **sequential** (chunks and batches). Parallelizing **transcribe chunks** and **interpret batches** gives the biggest wins. Deferring or backgrounding **YouTube metadata** at save time removes an unnecessary stall at the end.
- **Quality:** Improvements come from **language hint to Whisper**, **language (and optionally mood) in the interpretation prompt**, and a **stricter prompt** (order, one object per line). An optional **two-phase interpret** (translate then meaning) can further improve meaning quality on long songs.

Implementing items 1–3 and 5–6 above will give a strong balance of faster time-to-play and better interpretations without a large refactor.
