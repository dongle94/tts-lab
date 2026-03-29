# Phase 4 Technical Comparison

Source data:

- `results/phase1/supertonic2/benchmark.json`
- `results/phase2/melotts/benchmark.json`
- `results/phase3/qwen3-tts/benchmark.json`
- `results/phase3/qwen3-tts/phase3_modes.json`

## 1. Core Metrics

| Model | Cold start (`load_ready_sec`) | Avg RTF | Avg chars/sec | Peak memory | GPU required |
|---|---:|---:|---:|---:|---|
| Supertonic2 | 13.295 s | 0.0244 | 142.8 | 0.596 GB RSS | No |
| MeloTTS | 37.756 s | 0.2226 | 42.1 | 2.483 GB RSS | No |
| Qwen3-TTS | 126.974 s | 0.8377 | 9.51 | 4.336 GB GPU alloc | Yes |

Notes:

- For Supertonic2 and MeloTTS, `peak memory` is process RSS peak because the runs were CPU-only in this workspace.
- For Qwen3-TTS, `peak memory` uses GPU allocated peak bytes from `benchmark.json` because the phase 3 run executed on CUDA.
- Qwen3-TTS `benchmark.json` is a CustomVoice-centric benchmark; English/mixed language cases are affected by a language hardcoding bug and should not be used for language-sensitive comparison.

## 2. Technical Limits and Constraints

### Supertonic2

- `PH1_STEP_2`, `PH1_STEP_5`, and `PH1_STEP_10` generation time is effectively flat on the current WSL CPU stack: 0.3975 s, 0.3886 s, and 0.3883 s.
- Interpretation: the benchmark here is dominated by local CPU/WSL execution characteristics, so step count does not materially change wall time in this environment.
- Result: good for low-latency Korean/English baseline synthesis, but step tuning does not yield a meaningful runtime separation here.

### MeloTTS

- Phase 2 `LONG_CHUNK` is not trustworthy as-is.
- The chunked path produced `ph2_long_chunk_chunk1.wav` + `chunk2.wav` + `chunk3.wav` totaling about 28.524 s, while the final merged `ph2_long_chunk.wav` was 15.523 s for the same text.
- Root cause: the chunking path already normalized each chunk to 24 kHz, then the final save path resampled again from `chunk_source_sr=44100`.
- Action: remove the double resampling before using MeloTTS chunking numbers in Phase 4 comparisons.

### Qwen3-TTS

- `benchmark.json` records `EN_S_01`, `EN_L_01`, `MIX_01`, and `MIX_02` with `lang: "ko"` because `QwenAdapter` currently hardcodes `language="Korean"`.
- Result: Qwen3-TTS English and mixed-language numbers in `benchmark.json` are not reliable for Phase 4 language comparison.
- Streaming limitation: the installed `qwen_tts` wrapper exposes only simulated streaming text input via `non_streaming_mode`; it does not expose true streaming generation or packet callbacks, so first packet latency was not measurable at the package level.

## 3. Use-Case Recommendation Matrix

| Use case | Best model | Why |
|---|---|---|
| Korean production TTS with low latency and low operating cost | Supertonic2 | Fastest by a wide margin, smallest memory footprint, and easiest to operate on CPU/WSL. |
| Korean/English mixed workloads with accent variety | MeloTTS | Better multilingual support and accent coverage than Supertonic2, while still CPU-friendly. |
| Personalized Korean voice, voice cloning, voice design, and style control | Qwen3-TTS | Best quality ceiling for Korean style control and cloning/design workflows, but heavier to run. |
| High-volume batch synthesis on limited hardware | Supertonic2 | Lowest compute and memory cost. |
| Multi-accent global content where pronunciation variety matters | MeloTTS | Broader accent coverage than Supertonic2 and lower operational burden than Qwen3-TTS. |
| Premium branding voice, bespoke tone, or assistant persona design | Qwen3-TTS | Strongest control surface, especially for CustomVoice and VoiceDesign. |

## 4. Operating Cost and Difficulty

| Model | Operating cost | Setup difficulty | Runtime risk | Comment |
|---|---|---|---|---|
| Supertonic2 | Low | Low | Low | Works well on CPU/WSL, minimal memory pressure, simple deployment path. |
| MeloTTS | Medium | Medium | Medium | More memory than Supertonic2, broader accent coverage, but chunking needs care. |
| Qwen3-TTS | High | High | High | Largest cold start, highest memory footprint, GPU-dependent for practical use, and has a few workflow caveats. |

## 5. Decision Summary

- If the target is low-cost Korean production synthesis, choose **Supertonic2**.
- If the target needs multilingual or accent-aware synthesis, choose **MeloTTS**.
- If the target needs premium personalization, cloning, and voice design, choose **Qwen3-TTS**.
- For Phase 4 comparison work, exclude the known-invalid subsets:
  - MeloTTS `LONG_CHUNK`
  - Qwen3-TTS English/mixed cases in `benchmark.json`
