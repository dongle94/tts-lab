# TTS 모델 탐색 작업 지침서

> **작성일**: 2026-03-28
> **작성**: Claude (CCG 워크플로 — Codex + Gemini 어드바이저 합성)
> **공유 대상**: Claude, Codex, Gemini (모든 에이전트 공통)
> **상태**: 진행 중

---

## 목적

MeloTTS, Supertonic2, Qwen3-TTS 세 가지 TTS 모델을 직접 설치·실행·비교하여, 각 모델의 실용적 강점과 한계를 파악한다. 프로그램을 만드는 것이 아니라 **실험·평가·의사결정**이 목표다.

---

## 에이전트 역할 정의 (Agent Role Definition)

| 에이전트 | 역할 | 핵심 책임 |
|---|---|---|
| **Claude** | Orchestrator & Synthesizer | 전체 로드맵 관리, 에이전트 간 결과물 충돌 조정, 최종 인사이트 도출 및 의사결정 |
| **Codex** | Technical Analyst | 환경 구축, 추론 실행, 성능 벤치마킹(RTF/latency), 기술적 병목 분석, 리스크 기록 |
| **Gemini** | UX & Documentation Specialist | 테스트 스크립트 설계, 음성 품질 평가(MOS), 대안 제시, 사용자 관점 가이드 작성 |

---

## 전제 조건 (Prerequisites)

모든 에이전트가 작업 전 반드시 숙지해야 할 사항:

### Python 환경 관리 — uv 사용

모든 Python 환경은 **uv**로 관리한다. 모델마다 독립된 uv 환경을 별도로 운영한다.

```bash
# uv 설치 (미설치 시)
curl -Ls https://astral.sh/uv/install.sh | sh
# 또는
pip install uv
```

| 모델 | uv 환경 디렉터리 | Python 버전 |
|---|---|---|
| MeloTTS | `envs/melotts/` | 3.10+ |
| Supertonic2 | `envs/supertonic2/` | 3.9–3.12 |
| Qwen3-TTS | `envs/qwen3-tts/` | **3.12 필수** |

**환경 생성 방법:**

```bash
# MeloTTS
uv venv envs/melotts --python 3.10
source envs/melotts/bin/activate
uv pip install -e path/to/MeloTTS

# Supertonic2
uv venv envs/supertonic2 --python 3.11
source envs/supertonic2/bin/activate
uv pip install supertonic onnxruntime==1.23.1

# Qwen3-TTS
uv venv envs/qwen3-tts --python 3.12
source envs/qwen3-tts/bin/activate
uv pip install qwen-tts soundfile
# flash-attn은 선택 (빌드 시간 큼)
# uv pip install flash-attn --no-build-isolation
```

> **원칙**: 세 환경은 절대 혼용하지 않는다. 패키지 추가 시 반드시 해당 모델 환경을 activate한 상태에서 진행.

### 하드웨어 분리 원칙

- **CPU 환경**: MeloTTS, Supertonic2 전용
- **GPU 환경**: Qwen3-TTS 전용 (CUDA 필수, VRAM 8GB+ 권장 / VoiceDesign 모드는 12GB+ 필요)
- 세 모델을 동일 환경에 억지로 묶지 않는다.

### 리서치 문서 위치

| 모델 | 문서 경로 |
|---|---|
| MeloTTS | `docs/research/melotts.md` |
| Supertonic2 | `docs/research/supertonic2.md` |
| Qwen3-TTS | `docs/research/qwen3-tts.md` |

### 라이선스 요약

| 모델 | 코드 | 가중치 | 상업적 이용 |
|---|---|---|---|
| MeloTTS | MIT | MIT | 자유 |
| Supertonic2 | MIT | **OpenRAIL-M** | 가능 (유해 목적 금지 조항 있음) |
| Qwen3-TTS | Apache 2.0 | Apache 2.0 | 자유 |

> Supertonic2: 코드와 가중치 라이선스가 다르다. 배포 시 가중치 라이선스(OpenRAIL-M) 전문 확인 필요.
> Qwen3-TTS: 보이스 클로닝 기능은 내부 정책(동의, 목적) 통제가 중요하다.

### 기술 비교 매트릭스 (전체 참조용)

| 항목 | MeloTTS | Supertonic2 | Qwen3-TTS |
|---|---|---|---|
| 아키텍처 | VITS 계열, 비자기회귀 | Flow-matching, ONNX | 이산 토크나이저 + LM |
| 실행 하드웨어 | CPU (GPU 불필요) | CPU/ONNX (GPU 불필요) | **CUDA GPU 필수** |
| 한국어 화자 | 단일 (`KR`) | 10개 중 KO 가능 | Sohee 프리셋 + 클로닝 |
| 보이스 클로닝 | 없음 | 없음 | **3초 제로샷** |
| 스타일 제어 | 속도만 | speed + total-step + voice JSON | **자연어 지시 (감정/속도/음색)** |
| 스트리밍 | 없음 | 없음 | **있음 (97ms 첫 패킷)** |
| 파인튜닝 | 가능 (데이터 필요) | 없음 | **Base 모델 SFT** |
| 설치 난이도 | 낮음 | 낮음~중간 | 높음 |
| 콜드 스타트 | 느림 (BERT ~10-30s) | 빠름 | GPU 로딩 부담 |
| 모델 크기 | ~208MB/언어 | ~305MB | 0.6B / 1.7B |
| 강점 | 단순, 가벼운 CPU TTS | 가장 빠른 로컬 TTS | 개인화, 스트리밍, 스타일 |
| 약점 | 기능 제한 | 클로닝/파인튜닝 부재 | GPU 의존, 환경 복잡 |

---

## 통신 프로토콜 (Communication Protocol)

에이전트 간 정보 공유 방식:

- **공유 요약**: 각 Phase 완료 시 `docs/SUMMARY.md`에 핵심 결과 업데이트
- **기술 데이터 (Codex 담당)**: JSON 형식으로 벤치마크 수치 기록
- **정성 평가 (Gemini 담당)**: Markdown 테이블로 모델별 장단점 기록
- **충돌 해결**: 기술적 가능성 vs 사용자 요구사항 충돌 시 → Claude가 최종 결정
- **Phase 전환**: 다음 Phase 진입 전 Claude 승인 필요

---

## Phase 계획

### Phase 0 — 공통 평가 하네스 구축

**목적**: 세 모델을 동일한 기준으로 비교할 수 있는 측정 파이프라인 확보

**R&R**:

| | Claude | Codex | Gemini |
|---|---|---|---|
| 역할 | 실험 우선순위 결정, Phase 1 진입 승인 | 평가 스크립트 구조 설계 | 테스트 문장 세트 작성 |

**Codex 작업 내용**:
- 생성 시간 / 오디오 길이 / RTF / chars/sec / first packet latency 측정 스크립트
- cold start time 측정 로직
- 메모리 사용량 기록 구조
- ASR 기반 WER/CER 파이프라인
- 샘플레이트 정규화 (모델별 출력 샘플레이트가 다를 수 있음)

**Gemini 작업 내용** (테스트 문장 세트 설계):
- 한국어 짧은 문장 / 긴 문장 / 숫자+기호 포함 문장
- 영어 짧은 문장 / 긴 문장
- 한국어+영어 혼합 문장
- 기능별 테스트 세트: 속도 제어 / 장문 합성 / 스트리밍 / 클로닝 / 스타일 지시

**완료 정의 (Definition of Done)**:
- 세 모델의 출력을 동일 포맷으로 저장 가능
- 요청별 메타데이터 기록 가능
- 재실행 시 결과 비교 가능

---

### Phase 1 — CPU 기준선: Supertonic2 검증

**목적**: 가장 설치가 간단한 모델로 CPU TTS 기준선 확보. `total-step`으로 품질-속도 곡선 파악.

**환경**:
- CPU 전용
- `source envs/supertonic2/bin/activate`
- Python 3.9–3.12, `onnxruntime==1.23.1` (버전 고정 필수)

**R&R**:

| | Claude | Codex | Gemini |
|---|---|---|---|
| 역할 | 제약사항 준수 여부 감시, Phase 2 승인 | 추론 실행, 성능 측정 | 음성 스타일(F1-F5, M1-M5) 청취 평가, 레이블링 |

**Codex 테스트 항목**:
- 한국어/영어 기본 합성
- M1–M5, F1–F5 전체 음성 스타일 로딩 확인
- `total-step=2/5/10` 품질-속도 비교
- `speed=0.9~1.5` 범위 반응
- 긴 문장 청킹 실험 (문장부호 기준 ~100자 단위)
- 배치 추론 안정성

**측정 지표**: chars/sec, RTF, 모델 로드 시간, step별 품질 개선폭, 장문 오류율

**완료 정의**:
- CPU에서 안정적 합성
- `total-step` 증가 시 품질이 개선되거나 최소한 악화되지 않음
- 장문에서 크래시 없이 반복 실행 가능

**기술적 주의사항 (Gotcha)**:
- `onnxruntime` 버전 불일치 → 실행 오류 발생 가능
- voice-style JSON 경로 오류 → 즉시 실패
- GPU 모드는 공식 미지원 — CPU 기준으로만 검증

---

### Phase 2 — CPU 기준선: MeloTTS 검증

**목적**: CPU 실시간 TTS의 또 다른 기준선. 기능이 단순하므로 **기본 품질 + 속도 제어 + 콜드 스타트**에 집중.

**환경**:
- CPU 전용
- `source envs/melotts/bin/activate`
- 소스 클론 후 `uv pip install -e .`
- Windows는 Docker/WSL 권장

**R&R**:

| | Claude | Codex | Gemini |
|---|---|---|---|
| 역할 | Supertonic2와 결과 대조 | 추론 실행, cold start 반복 측정 | 한국어 발음/자연성 청취 평가 |

**Codex 테스트 항목**:
- 한국어 단일 화자 합성
- 영어 액센트별 화자 선택 (EN-US, EN-BR, EN_INDIA, EN-AU)
- `speed=0.5/1.0/1.5/2.0` 반응
- cold start 반복 측정 (첫 실행 vs 이후 반복 호출)
- 동일 인스턴스 재사용 성능
- 긴 텍스트 forward-only 합성 안정성

**측정 지표**: cold start time, steady-state latency, RTF, speed 비례 오디오 길이 변화, WER/CER

**완료 정의**:
- CPU에서 실시간 수준 동작
- 반복 호출 시 안정적
- 속도 제어가 실제 재생 길이 변화로 이어짐

**기술적 주의사항 (Gotcha)**:
- BERT 로딩으로 첫 실행이 느림 → `TTS()` 인스턴스 한 번만 생성 후 재사용 필수
- 스트리밍 없음 → 긴 문장은 문장부호 기준으로 청킹
- 한국어 단일 화자 → 화자 다양성 평가 불가
- 감정/피치/스타일 제어 없음 — 기대하지 말 것

---

### Phase 3 — GPU 기능 검증: Qwen3-TTS

**목적**: 보이스 클로닝, 자연어 스타일 제어, 스트리밍 등 고급 기능 검증. 환경이 가장 복잡하므로 마지막에 진행.

**환경**:
- CUDA GPU 필수
- `source envs/qwen3-tts/bin/activate`
- Python 3.12 필수, `uv pip install qwen-tts soundfile`
- `flash-attn` 선택 설치: `uv pip install flash-attn --no-build-isolation`

**권장 설치 순서**:
1. Python 3.12 GPU 환경 생성
2. `qwen-tts` 설치
3. `flash-attn` 선택 설치
4. **0.6B 모델로 smoke test 먼저**
5. 1.7B 모델로 확대
6. CustomVoice → Base → VoiceDesign 순서로 검증

**R&R**:

| | Claude | Codex | Gemini |
|---|---|---|---|
| 역할 | CPU 모델 결과와 비교, 스타일 제어 유용성 평가 승인 | GPU 환경 구축, 추론 실행, 메모리/레이턴시 측정 | instruct 파라미터 테스트 문장 설계, 보이스 클로닝 품질 청취 평가 |

**Codex 테스트 항목**:
- 프리셋 스피커 `Sohee` 기본 합성 (한국어)
- 자연어 `instruct` 스타일 제어 (감정/속도/피치 각각 + 복합)
- 3초 레퍼런스 오디오로 보이스 클로닝
- `CustomVoice`, `Base`, `VoiceDesign` 모드별 차이 비교
- 스트리밍 first packet latency 측정
- 긴 문장 연속 합성 안정성
- 레퍼런스 오디오 품질 저하에 대한 민감도

**측정 지표**:
- first packet latency, 전체 합성 시간, 스트리밍 chunk 간격
- GPU 메모리 사용량
- 클로닝 화자 유사도 점수
- 스타일 지시 적합도 점수
- 장문 안정성, OOM 발생률

**완료 정의**:
- 스트리밍이 실제로 동작
- 3초 레퍼런스로 화자 정체성이 어느 정도 보존됨
- 자연어 지시가 감정/속도/톤에 반영됨
- GPU 메모리 초과 없이 안정화

**기술적 주의사항 (Gotcha)**:
- CUDA/드라이버/torch 조합에 따라 설치 실패 가능 → 0.6B로 먼저 확인
- `flash-attn`은 선택이지만 빌드 난도가 높음 → 없는 상태로도 먼저 동작 확인
- 레퍼런스 음성이 짧거나 noisy하면 클로닝 품질 크게 저하
- `ref_text` 정확도가 클로닝 품질에 영향 — 정확히 전사할 것
- 1.7B는 메모리 부담 큼 → **0.6B 검증 후 확대**

---

### Phase 4 — 통합 비교 및 의사결정

**목적**: 모델별 강점에 따라 서비스 포지션을 분리. "어떤 모델이 제일 좋다"가 아니라 **어떤 시나리오에 어떤 모델을 쓸지** 결정.

**R&R**:

| | Claude | Codex | Gemini |
|---|---|---|---|
| 역할 | 최종 권장 아키텍처 확정 및 문서화 | 기술적 한계점 및 최적화 방안 기술 | 사용자 가이드 및 유지보수 문서 작성 |

**평가 항목**:
- 운영 비용 / 응답 지연 / 음성 다양성
- 유지보수 난이도 / 라이선스 리스크
- 각 유스케이스 적합도

**권장 포지셔닝 (잠정)**:

| 시나리오 | 추천 모델 |
|---|---|
| 내부 오프라인 도구, 최소 의존성 | MeloTTS |
| 고속 배치 생성, 로컬 CPU TTS | Supertonic2 |
| 사용자 음성 클로닝, 실시간 스트리밍 | Qwen3-TTS Base |
| 감정/스타일 제어 필요 | Qwen3-TTS CustomVoice |
| 새 목소리 디자인 | Qwen3-TTS VoiceDesign |

**완료 정의**:
- 유스케이스별 최적 모델 추천안이 담긴 최종 요약 완성
- `docs/SUMMARY.md` 최종 버전 업데이트

---

### Phase 5 — (선택) Qwen3-TTS 파인튜닝 파일럿

> Phase 4까지 완료 후 필요 시 진행. Qwen3-TTS만 공개 파인튜닝 경로 있음.

**R&R**:

| | Claude | Codex | Gemini |
|---|---|---|---|
| 역할 | 파인튜닝 진행 여부 최종 결정 | 데이터 준비 파이프라인, SFT 실행 | 파인튜닝 전후 청취 비교 평가 |

**Codex 작업**:
- 단일 화자 데이터 준비 (`prepare_data.py`)
- `sft_12hz.py`로 Base 모델 SFT
- 에포크별 체크포인트 비교

**측정 지표**: 파인튜닝 전후 화자 유사도, WER/CER, 스타일 일관성, 과적합 여부

**기술적 주의사항**:
- LoRA 미지원 — 전체 SFT 비용 큼
- 멀티 스피커 파인튜닝은 아직 제한적
- 데이터 정제 품질이 결과를 좌우함

---

## 백엔드/인프라 고려사항 (향후 서비스화 시 참조)

### 서비스 분리 구조

```
CPU 서비스 풀: MeloTTS, Supertonic2
GPU 서비스 풀: Qwen3-TTS
```

### 요청 라우팅 전략

| 요청 유형 | 담당 모델 |
|---|---|
| preset voice, 고속 처리 | Supertonic2 |
| speed-only, 초경량 | MeloTTS |
| voice cloning | Qwen3-TTS Base |
| streaming, real-time | Qwen3-TTS |
| high-throughput offline batch | Supertonic2 |

### 운영 원칙

- 각 모델별 별도 컨테이너
- warm worker 유지로 cold start 숨김
- 요청 큐 모델별 분리
- 장문은 청킹 큐로 처리

### 관측성 (Observability)

요청 단위 기록 항목: 모델명, 버전, 언어, 텍스트 길이, step/speed/instruct, ref audio 길이, cold start 여부, latency breakdown, GPU/CPU 메모리, 실패 유형 (설치/로딩/OOM/품질 저하)

---

## 모델별 리스크 요약

| 모델 | 주요 리스크 | 완화 방법 |
|---|---|---|
| MeloTTS | 콜드 스타트 느림 | 상시 warm 인스턴스 유지 |
| MeloTTS | 스트리밍 없음 | 장문 → 문장 단위 청킹 |
| Supertonic2 | `onnxruntime` 버전 충돌 | 락파일 + 컨테이너 고정 |
| Supertonic2 | 클로닝/파인튜닝 없음 | 고정 음성 서비스로만 포지셔닝 |
| Qwen3-TTS | GPU/드라이버/메모리 요구 | 0.6B 먼저, 1.7B는 별도 GPU 풀 |
| Qwen3-TTS | ref audio 품질 민감 | 3~10초, 무잡음, 정확한 `ref_text` |
| Qwen3-TTS | 파인튜닝 비용 큼 | 프리셋/클로닝으로 요구사항 먼저 검증 |

---

## 결과물 저장 규칙

- 오디오 출력: `results/{phase}/{model}/{test_id}.wav`
- 벤치마크 JSON (Codex): `results/{phase}/{model}/benchmark.json`
- 청취 평가 Markdown (Gemini): `results/{phase}/{model}/mos_eval.md`
- 단계별 요약: `docs/SUMMARY.md` (각 Phase 완료 시 업데이트)

---

*본 문서는 Claude (CCG — Codex 기술 분석 + Gemini UX/문서 합성)가 작성했습니다.*
*리서치 소스: `docs/research/melotts.md`, `docs/research/supertonic2.md`, `docs/research/qwen3-tts.md`*
