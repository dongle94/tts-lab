# Supertonic2 (SupertonicTTS) 리서치 문서

> **작성일**: 2026-03-28
> **출처**: arXiv:2503.23108, GitHub supertone-inc/supertonic, HuggingFace Supertone/supertonic-2

---

## 1. 개요

Supertonic2는 Supertone Inc.가 개발한 **온디바이스 다국어 TTS 시스템**이다. 66M 파라미터 규모의 모델로, ONNX Runtime을 통해 클라우드 없이 로컬에서 실행된다. 논문명은 *"SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System"* (arXiv:2503.23108).

### 버전 히스토리

| 버전 | 출시 | 파라미터 | 언어 |
|---|---|---|---|
| Supertonic 1 | 2025-11 | 44M | 영어 전용 |
| Supertonic 2 | 2026-01-06 | 66M | EN / KO / ES / PT / FR |

---

## 2. 아키텍처

**Flow-matching 기반** (Diffusion/VITS/AR 계열 아님)

### 2.1 Speech Autoencoder

- **Encoder**: Vocos 기반, 10개 ConvNeXt 블록
  - 228-dim 멜 스펙트로그램 → 24-dim 연속 잠재 공간
  - 커널 크기 7, 중간 차원 2048
- **Decoder**: 10개 dilated ConvNeXt 블록 (causal convolution)
  - 44.1 kHz 웨이브폼 직접 출력 (배포 ONNX는 24 kHz)
- **학습**: GAN 프레임워크 (reconstruction + adversarial + feature-matching loss)

### 2.2 Text-to-Latent (TTL) 모듈

- **알고리즘**: Flow-matching (추론 시 Euler method)
- **Reference Encoder**: 6개 ConvNeXt 블록 + 2개 cross-attention 레이어
  - 압축된 잠재 표현(voice style) 처리
- **Text Encoder**: 128-dim 캐릭터 임베딩 + 6개 ConvNeXt 블록 + 4개 self-attention + 2개 cross-attention
  - 캐릭터 레벨 입력 — **G2P 모듈 불필요**
- **Vector Field Estimator**: dilated ConvNeXt 블록 × 4 (dilation 1,2,4,8) + 표준 블록 × 2, 4회 반복
  - 시간, 텍스트, 레퍼런스로 조건화

### 2.3 Duration Predictor

- ~0.5M 파라미터
- 텍스트 + 레퍼런스 임베딩 연결 → 총 발화 시간 스칼라 예측

### 2.4 ONNX 배포 파이프라인

```
Duration Predictor → Text Encoder → Vector Estimator → Vocoder
```

스타일 벡터: `style_ttl` (음색) + `style_dp` (템포)

### 2.5 핵심 기술 혁신

1. **저차원 잠재 공간 + 시간 압축** (Kc=6): flow-matching 연산량 절감
2. **ConvNeXt 일관 사용**: Transformer 중심 설계 대비 빠른 추론
3. **캐릭터 레벨 입력**: G2P/음소변환 없이 유니코드 직접 입력
4. **Context-sharing batch expansion** (Ke=4): 동일 텍스트-음성 쌍을 여러 노이즈/타임스텝으로 반복 제시 → 메모리 효율 + 수렴 개선
5. **Length-Aware RoPE** (arXiv:2509.11084): 텍스트-음성 정렬 개선
6. **Self-Purifying Flow Matching** (arXiv:2509.19091): 노이즈 레이블 학습 지원

---

## 3. 학습 데이터

| 컴포넌트 | 데이터셋 | 규모 |
|---|---|---|
| Speech Autoencoder | 공개 + 내부 데이터 | 11,167시간, ~14,000 화자 |
| TTL + Duration Predictor | LJSpeech, VCTK, Hi-Fi TTS, LibriTTS | 945시간, 2,576 화자 (영어 전용) |

**샘플레이트**: 44.1 kHz (연구 모델) / 24 kHz (배포 ONNX)
**학습 하드웨어**: 4× NVIDIA RTX 4090

---

## 4. 성능 벤치마크

### 4.1 추론 속도 (2-step, characters/sec)

| 하드웨어 | 단문 (59자) | 중문 (152자) | 장문 (266자) | RTF |
|---|---|---|---|---|
| M4 Pro CPU | 912 | 1,048 | 1,263 | 0.012–0.015 |
| M4 Pro WebGPU | 996 | 1,801 | 2,509 | 0.006–0.014 |
| RTX 4090 | 2,615 | 6,548 | **12,164** | **0.001–0.005** |
| ElevenLabs Flash v2.5 | 144–287 | — | — | 0.057–0.133 |
| Kokoro | 104–117 | — | — | — |
| OpenAI TTS-1 | 37–82 | — | — | 0.201–0.471 |
| Gemini 2.5 Flash TTS | 12–24 | — | — | — |

### 4.2 품질 (LibriSpeech test-clean, zero-shot TTS)

| 지표 | Supertonic | VALL-E (410M) | DiTTo-TTS (940M) | F5-TTS (349M) |
|---|---|---|---|---|
| WER | **2.64%** | 5.9% | 2.56% | 2.42% |
| CER | **0.83%** | — | — | — |
| 파라미터 | **44M** | 410M | 940M | 349M |
| RTF (RTX4090) | **0.02** | — | — | 0.31 |

### 4.3 음성 복원 품질 (LibriTTS-clean)

| 지표 | SupertonicTTS | BigVGAN |
|---|---|---|
| NISQA | 4.06 | 4.11 |
| UTMOSv2 | 3.13 | 3.16 |
| V/UV F1 | 0.9587 | 0.9735 |

---

## 5. 설치 및 환경

### 5.1 저장소 정보

- **GitHub**: https://github.com/supertone-inc/supertonic
- **HuggingFace 모델**: https://huggingface.co/Supertone/supertonic-2
- **PyPI**: https://pypi.org/project/supertonic/ (v1.1.2)
- **Python API 문서**: https://supertone-inc.github.io/supertonic-py/

### 5.2 요구 사항

- Python 3.9 – 3.12
- `onnxruntime==1.23.1` (버전 고정 필수)
- 저장소 용량: ~305 MB (ONNX 가중치)
- GPU 불필요 — CPU 추론이 기본/권장 방식

### 5.3 설치 방법

두 가지 방법이 있으며 **사용 방식이 다르다** — Python API를 쓸 것인지, CLI 스크립트를 직접 실행할 것인지에 따라 선택한다.

#### 방법 1: PyPI — Python API 사용 시 (권장)

```bash
pip install supertonic
```

자동으로 모델 가중치 다운로드. 이후 섹션 6.1의 Python 코드로 바로 사용 가능.

#### 방법 2: 저장소 클론 — CLI 스크립트 / 다른 언어 런타임 사용 시

```bash
git clone https://github.com/supertone-inc/supertonic.git
cd supertonic

# ONNX 가중치 다운로드 (Git LFS 필요)
git clone https://huggingface.co/Supertone/supertonic-2 assets

# Python 환경 세팅 (uv는 빠른 패키지 관리자 — pip 대체)
cd py
uv sync          # uv 미설치 시: pip install -r requirements.txt
uv run example_onnx.py  # uv 미설치 시: python example_onnx.py
```

> `uv` 설치: `pip install uv` 또는 `curl -Ls https://astral.sh/uv/install.sh | sh`

#### 저장소 구조

```
supertonic/
├── assets/         # ONNX 모델 + voice style JSON (HF에서 다운로드)
├── py/             # Python + ONNX Runtime
├── nodejs/         # Node.js
├── web/            # 브라우저 (WebGPU / WASM)
├── java/           # JVM
├── cpp/            # C++
├── csharp/         # .NET
├── go/             # Go
├── swift/          # macOS
├── ios/            # iOS
├── rust/           # Rust
└── flutter/        # Dart/Flutter
```

---

## 6. 사용법

### 6.1 Python API (방법 1 — PyPI 설치 기반)

```python
from supertonic import TTS

# 초기화 (최초 실행 시 ~305 MB 자동 다운로드)
tts = TTS(auto_download=True)

# 음성 스타일 선택 (M1–M5: 남성, F1–F5: 여성)
style = tts.get_voice_style(voice_name="F2")  # F2가 표현력 최고 평가

# 합성 — lang 코드: "en", "ko", "es", "pt", "fr"
wav, duration = tts.synthesize("안녕하세요, 수퍼토닉입니다.", voice_style=style, lang="ko")
# wav: (1, num_samples) numpy array

# 저장
tts.save_audio(wav, "output.wav")
```

### 6.2 CLI (방법 2 — 저장소 클론 기반)

> Voice Style JSON 파일은 저장소 클론 후 생성된 `assets/` 폴더 안에 있다 (`F1.json` ~ `F5.json`, `M1.json` ~ `M5.json`).

```bash
# 기본 실행 (assets/ 폴더 내 기본 스타일로 실행)
uv run example_onnx.py  # uv 미설치 시: python example_onnx.py

# 파라미터 지정
uv run example_onnx.py \
  --voice-style ../assets/F2.json \
  --text "Hello, this is Supertonic." \
  --lang en \
  --total-step 5 \
  --speed 1.05 \
  --save-dir results

# 배치 모드 (여러 화자/텍스트 동시)
uv run example_onnx.py \
  --voice-style ../assets/M1.json ../assets/M2.json \
  --text "Hello." "Hola." \
  --lang en es \
  --batch
```

#### CLI 파라미터 레퍼런스

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--voice-style` | M1.json | 음성 프로파일 JSON 경로 (`assets/` 폴더 안) |
| `--text` | — | 합성할 텍스트 |
| `--lang` | en | 언어: `en`, `ko`, `es`, `pt`, `fr` |
| `--total-step` | 5 | 디노이징 반복 수 — 2 (최고속), 5 (균형), 10 (최고 품질) |
| `--speed` | 1.05 | 발화 속도 (0.9–1.5 권장) |
| `--save-dir` | results | WAV 출력 디렉터리 |
| `--batch` | off | 청킹 없이 배치 추론 |
| `--use-gpu` | off | GPU 실행 (미지원 상태) |

### 6.3 추론 파이프라인 내부 흐름

```
Raw text
  → UnicodeProcessor (정규화, 언어 태그, char→uint16)
  → Duration Predictor ONNX (총 프레임 수 예측)
  → Text Encoder ONNX (ConvNeXt + self-attention)
  → Vector Estimator ONNX (반복 디노이징, Euler method)
  → Vocoder ONNX (잠재 → 44.1 kHz 웨이브폼)
  → soundfile WAV (16-bit)
```

---

## 7. 지원 언어 및 음성 스타일

| 언어 | 코드 |
|---|---|
| 영어 | `en` |
| 한국어 | `ko` |
| 스페인어 | `es` |
| 포르투갈어 | `pt` |
| 프랑스어 | `fr` |

**빌트인 음성**: M1–M5 (남성), F1–F5 (여성) — 총 10개

**커스텀 음성**: [Voice Builder](https://supertonic.supertone.ai/voice_builder) 웹 툴로 제작 가능

Voice Builder 사용 흐름:
1. 위 URL에서 레퍼런스 오디오 업로드
2. 커스텀 스타일 JSON 파일 다운로드
3. 코드에서 해당 JSON을 `--voice-style` 또는 `get_voice_style(path="...")` 경로로 지정

```python
# 커스텀 스타일 JSON 적용 예시 (Python API)
style = tts.get_voice_style(path="my_custom_voice.json")
wav, _ = tts.synthesize("텍스트", voice_style=style, lang="ko")
```

---

## 8. 라이선스 및 상업적 활용

| 구성 요소 | 라이선스 | 상업적 활용 |
|---|---|---|
| 코드 (GitHub) | MIT | 자유롭게 가능 |
| 모델 가중치 (HuggingFace) | OpenRAIL-M | 가능 (특정 유해 사용 금지 조항 있음) |

OpenRAIL-M은 상업적 사용은 허용하되, 딥페이크/허위 정보 생성 등 특정 유해 목적 사용을 금지한다. 프로덕션 배포 전 라이선스 전문 확인 필요.

---

## 9. 다른 TTS 모델과의 비교

| 모델 | 속도 | 보이스 클로닝 | 파인튜닝 | 품질 | 온디바이스 |
|---|---|---|---|---|---|
| **Supertonic2** | **최고** (12,164 c/s) | 없음 | 없음 | 높음 (특히 운율) | ONNX |
| Kokoro-82M | 빠름 (104–117 c/s) | 없음 | 커뮤니티 LoRA | 높음 | 가능 |
| F5-TTS | 보통 | **있음** (zero-shot) | 있음 | 매우 높음 | 가능 |
| StyleTTS2 | 보통 | 제한적 | 있음 | 매우 높음 | 가능 |
| CoquiTTS XTTS v2 | 보통 | **있음** | 있음 | 높음 | 가능 |
| ElevenLabs Flash v2.5 | 144–287 c/s (클라우드) | 있음 | 없음 (비공개) | 매우 높음 | 없음 |

**핵심 포지셔닝**: Supertonic2는 **속도 최우선** 시나리오에 최적. 커스텀 음성·파인튜닝이 필요하면 F5-TTS 또는 CoquiTTS XTTS v2가 적합.

---

## 10. 커뮤니티 평가

### 장점 (HuggingFace 디스커션 기준)

- "Kokoro 이후 가장 신뢰할 수 있는 모델" (`sharadcodes`)
- "F2 음성은 억양 표현이 탁월하고 장문에서도 일관성 유지" (`ken107`)
- "iOS eBook 리더에서 ONNX로 100% 오프라인 실행, 낮은 레이턴시" (`harim95`)
- 한국어 품질 특히 우수 (Supertone 본사 한국 기업, 발음·억양 최적화)

### 단점

- **보이스 클로닝 없음**: 고정 10개 프리셋만 지원, 화자 적응 불가
- **파인튜닝 파이프라인 미공개**: 커스텀 음성 추가 불가 (GitHub issue #64)
- **프랑스어**: 장문에서 품질 저하 보고 있음
- GPU 모드 코드는 존재하나 공식적으로 미지원 상태

---

## 11. 실전 활용 팁

- **속도 vs 품질 트레이드오프**: `--total-step 2` (최고속), `--total-step 5` (균형), `--total-step 10` (최고 품질)
- **추천 음성**: F2가 표현력·운율 면에서 가장 호평
- **장문 처리**: 문장부호 기준 ~100자 단위로 청킹 후 큐 처리 → 거의 실시간 성능
- **프로덕션 API**: 커뮤니티 FastAPI 래퍼 (`Deveraux-Parker/supertonic-2-afterburner-fastAPI`) 활용 시 245 req/s, p50 284ms
- **브라우저 배포**: WebGPU 지원 브라우저에서 WASM 폴백으로 서버 없이 동작

---

## 12. 관련 링크

| 자료 | URL |
|---|---|
| arXiv 논문 | https://arxiv.org/abs/2503.23108 |
| GitHub 저장소 | https://github.com/supertone-inc/supertonic |
| HuggingFace 모델 | https://huggingface.co/Supertone/supertonic-2 |
| HuggingFace 데모 | https://huggingface.co/spaces/Supertone/supertonic-2 |
| PyPI 패키지 | https://pypi.org/project/supertonic/ |
| Python API 문서 | https://supertone-inc.github.io/supertonic-py/ |
| 동반 논문 (RoPE) | https://arxiv.org/abs/2509.11084 |
| 동반 논문 (Flow) | https://arxiv.org/abs/2509.19091 |
| FastAPI 래퍼 | https://github.com/Deveraux-Parker/supertonic-2-afterburner-fastAPI |
