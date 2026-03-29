# MeloTTS 리서치 문서

> **작성일**: 2026-03-28
> **출처**: GitHub myshell-ai/MeloTTS, HuggingFace myshell-ai, docs.myshell.ai

---

## 1. 개요

**MeloTTS**는 MyShell.ai와 MIT 연구진이 2024년 2월 공개한 오픈소스 다국어 TTS 라이브러리다.
라이선스: **MIT** (상업적 이용 가능)

- **개발**: MyShell.ai (Wenliang Zhao, Xumin Yu — Tsinghua Univ., Zengyi Qin — MIT)
- **논문**: 없음 (소프트웨어로만 공개, arXiv 미등재)
- **기반 아키텍처**: VITS → VITS2 → Bert-VITS2 계보 (비자기회귀 방식)

### 핵심 특징

- **CPU 실시간 추론** — GPU 없이 사용 가능
- **7개 언어 지원** — 영어(5가지 액센트), 한국어, 중국어, 일본어, 스페인어, 프랑스어
- **속도 제어 가능** — 유일하게 노출된 프로소디 파라미터
- **단일 forward pass** — 전체 오디오를 한 번에 생성 (스트리밍 없음)
- **보이스 클로닝 없음** — 고정 프리셋 화자만 지원

> **MeloTTS vs OpenVoice**: 같은 팀 제품. MeloTTS는 TTS 품질/속도 최적화, OpenVoice v2는 보이스 클로닝 담당.

---

## 2. 아키텍처

**VITS 계열 비자기회귀(Non-autoregressive) 모델**

| 컴포넌트 | 역할 |
|---|---|
| Text Encoder | 음소 + 성조 처리, BERT 임베딩 통합 |
| Stochastic Duration Predictor | 발화 타이밍 결정 |
| Flow Module | 잠재 표현 변환 |
| HiFi-GAN Vocoder | 최종 오디오 파형 합성 |

**텍스트 처리 파이프라인:**
1. 문장 분리 (언어별 규칙)
2. G2P — 음소 변환 (`g2p_en`, `pypinyin`, `jamo` 등 언어별)
3. 성조 추출 (중국어 등 성조 언어)
4. BERT 임베딩 생성 (컨텍스트 기반 프로소디)
5. 추론

**사용 BERT 모델**: `bert-base-uncased` (영어), `bert-base-multilingual-uncased` (다국어)

---

## 3. 모델 체크포인트

언어별로 별도 체크포인트 (~208 MB)가 제공된다.

| HuggingFace 저장소 | 언어 코드 | 화자 수 |
|---|---|---|
| `myshell-ai/MeloTTS-English` | `EN` | 5 (미국/영국/인도/호주/기본) |
| `myshell-ai/MeloTTS-English-v2` | `EN_V2` | 5 |
| `myshell-ai/MeloTTS-English-v3` | `EN_NEWEST` | 1 |
| `myshell-ai/MeloTTS-Korean` | `KR` | 1 |
| `myshell-ai/MeloTTS-Chinese` | `ZH` | 1 |
| `myshell-ai/MeloTTS-Japanese` | `JP` | 1 |
| `myshell-ai/MeloTTS-French` | `FR` | 1 |

> 영어 이외 언어는 모두 단일 화자다. 한국어도 `KR` 하나뿐.

---

## 4. 링크 모음

| 항목 | URL |
|---|---|
| GitHub (공식) | https://github.com/myshell-ai/MeloTTS |
| HuggingFace — 한국어 모델 | https://huggingface.co/myshell-ai/MeloTTS-Korean |
| HuggingFace — 영어 모델 | https://huggingface.co/myshell-ai/MeloTTS-English |
| 온라인 데모 (설치 불필요) | https://huggingface.co/spaces/mrfakename/MeloTTS |
| 공식 기술 문서 | https://docs.myshell.ai/technology/melotts |
| 설치 가이드 (공식) | https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md |
| 학습 가이드 (공식) | https://github.com/myshell-ai/MeloTTS/blob/main/docs/training.md |

---

## 5. 설치

> **Windows는 공식 미지원** — Docker를 사용하거나 WSL에서 실행.
> **macOS**에서 설치 오류 시 Docker 방법 사용.

### 방법 1: 소스 설치 (Linux / WSL 권장)

```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS

pip install -e .
python -m unidic download  # 일본어 형태소 분석기 (전 언어 사용 시 필요)
```

### 방법 2: Docker (Windows / macOS 권장)

```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
docker build -t melotts .

# CPU 실행
docker run -it -p 8888:8888 melotts

# GPU 실행
docker run --gpus all -it -p 8888:8888 melotts
```

실행 후 브라우저에서 `http://localhost:8888` 접속 → WebUI 사용 가능.

---

## 6. 사용법

### 6.1 Python API

```python
from melo.api import TTS

# 모델 로드 — 최초 실행 시 체크포인트 자동 다운로드 (~208 MB)
model = TTS(language='KR', device='auto')
# device: 'auto' | 'cpu' | 'cuda' | 'cuda:0' | 'mps'

# 사용 가능한 화자 ID 확인
speaker_ids = model.hps.data.spk2id
print(speaker_ids)  # {'KR': 0}

# 합성
model.tts_to_file(
    text="안녕하세요! 오늘은 날씨가 정말 좋네요.",
    speaker_id=speaker_ids['KR'],
    output_path='output.wav',
    speed=1.0,   # 0.5~2.0 권장. 1.0 = 기본 속도
)
```

**영어 — 액센트별 화자 선택:**

```python
model = TTS(language='EN', device='auto')
speaker_ids = model.hps.data.spk2id
# speaker_ids 목록: 'EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU', 'EN-Default'

model.tts_to_file("Did you ever hear a folk tale about a giant turtle?",
                  speaker_ids['EN-US'], 'en_us.wav', speed=1.0)
model.tts_to_file("Did you ever hear a folk tale about a giant turtle?",
                  speaker_ids['EN-BR'], 'en_br.wav', speed=1.0)
```

**중국어 — 중영 혼용 지원:**

```python
model = TTS(language='ZH', device='auto')
speaker_ids = model.hps.data.spk2id
model.tts_to_file("text-to-speech 领域近年来发展迅速",
                  speaker_ids['ZH'], 'zh.wav', speed=1.0)
```

> **주의**: 모델 로드 시 BERT 로딩으로 **콜드 스타트가 느릴 수 있다** (첫 실행 10~30초). 한 번 로드 후 같은 인스턴스로 반복 호출하면 빠르다.

### 6.2 CLI

설치 후 `melo` / `melotts` 명령어가 등록된다.

```bash
# 기본 사용
melo "안녕하세요" output.wav --language KR

# 언어 + 화자 지정
melo "Hello, world!" output.wav --language EN --speaker EN-US

# 속도 조절
melo "Hello, world!" output.wav --language EN --speaker EN-BR --speed 1.3

# 중국어
melo "text-to-speech 领域近年来发展迅速" zh.wav -l ZH

# 텍스트 파일에서 읽기
melo input.txt output.wav --language KR --file

# 도움말
melo --help
```

### 6.3 WebUI

Docker 실행 시 자동으로 `http://localhost:8888`에서 Gradio UI 제공.
설치 불필요한 온라인 데모: https://huggingface.co/spaces/mrfakename/MeloTTS

---

## 7. 언어별 화자 ID 전체 목록

| 언어 코드 | 사용 가능한 speaker_ids |
|---|---|
| `EN` | `EN-US`, `EN-BR`, `EN_INDIA`, `EN-AU`, `EN-Default` |
| `EN_V2` | `EN-US`, `EN-BR`, `EN_INDIA`, `EN-AU`, `EN-Default` |
| `EN_NEWEST` | `EN-Newest` |
| `KR` | `KR` |
| `ZH` | `ZH` |
| `JP` | `JP` |
| `ES` | `ES` |
| `FR` | `FR` |

---

## 8. 제어 가능한 파라미터

| 파라미터 | 방법 | 비고 |
|---|---|---|
| 언어 | `TTS(language=...)` | 모델 자체가 언어별로 분리됨 |
| 화자/액센트 | `speaker_ids['EN-US']` 등 | 영어만 다중 화자, 나머지 단일 |
| 속도 | `speed=1.0` (float) | 유일한 프로소디 제어 수단 |
| 디바이스 | `TTS(..., device='auto')` | auto/cpu/cuda/mps |

**지원하지 않는 제어**: 감정, 피치, 에너지, 스타일 — MeloTTS에서는 속도만 제어 가능.

---

## 9. 커스텀 보이스 파인튜닝

`Base` 모델 없이 처음부터 파인튜닝. 공식 `docs/training.md` 기준.

### 데이터 준비

오디오는 **44100 Hz WAV** 필수. 메타데이터는 파이프 구분:

```
data/audio_001.wav|내_화자명|KR|안녕하세요.
data/audio_002.wav|내_화자명|KR|오늘 날씨가 좋네요.
```

전사 텍스트가 없으면 Whisper 등 ASR로 생성.

### 학습 실행

```bash
# 환경 설정
pip install -e .
cd melo

# 메타데이터 전처리 (config.json, train.list 등 자동 생성)
python preprocess_text.py --metadata data/example/metadata.list

# 학습 (GPU 수 지정)
bash train.sh data/example/config.json 1

# 추론 (학습된 체크포인트로)
python infer.py \
  --text "학습된 목소리로 읽어줘" \
  -m /path/to/checkpoint/G_1000.pth \
  -o output_dir/
```

> 메모리 부족(OOM) 시 `config.json`의 `batch_size` 줄이기.
> `train.sh`는 gloo 충돌 복구를 위한 자동 재시작 포함.

**한계**: One-shot/Few-shot 클로닝 없음 — 충분한 학습 데이터 필요. LoRA 미지원.

---

## 10. 알려진 한계

| 항목 | 내용 |
|---|---|
| 보이스 클로닝 없음 | 고정 프리셋만 지원. 클로닝은 OpenVoice v2 사용 |
| 한국어 화자 1개 | `KR` 단일 화자, 다양성 없음 |
| 감정/스타일 제어 없음 | 속도만 제어 가능 |
| 스트리밍 없음 | 전체 텍스트를 한 번에 처리 후 출력 |
| 콜드 스타트 느림 | BERT 로드 시간 문제 — 한 번 로드 후 재사용 권장 |
| Windows 미지원 | Docker 또는 WSL 필요 |
| 한국어-영어 혼용 미지원 | 중국어만 혼용 공식 지원 |
| 논문 없음 | 아키텍처 학술 문서 미존재 |

---

## 11. 다른 TTS 모델과 비교

| 모델 | 속도 | 한국어 | 보이스 클로닝 | 감정 제어 | GPU 필요 |
|---|---|---|---|---|---|
| **MeloTTS** | 빠름 (CPU 실시간) | 단일 화자 | 없음 | 없음 (속도만) | 불필요 |
| Supertonic2 | 매우 빠름 (ONNX) | 단일 화자 | 없음 | 없음 | 불필요 |
| Qwen3-TTS | 보통 | 프리셋 1개(Sohee) + 클로닝 | 있음 (3초) | 자연어 지시 | 필요 |
| F5-TTS | 보통 | 제한적 | 있음 (zero-shot) | 없음 | 권장 |
| XTTS-v2 | 보통 | 있음 | 있음 | 없음 | 권장 |

**MeloTTS 포지셔닝**: GPU 없이 빠르게 돌아가는 한국어 TTS가 필요할 때. 음성 다양성이나 스타일 제어가 필요하면 Qwen3-TTS 또는 Supertonic2+VoiceBuilder 병행 검토.

---

## 12. 실전 팁

- **콜드 스타트 대응**: 프로세스 시작 시 한 번만 `TTS()` 인스턴스 생성, 이후 `tts_to_file()` 반복 호출
- **장문 처리**: 문장부호(`.`, `?`, `!`) 기준으로 청킹 후 순차 합성 → 더 안정적인 발음
- **속도 범위**: `0.8`(천천히) ~ `1.3`(빠르게) 구간이 자연스러움. 극단값은 품질 저하
- **한국어 품질**: 표준어/뉴스 문체에서 품질이 좋음. 구어체나 은어는 다소 어색할 수 있음
- **`device='auto'`**: GPU가 있으면 자동으로 CUDA 사용, 없으면 CPU로 폴백

---

*참고: GitHub myshell-ai/MeloTTS, HuggingFace myshell-ai/MeloTTS-Korean, docs.myshell.ai/technology/melotts*
