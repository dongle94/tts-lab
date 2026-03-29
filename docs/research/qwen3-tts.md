# Qwen3-TTS 리서치 문서

> 작성일: 2026-03-28
> 목적: Qwen3-TTS 모델 이해 및 커스텀 TTS 활용

---

## 1. 개요

**Qwen3-TTS**는 Alibaba Qwen 팀이 2026년 1월 22일 공개한 오픈소스 다국어 TTS 모델 시리즈다.
라이선스: **Apache 2.0** (상업적 이용 가능)

### 핵심 특징

- 3초 음성으로 **제로샷 보이스 클로닝**
- 자연어 지시문으로 **감정/속도/음색 제어** (숫자 슬라이더 없음)
- 9개 **프리셋 스피커** (한국어 포함)
- **10개 언어** 지원: 한국어, 중국어, 영어, 일본어, 독일어, 프랑스어, 러시아어, 포르투갈어, 스페인어, 이탈리아어
- 스트리밍 생성, 첫 패킷 지연 **97ms** (0.6B 기준)
- 10분 이상 장문 안정 합성

---

## 2. 아키텍처

### 이중 토크나이저 설계

| 토크나이저 | 주파수 | 특징 |
|---|---|---|
| Qwen-TTS-Tokenizer-12Hz | 12.5 Hz, 16-layer multi-codebook | 현재 오픈소스 모델 전부 사용 |
| Qwen-TTS-Tokenizer-25Hz | 단일 코드북, 시맨틱 중심 | 논문에만 언급, 미공개 |

- **LM 백본**: 멀티 코드북 이산 언어 모델 (LM + DiT 캐스케이딩 에러 구조 회피)
- **Multi-Token Prediction**: 멀티 코드북 시퀀스 동시 디코딩

### 3단계 사전학습

1. 500만 시간 이상 다국어 음성 일반 학습
2. 고품질 데이터 정제 (환각 감소)
3. 컨텍스트 확장 8K → 32K (장문 합성)

**사후학습**: DPO + 스피커 파인튜닝

---

## 3. 모델 라인업

| 모델 | 크기 | 용도 |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | 프리셋 스피커 + 자연어 스타일 지시 (추천) |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | 자연어로 새 목소리 디자인 |
| `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | 보이스 클로닝 + 파인튜닝 베이스 |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | 경량 프리셋 스피커 |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | 경량 보이스 클로닝 + 파인튜닝 |
| `Qwen3-TTS-Tokenizer-12Hz` | — | 독립 오디오 인코더/디코더 |

> 자신의 목소리로 TTS → `Base` 모델로 클로닝 또는 파인튜닝
> 감정/스타일 제어 → `CustomVoice` 또는 `VoiceDesign` 모델

---

## 4. 프리셋 스피커 목록

| 이름 | 특징 | 언어 |
|---|---|---|
| Vivian | 밝고 날카로운 젊은 여성 | 중국어 |
| Serena | 따뜻하고 부드러운 젊은 여성 | 중국어 |
| Uncle_Fu | 낮고 묵직한 중년 남성 | 중국어 |
| Dylan | 자연스러운 베이징 남성 청년 | 중국어 (베이징 사투리) |
| Eric | 약간 허스키한 쓰촨 남성 | 중국어 (쓰촨 사투리) |
| Ryan | 리듬감 있는 역동적 남성 | 영어 |
| Aiden | 밝은 미국 남성, 깔끔한 중음 | 영어 |
| Ono_Anna | 경쾌하고 귀여운 일본 여성 | 일본어 |
| **Sohee** | **따뜻하고 감성적인 한국 여성** | **한국어** |

---

## 5. 링크 모음

| 항목 | URL |
|---|---|
| GitHub (공식) | https://github.com/QwenLM/Qwen3-TTS |
| HuggingFace 컬렉션 | https://huggingface.co/collections/Qwen/qwen3-tts |
| 기술 논문 (arXiv) | https://arxiv.org/abs/2601.15621 |
| 온라인 데모 (설치 불필요) | https://huggingface.co/spaces/Qwen/Qwen3-TTS |
| Voice Design 데모 | https://huggingface.co/spaces/Qwen/Qwen3-TTS-Voice-Design |
| DashScope API 문서 | https://www.alibabacloud.com/help/en/model-studio/qwen-tts |

---

## 6. 설치

```bash
# Python 3.12 필수
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

pip install -U qwen-tts soundfile

# FlashAttention2 (GPU 메모리 절약, 선택 — 설치하지 않으면 아래 코드에서 attn_implementation 인자 제거 필요)
pip install -U flash-attn --no-build-isolation
```

> **FlashAttention2 미설치 시**: 아래 코드 예시에서 `attn_implementation="flash_attention_2"` 인자를 제거하면 정상 동작한다.

---

## 7. 사용법 — 3가지 모드

이 문서에서 다루는 주요 사용 패턴은 다음 세 가지다:

| 모드 | 모델 | 한 줄 설명 |
|---|---|---|
| **A. 프리셋 스피커** | `1.7B-CustomVoice` | 9개 내장 목소리 중 선택해서 바로 사용 |
| **B. 지시문으로 스타일 제어** | `1.7B-CustomVoice` | 같은 모델에 `instruct` 파라미터로 감정/속도/톤 지시 |
| **C. 내 목소리 3초 클로닝** | `1.7B-Base` | 레퍼런스 오디오(3초+) 넣으면 그 목소리로 합성 |

> **VoiceDesign** (`1.7B-VoiceDesign`)은 위 세 가지와 별개로, 존재하지 않는 새 목소리를 자연어로 설계하는 모드다 (7-D 참고).

---

### 자연어 지시 (instruct) 작성 가이드

모드 A/B/D 모두 `instruct` 파라미터를 지원한다. 숫자 슬라이더 없이 텍스트로만 스타일을 제어한다.

| 요소 | 예시 |
|---|---|
| 감정 | `"기쁘고 들뜬 목소리로"`, `"슬프고 지친 어조로"`, `"분노한 듯이"` |
| 속도 | `"천천히 또박또박"`, `"빠르게 신나게"`, `"느리게 침착하게"` |
| 피치/음조 | `"낮고 묵직한 목소리로"`, `"높고 밝은 톤으로"` |
| 페르소나 | `"다큐멘터리 내레이터처럼"`, `"친근한 친구처럼"`, `"전문 아나운서처럼"` |
| 복합 지시 | `"처음엔 긴장된 듯 천천히, 후반부에 안도하며 웃음기 있게"` |

**팁:**
- "좋게", "자연스럽게" 같은 추상적 표현보다 **구체적 묘사**가 효과적
- 여러 차원을 **동시에** 지정 가능 (피치 + 속도 + 감정)
- `instruct` 생략 시 스피커의 기본 스타일로 출력됨
- 한국어 지시문으로도 잘 작동함

---

### 모드 A — 프리셋 스피커 (CustomVoice, 지시 없이)

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # flash-attn 미설치 시 이 줄 제거
)

# 프리셋 스피커로 바로 합성 (instruct 생략 = 기본 스타일)
wavs, sr = model.generate_custom_voice(
    text="오늘 날씨가 정말 좋네요!",
    language="Korean",
    speaker="Sohee",
)
sf.write("output_preset.wav", wavs[0], sr)

# 지원 목소리/언어 목록 확인
print(model.get_supported_speakers())
print(model.get_supported_languages())
```

---

### 모드 B — 지시문으로 스타일 제어 (CustomVoice + instruct)

모드 A와 동일한 모델이다. `instruct` 파라미터만 추가하면 된다.

```python
# 단일 문장 — 감정 지시
wavs, sr = model.generate_custom_voice(
    text="오늘 날씨가 정말 좋네요!",
    language="Korean",
    speaker="Sohee",
    instruct="밝고 활기차게, 약간 빠르게",
)
sf.write("output_instruct.wav", wavs[0], sr)

# 배치 — 언어/스피커/지시 각각 지정
wavs, sr = model.generate_custom_voice(
    text=["첫 번째 문장입니다.", "This is the second sentence."],
    language=["Korean", "English"],
    speaker=["Sohee", "Ryan"],
    instruct=["차분하고 낮은 톤으로", "Very excited!"]
)
sf.write("out_ko.wav", wavs[0], sr)
sf.write("out_en.wav", wavs[1], sr)
```

---

### 모드 C — 내 목소리 3초 클로닝 (Base)

> **레퍼런스 오디오 권장 조건**
> - 길이: 3초 이상 (5~10초 권장)
> - 포맷: WAV (16kHz 이상), MP3도 가능
> - 배경 소음 없을수록 좋음
> - `ref_text`에 해당 오디오의 발화 내용을 정확히 전사하면 품질이 크게 향상됨

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # flash-attn 미설치 시 이 줄 제거
)

# 기본 클로닝 (전사 있을 때 — 품질 최상)
wavs, sr = model.generate_voice_clone(
    text="이 목소리로 이 문장을 읽어줘.",
    language="Korean",
    ref_audio="my_voice.wav",       # 로컬 파일 경로 / URL / base64 / (numpy배열, sr) 튜플
    ref_text="레퍼런스 오디오 속 발화 내용을 그대로 적어주세요.",
)
sf.write("cloned.wav", wavs[0], sr)

# 전사 없이도 가능 (품질 다소 낮음)
wavs, sr = model.generate_voice_clone(
    text="클로닝 테스트",
    language="Korean",
    ref_audio="my_voice.wav",
    ref_text="",
    x_vector_only_mode=True,        # 전사 없이 목소리 특성만 추출
)

# 같은 목소리로 여러 문장 생성할 때 — 프롬프트 재사용으로 속도 향상
prompt = model.create_voice_clone_prompt(
    ref_audio="my_voice.wav",
    ref_text="레퍼런스 전사 내용",
)
wavs, sr = model.generate_voice_clone(
    text=["문장 A", "문장 B", "문장 C"],
    language=["Korean", "Korean", "Korean"],
    voice_clone_prompt=prompt,
)
for i, wav in enumerate(wavs):
    sf.write(f"cloned_{i}.wav", wav, sr)
```

---

### 모드 D — 새 목소리 디자인 (VoiceDesign)

존재하지 않는 목소리를 자연어로 묘사해서 생성한다. 프리셋이나 클로닝과 달리 **레퍼런스 없이 완전히 새로운 목소리**를 만드는 모드다.

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # flash-attn 미설치 시 이 줄 제거
)

# instruct로 원하는 목소리 특성을 묘사
wavs, sr = model.generate_voice_design(
    text="안녕하세요. 오늘의 뉴스입니다.",
    language="Korean",
    instruct="30대 여성 아나운서 목소리. 또렷하고 권위 있게, 약간 낮은 피치.",
)
sf.write("designed_voice.wav", wavs[0], sr)
```

---

### 웹 UI 데모 (로컬)

설치 후 브라우저에서 클릭으로 테스트할 수 있다.

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
```

---

## 9. 파인튜닝 (내 목소리 학습)

`Base` 모델로 단일 화자 지도학습 파인튜닝 가능.

### 데이터 준비

```bash
# train_raw.jsonl 형식
{"audio": "data/0001.wav", "text": "안녕하세요.", "ref_audio": "ref/my_ref.wav"}
{"audio": "data/0002.wav", "text": "반갑습니다.", "ref_audio": "ref/my_ref.wav"}
```

```bash
# 오디오 코드 추출
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

### 파인튜닝 실행

```bash
python finetuning/sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ./my_tts_model \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name my_speaker
```

- 에포크마다 체크포인트 저장: `./my_tts_model/checkpoint-epoch-{N}`
- LoRA 지원: 아직 미지원 (계획 중)
- 멀티 스피커 파인튜닝: 계획 중

---

## 10. DashScope 클라우드 API (GPU 없는 경우)

GPU 없이 API로 사용 가능.

```python
import dashscope
from dashscope.audio.tts_v3 import SpeechSynthesizer

result = SpeechSynthesizer.call(
    model="qwen3-tts-instruct-flash",
    text="안녕하세요, 테스트입니다.",
    voice="Sohee",
    speech_rate=1.0,  # 0.5 ~ 2.0
)
with open("api_output.mp3", "wb") as f:
    f.write(result.get_audio_data())
```

| 모델명 | 용도 |
|---|---|
| `qwen3-tts-flash` | 범용, 저비용 |
| `qwen3-tts-instruct-flash` | 감정/스타일 지시 |
| `qwen3-tts-vc-2026-01-22` | 보이스 클로닝 |
| `qwen3-tts-vd-2026-01-26` | 보이스 디자인 |

- 가격: 10,000자당 $0.115
- Rate Limit: 180 RPM
- 스트리밍: PCM 24kHz (`stream: true`, `response_format: "pcm"`)

---

## 11. 성능 비교

| 지표 | Qwen3-TTS | 경쟁 모델 |
|---|---|---|
| 영어 제로샷 WER | **1.24%** (SOTA) | — |
| 다국어 10개 평균 WER | 1.835% | CosyVoice3, MiniMax 대비 우위 |
| 교차 언어 클로닝 (zh→ko) | 4.82 | CosyVoice3: 14.4 (66% 격차) |
| 스트리밍 지연 (0.6B) | **97ms** | — |
| 토크나이저 재구성 PESQ | 3.21 | SOTA |

---

## 12. 추천 시작 경로

```
GPU 없음 → DashScope API 사용
         → https://huggingface.co/spaces/Qwen/Qwen3-TTS 온라인 데모

GPU 있음 → 감정/스타일 제어 원함 → Qwen3-TTS-12Hz-1.7B-CustomVoice
         → 내 목소리 클로닝 원함 → Qwen3-TTS-12Hz-1.7B-Base
         → 새 목소리 설계 원함  → Qwen3-TTS-12Hz-1.7B-VoiceDesign
         → 내 목소리로 완전히 파인튜닝 → 1.7B-Base + finetuning/sft_12hz.py
```

---

*참고: 논문 arXiv:2601.15621, GitHub QwenLM/Qwen3-TTS, HuggingFace Qwen/Qwen3-TTS*
