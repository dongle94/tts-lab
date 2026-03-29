#!/usr/bin/env python3
"""Phase 0 common benchmark harness for MeloTTS, Supertonic2, and Qwen3-TTS.

The script is intentionally self-contained:
- benchmarks one model per process so cold-start and memory are isolated
- normalizes every output to a common sample rate before saving
- writes results to results/{phase}/{model}/benchmark.json

The benchmark schema is stable across models so results can be compared directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is required. Activate the model uv environment first, for example: "
        "source envs/supertonic2/bin/activate"
    ) from exc


TARGET_SAMPLE_RATE = 24_000
DEFAULT_PHASE = "phase0"
DEFAULT_MODEL = "all"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PYTHONS = {
    "melotts": PROJECT_ROOT / "envs" / "melotts" / "bin" / "python",
    "supertonic2": PROJECT_ROOT / "envs" / "supertonic2" / "bin" / "python",
    "qwen3-tts": PROJECT_ROOT / "envs" / "qwen3-tts" / "bin" / "python",
}
BASE_TEXT_CASES = [
    {
        "id": "KO_S_01",
        "set": "base",
        "name": "ko_s_01",
        "text": "안녕하세요. 만나서 반갑습니다.",
    },
    {
        "id": "KO_L_01",
        "set": "base",
        "name": "ko_l_01",
        "text": (
            "인공지능 기술의 발전은 우리 사회 전반에 걸쳐 유례없는 변화를 불러오고 있으며, "
            "특히 음성 합성 기술은 인간과 기계 사이의 소통을 더욱 자연스럽게 만들어주고 있습니다."
        ),
    },
    {
        "id": "KO_N_01",
        "set": "base",
        "name": "ko_n_01",
        "text": "2026년 3월 28일 오후 2시 30분, 온도는 22.5도입니다. (문의: 02-1234-5678)",
    },
    {
        "id": "KO_C_01",
        "set": "base",
        "name": "ko_c_01",
        "text": "야, 너 어제 그 뉴스 봤어? 진짜 대박이더라! 말도 안 돼.",
    },
    {
        "id": "EN_S_01",
        "set": "base",
        "name": "en_s_01",
        "text": "Hello, it's a pleasure to meet you.",
    },
    {
        "id": "EN_L_01",
        "set": "base",
        "name": "en_l_01",
        "text": (
            "Large language models have demonstrated remarkable capabilities in understanding "
            "and generating human-like text across a wide range of topics and styles."
        ),
    },
    {
        "id": "MIX_01",
        "set": "base",
        "name": "mix_01",
        "text": "오늘 점심은 햄버거 세트랑 아이스 아메리카노 어때?",
    },
    {
        "id": "MIX_02",
        "set": "base",
        "name": "mix_02",
        "text": "이번 프로젝트의 핵심은 Efficiency와 Scalability를 확보하는 것입니다.",
    },
]
TEST_SET_CASES = {
    "base": BASE_TEXT_CASES,
    "SPD": [
        {
            "id": "SPD_01",
            "set": "SPD",
            "name": "spd_01",
            "text": "긴급 상황입니다! 모두 신속하게 대피해 주시기 바랍니다.",
        },
        {
            "id": "SPD_02",
            "set": "SPD",
            "name": "spd_02",
            "text": "아주 천천히, 한 글자씩 또박또박 읽어보겠습니다.",
        },
    ],
    "LNG": [
        {
            "id": "LNG_01",
            "set": "LNG",
            "name": "lng_01",
            "text": (
                "인공지능(AI)이 인간의 목소리를 완벽하게 흉내 내는 시대가 도래했습니다. "
                "과거의 기계적인 음성과는 달리, 최신 모델들은 감정과 뉘앙스까지 담아낼 수 있게 되었습니다. "
                "이는 고객 서비스, 오디오북 제작, 그리고 게임 산업 등 다양한 분야에서 혁신을 일으키고 있습니다. "
                "하지만 기술의 발전과 함께 딥페이크 음성을 악용한 범죄 우려도 커지고 있어, "
                "이에 대한 윤리적 가이드라인과 기술적 방어책 마련이 시급한 상황입니다."
            ),
        }
    ],
    "STY": [
        {
            "id": "STY_01",
            "set": "STY",
            "name": "sty_01",
            "text": "정말 고마워! 네 덕분에 살았어.",
        },
        {
            "id": "STY_02",
            "set": "STY",
            "name": "sty_02",
            "text": "미안해, 내가 다 잘못했어. 한 번만 용서해 줘.",
        },
        {
            "id": "STY_03",
            "set": "STY",
            "name": "sty_03",
            "text": "조용히 해! 지금 공부하는 중이잖아.",
        },
        {
            "id": "STY_04",
            "set": "STY",
            "name": "sty_04",
            "text": "옛날 옛적 어느 마을에, 아주 착한 나무꾼이 살고 있었어요.",
        },
    ],
    "CLN": [
        {
            "id": "CLN_01",
            "set": "CLN",
            "name": "cln_01",
            "text": "오늘의 주요 뉴스를 전해드립니다. 정부는 오늘 새로운 경제 정책을 발표했습니다.",
            "requires_reference_audio": True,
            "note": "차분한 여성 아나운서 톤",
        },
        {
            "id": "CLN_02",
            "set": "CLN",
            "name": "cln_02",
            "text": "와! 저기 봐! 엄청 큰 비행기가 날아가고 있어!",
            "requires_reference_audio": True,
            "note": "장난기 넘치는 남성 아이 톤",
        },
    ],
}
DEFAULT_WARMUP_TEXT = "벤치마크 워밍업 문장입니다."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 common TTS benchmark harness")
    parser.add_argument("--phase", default=DEFAULT_PHASE, help="Phase name used in results path")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["all", "melotts", "supertonic2", "qwen3-tts"],
        help="Model to benchmark, or 'all' for every supported model",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory for benchmark outputs",
    )
    parser.add_argument(
        "--python-melotts",
        type=Path,
        default=DEFAULT_MODEL_PYTHONS["melotts"],
        help="Python executable for MeloTTS worker process",
    )
    parser.add_argument(
        "--python-supertonic2",
        type=Path,
        default=DEFAULT_MODEL_PYTHONS["supertonic2"],
        help="Python executable for Supertonic2 worker process",
    )
    parser.add_argument(
        "--python-qwen3",
        type=Path,
        default=DEFAULT_MODEL_PYTHONS["qwen3-tts"],
        help="Python executable for Qwen3-TTS worker process",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal mode used for per-model isolated execution",
    )
    parser.add_argument(
        "--text-case",
        action="append",
        nargs=2,
        metavar=("NAME", "TEXT"),
        help="Additional benchmark case; may be specified multiple times",
    )
    parser.add_argument(
        "--test-set",
        action="append",
        choices=["base", "SPD", "LNG", "STY", "CLN", "all"],
        help="Select one or more test sets from results/phase0/test_scripts.md",
    )
    parser.add_argument(
        "--warmup-text",
        default=DEFAULT_WARMUP_TEXT,
        help="Warmup text used to trigger cold-start synthesis",
    )
    return parser.parse_args()


def get_process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    try:
        import resource  # type: ignore

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss * 1024)
    except Exception:
        return 0


class MemorySampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_rss_bytes = 0
        self._peak_gpu_allocated_bytes = 0
        self._peak_gpu_reserved_bytes = 0

    def start(self) -> None:
        self._peak_rss_bytes = max(self._peak_rss_bytes, get_process_rss_bytes())
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(0.05)

    def _sample(self) -> None:
        self._peak_rss_bytes = max(self._peak_rss_bytes, get_process_rss_bytes())
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                self._peak_gpu_allocated_bytes = max(
                    self._peak_gpu_allocated_bytes,
                    int(torch.cuda.max_memory_allocated()),
                )
                self._peak_gpu_reserved_bytes = max(
                    self._peak_gpu_reserved_bytes,
                    int(torch.cuda.max_memory_reserved()),
                )
        except Exception:
            pass

    def stop(self) -> dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()
        return {
            "rss_peak_bytes": int(self._peak_rss_bytes),
            "gpu_allocated_peak_bytes": int(self._peak_gpu_allocated_bytes),
            "gpu_reserved_peak_bytes": int(self._peak_gpu_reserved_bytes),
        }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_float32_mono(audio: Any) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr.mean(axis=0)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 1.5:
        arr = arr / max_abs
    return np.clip(arr, -1.0, 1.0)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if audio.size == 0 or source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")

    source_duration = audio.shape[0] / float(source_sr)
    target_length = max(1, int(round(source_duration * target_sr)))
    source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False, dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False, dtype=np.float64)
    resampled = np.interp(target_positions, source_positions, audio.astype(np.float64))
    return resampled.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    ensure_dir(path.parent)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        dtype = np.int16
        scale = 32768.0
    elif sample_width == 1:
        dtype = np.uint8
        scale = 128.0
    else:
        raise ValueError(f"unsupported wav sample width: {sample_width}")

    arr = np.frombuffer(frames, dtype=dtype)
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    if sample_width == 1:
        arr = arr.astype(np.float32) - 128.0
    audio = arr.astype(np.float32) / scale
    return audio, sample_rate


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return audio.shape[0] / float(sample_rate)


def chars_per_second(text: str, generation_time_sec: float) -> float:
    if generation_time_sec <= 0:
        return 0.0
    return len(text) / generation_time_sec


@dataclasses.dataclass
class BenchResult:
    name: str
    text: str
    text_chars: int
    kind: str
    lang: str
    voice_name: Optional[str]
    speed: Optional[float]
    total_step: Optional[int]
    chunking: bool
    chunk_count: int
    source_sample_rate: int
    normalized_sample_rate: int
    output_audio_path: str
    generation_time_sec: float
    audio_duration_sec: float
    rtf: float
    chars_per_sec: float


class BaseAdapter:
    model_name: str
    display_name: str
    target_device: str

    def load(self) -> None:
        raise NotImplementedError

    def synthesize(self, text: str, output_path: Path, **kwargs: Any) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class MeloAdapter(BaseAdapter):
    model_name = "melotts"
    display_name = "MeloTTS"
    target_device = "cpu"

    def __init__(self) -> None:
        self.models: dict[str, tuple[Any, dict[str, int]]] = {}

    def load(self) -> None:
        from melo.api import TTS  # type: ignore

        for language in ["KR", "EN"]:
            model = TTS(language=language, device=self.target_device)
            self.models[language] = (model, dict(model.hps.data.spk2id))

    def _get_model(self, language: str) -> tuple[Any, dict[str, int]]:
        if not self.models:
            raise RuntimeError("MeloTTS model is not loaded")
        normalized = language.upper()
        if normalized not in self.models:
            raise RuntimeError(f"MeloTTS language not loaded: {language}")
        return self.models[normalized]

    def synthesize(self, text: str, output_path: Path, **kwargs: Any) -> tuple[np.ndarray, int]:
        if not self.models:
            raise RuntimeError("MeloTTS model is not loaded")
        language = str(kwargs.get("language", "KR")).upper()
        speaker_name = str(kwargs.get("speaker_name") or kwargs.get("voice_name") or language)
        speed = float(kwargs.get("speed", 1.0))
        model, speaker_ids = self._get_model(language)
        if speaker_name not in speaker_ids:
            speaker_name = language
        speaker_id = speaker_ids[speaker_name]
        model.tts_to_file(
            text=text,
            speaker_id=speaker_id,
            output_path=str(output_path),
            speed=speed,
        )
        return read_wav(output_path)


class SupertonicAdapter(BaseAdapter):
    model_name = "supertonic2"
    display_name = "Supertonic2"
    target_device = "cpu"

    def __init__(self) -> None:
        self.tts = None
        self.voice_style = None
        self._style_cache: dict[str, Any] = {}

    def load(self) -> None:
        from supertonic import TTS  # type: ignore

        self.tts = TTS(auto_download=True)
        self.voice_style = self.tts.get_voice_style(voice_name="F2")

    def _get_voice_style(self, voice_name: str) -> Any:
        if self.tts is None:
            raise RuntimeError("Supertonic2 model is not loaded")
        if voice_name not in self._style_cache:
            self._style_cache[voice_name] = self.tts.get_voice_style(voice_name=voice_name)
        return self._style_cache[voice_name]

    def synthesize(self, text: str, output_path: Path, **kwargs: Any) -> tuple[np.ndarray, int]:
        if self.tts is None or self.voice_style is None:
            raise RuntimeError("Supertonic2 model is not loaded")
        lang = str(kwargs.get("lang", "ko"))
        voice_name = str(kwargs.get("voice_name", "F2"))
        speed = float(kwargs.get("speed", 1.05))
        total_step = kwargs.get("total_step", 5)
        voice_style = self._get_voice_style(voice_name)

        synthesize_kwargs: dict[str, Any] = {
            "voice_style": voice_style,
            "lang": lang,
        }
        if speed is not None:
            synthesize_kwargs["speed"] = speed
        if total_step is not None:
            synthesize_kwargs["total_step"] = int(total_step)

        try:
            wav, _duration = self.tts.synthesize(text, **synthesize_kwargs)
        except TypeError:
            synthesize_kwargs.pop("total_step", None)
            try:
                wav, _duration = self.tts.synthesize(text, **synthesize_kwargs)
            except TypeError:
                synthesize_kwargs.pop("speed", None)
                wav, _duration = self.tts.synthesize(text, **synthesize_kwargs)
        audio = to_float32_mono(wav)
        return audio, 24_000


class QwenAdapter(BaseAdapter):
    model_name = "qwen3-tts"
    display_name = "Qwen3-TTS"

    def __init__(self) -> None:
        self.model = None
        self.model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        self.target_device = "cpu"
        self.dtype_name = "float32"

    def load(self) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("Qwen3-TTS requires torch to be installed") from exc

        if torch.cuda.is_available():
            self.model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            self.target_device = "cuda:0"
            dtype = torch.bfloat16
            self.dtype_name = "bfloat16"
        else:
            self.model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            self.target_device = "cpu"
            dtype = torch.float32
            self.dtype_name = "float32"

        from qwen_tts import Qwen3TTSModel  # type: ignore

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.target_device,
            dtype=dtype,
        )

    def synthesize(self, text: str, output_path: Path, **kwargs: Any) -> tuple[np.ndarray, int]:
        if self.model is None:
            raise RuntimeError("Qwen3-TTS model is not loaded")
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language="Korean",
            speaker="Sohee",
        )
        audio = to_float32_mono(wavs[0])
        return audio, int(sr)


def adapter_for(model_name: str) -> BaseAdapter:
    if model_name == "melotts":
        return MeloAdapter()
    if model_name == "supertonic2":
        return SupertonicAdapter()
    if model_name == "qwen3-tts":
        return QwenAdapter()
    raise ValueError(f"unknown model: {model_name}")


def expand_test_sets(selected_sets: Optional[list[str]]) -> list[str]:
    if not selected_sets:
        return ["base"]
    normalized: list[str] = []
    for selected in selected_sets:
        if selected == "all":
            for set_name in ["base", "SPD", "LNG", "STY", "CLN"]:
                if set_name not in normalized:
                    normalized.append(set_name)
            continue
        if selected not in normalized:
            normalized.append(selected)
    return normalized


def split_text_for_chunking(text: str, max_chars: int = 100) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", stripped) if part.strip()]
    if not sentence_parts:
        sentence_parts = [stripped]

    chunks: list[str] = []
    buffer = ""
    for part in sentence_parts:
        if len(part) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for index in range(0, len(part), max_chars):
                chunks.append(part[index : index + max_chars])
            continue
        if not buffer:
            buffer = part
        elif len(buffer) + 1 + len(part) <= max_chars:
            buffer = f"{buffer} {part}"
        else:
            chunks.append(buffer)
            buffer = part
    if buffer:
        chunks.append(buffer)
    return chunks


def build_supertonic_phase1_cases() -> list[dict[str, Any]]:
    ko_short = BASE_TEXT_CASES[0]
    ko_long = BASE_TEXT_CASES[1]
    en_short = BASE_TEXT_CASES[4]
    long_text = TEST_SET_CASES["LNG"][0]

    cases: list[dict[str, Any]] = [
        {
            "id": "PH1_KO_BASIC",
            "set": "phase1",
            "name": "ph1_ko_basic",
            "kind": "basic",
            "text": ko_short["text"],
            "lang": "ko",
            "voice_name": "F2",
            "speed": 1.05,
            "total_step": 5,
            "chunking": False,
        },
        {
            "id": "PH1_EN_BASIC",
            "set": "phase1",
            "name": "ph1_en_basic",
            "kind": "basic",
            "text": en_short["text"],
            "lang": "en",
            "voice_name": "F2",
            "speed": 1.05,
            "total_step": 5,
            "chunking": False,
        },
        {
            "id": "PH1_CHUNK_LNG",
            "set": "phase1",
            "name": "ph1_chunk_lng",
            "kind": "chunk",
            "text": long_text["text"],
            "lang": "ko",
            "voice_name": "F2",
            "speed": 1.05,
            "total_step": 5,
            "chunking": True,
            "chunk_max_chars": 100,
        },
    ]

    for voice_name in ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]:
        cases.append(
            {
                "id": f"PH1_STYLE_{voice_name}",
                "set": "phase1",
                "name": f"ph1_style_{voice_name.lower()}",
                "kind": "style",
                "text": ko_short["text"],
                "lang": "ko",
                "voice_name": voice_name,
                "speed": 1.05,
                "total_step": 5,
                "chunking": False,
            }
        )

    for total_step in [2, 5, 10]:
        cases.append(
            {
                "id": f"PH1_STEP_{total_step}",
                "set": "phase1",
                "name": f"ph1_step_{total_step}",
                "kind": "step",
                "text": ko_long["text"],
                "lang": "ko",
                "voice_name": "F2",
                "speed": 1.05,
                "total_step": total_step,
                "chunking": False,
            }
        )

    for speed in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        cases.append(
            {
                "id": f"PH1_SPEED_{str(speed).replace('.', '_')}",
                "set": "phase1",
                "name": f"ph1_speed_{str(speed).replace('.', '_')}",
                "kind": "speed",
                "text": ko_short["text"],
                "lang": "ko",
                "voice_name": "F2",
                "speed": speed,
                "total_step": 5,
                "chunking": False,
            }
        )

    return cases


def build_melotts_phase2_cases() -> list[dict[str, Any]]:
    phase2_kr = "안녕하세요. 멜로TTS Phase 2 벤치마크를 시작합니다."
    phase2_en = "Hello, this is the Phase 2 MeloTTS benchmark."
    long_text = (
        "음성 합성 시스템의 장문 처리 안정성을 확인하기 위해, 이 문장은 충분히 길게 작성되었습니다. "
        "문장부호가 자연스럽게 이어지도록 구성했고, 동일한 인스턴스에서 반복 합성을 해도 "
        "발음, 억양, 길이가 크게 흔들리지 않는지 확인하는 데 목적이 있습니다. "
        "속도와 관계없이 문장 전체가 끊기지 않고 forward-only로 처리되는지를 살펴봅니다."
    )

    cases: list[dict[str, Any]] = [
        {
            "id": "PH2_KR_REUSE",
            "set": "phase2",
            "name": "ph2_kr_reuse",
            "kind": "reuse",
            "text": phase2_kr,
            "language": "KR",
            "speaker_name": "KR",
            "speed": 1.0,
            "chunking": False,
        },
        {
            "id": "PH2_EN_US",
            "set": "phase2",
            "name": "ph2_en_us",
            "kind": "accent",
            "text": phase2_en,
            "language": "EN",
            "speaker_name": "EN-US",
            "speed": 1.0,
            "chunking": False,
        },
        {
            "id": "PH2_EN_BR",
            "set": "phase2",
            "name": "ph2_en_br",
            "kind": "accent",
            "text": phase2_en,
            "language": "EN",
            "speaker_name": "EN-BR",
            "speed": 1.0,
            "chunking": False,
        },
        {
            "id": "PH2_EN_INDIA",
            "set": "phase2",
            "name": "ph2_en_india",
            "kind": "accent",
            "text": phase2_en,
            "language": "EN",
            "speaker_name": "EN_INDIA",
            "speed": 1.0,
            "chunking": False,
        },
        {
            "id": "PH2_EN_AU",
            "set": "phase2",
            "name": "ph2_en_au",
            "kind": "accent",
            "text": phase2_en,
            "language": "EN",
            "speaker_name": "EN-AU",
            "speed": 1.0,
            "chunking": False,
        },
        {
            "id": "PH2_SPEED_0_5",
            "set": "phase2",
            "name": "ph2_speed_0_5",
            "kind": "speed",
            "text": phase2_kr,
            "language": "KR",
            "speaker_name": "KR",
            "speed": 0.5,
            "chunking": False,
        },
        {
            "id": "PH2_SPEED_1_0",
            "set": "phase2",
            "name": "ph2_speed_1_0",
            "kind": "speed",
            "text": phase2_kr,
            "language": "KR",
            "speaker_name": "KR",
            "speed": 1.0,
            "chunking": False,
        },
        {
            "id": "PH2_SPEED_1_5",
            "set": "phase2",
            "name": "ph2_speed_1_5",
            "kind": "speed",
            "text": phase2_kr,
            "language": "KR",
            "speaker_name": "KR",
            "speed": 1.5,
            "chunking": False,
        },
        {
            "id": "PH2_SPEED_2_0",
            "set": "phase2",
            "name": "ph2_speed_2_0",
            "kind": "speed",
            "text": phase2_kr,
            "language": "KR",
            "speaker_name": "KR",
            "speed": 2.0,
            "chunking": False,
        },
        {
            "id": "PH2_LONG_FORWARD",
            "set": "phase2",
            "name": "ph2_long_forward",
            "kind": "long",
            "text": long_text,
            "language": "KR",
            "speaker_name": "KR",
            "speed": 1.0,
            "chunking": False,
        },
        {
            "id": "PH2_LONG_CHUNK",
            "set": "phase2",
            "name": "ph2_long_chunk",
            "kind": "chunk",
            "text": long_text,
            "language": "KR",
            "speaker_name": "KR",
            "speed": 1.0,
            "chunking": True,
            "chunk_max_chars": 100,
        },
    ]
    return cases


def build_cases(
    selected_sets: Optional[list[str]] = None,
    extra_cases: Optional[list[tuple[str, str]]] = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for set_name in expand_test_sets(selected_sets):
        cases.extend(TEST_SET_CASES[set_name])
    if extra_cases:
        for name, text in extra_cases:
            cases.append(
                {
                    "id": name,
                    "set": "custom",
                    "name": name.lower(),
                    "text": text,
                }
            )
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case in cases:
        case_id = str(case["id"])
        if case_id in seen_ids:
            continue
        seen_ids.add(case_id)
        deduped.append(case)
    return deduped


def maybe_cuda_sync() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def benchmark_model(
    model_name: str,
    phase: str,
    output_root: Path,
    warmup_text: str,
    test_sets: list[str],
    text_cases: list[dict[str, Any]],
) -> Path:
    adapter = adapter_for(model_name)
    model_dir = output_root / phase / model_name
    audio_dir = model_dir / "audio"
    ensure_dir(audio_dir)

    sampler = MemorySampler()
    sampler.start()
    benchmark_started = time.perf_counter()
    adapter.load()
    maybe_cuda_sync()
    load_ready_sec = time.perf_counter() - benchmark_started

    warmup_audio_path = audio_dir / "warmup.wav"
    warmup_start = time.perf_counter()
    warmup_audio, warmup_sr = adapter.synthesize(warmup_text, warmup_audio_path)
    maybe_cuda_sync()
    warmup_generation_sec = time.perf_counter() - warmup_start

    warmup_normalized = resample_audio(to_float32_mono(warmup_audio), warmup_sr, TARGET_SAMPLE_RATE)
    write_wav(warmup_audio_path, warmup_normalized, TARGET_SAMPLE_RATE)

    cases: list[BenchResult] = []
    for case in text_cases:
        case_audio_path = audio_dir / f"{case['name']}.wav"
        start = time.perf_counter()
        chunking = bool(case.get("chunking", False))
        chunk_max_chars = int(case.get("chunk_max_chars", 100))
        language = str(case.get("language", case.get("lang", "ko")))
        lang = language.lower()
        voice_name = case.get("speaker_name", case.get("voice_name"))
        speed = case.get("speed")
        total_step = case.get("total_step")

        if chunking:
            chunks = split_text_for_chunking(case["text"], max_chars=chunk_max_chars)
            chunk_audio_paths: list[Path] = []
            chunk_waves: list[np.ndarray] = []
            chunk_source_sr: Optional[int] = None
            for index, chunk_text in enumerate(chunks, start=1):
                chunk_path = audio_dir / f"{case['name']}_chunk{index}.wav"
                chunk_audio_paths.append(chunk_path)
                chunk_audio, chunk_sr = adapter.synthesize(
                    chunk_text,
                    chunk_path,
                    language=language,
                    lang=lang,
                    speaker_name=voice_name,
                    voice_name=voice_name,
                    speed=speed,
                    total_step=total_step,
                )
                chunk_source_sr = chunk_source_sr or int(chunk_sr)
                chunk_waves.append(resample_audio(to_float32_mono(chunk_audio), int(chunk_sr), TARGET_SAMPLE_RATE))
            audio = np.concatenate(chunk_waves) if chunk_waves else np.array([], dtype=np.float32)
            sr = chunk_source_sr or TARGET_SAMPLE_RATE
        else:
            audio, sr = adapter.synthesize(
                case["text"],
                case_audio_path,
                language=language,
                lang=lang,
                speaker_name=voice_name,
                voice_name=voice_name,
                speed=speed,
                total_step=total_step,
            )
        maybe_cuda_sync()
        generation_time_sec = time.perf_counter() - start

        normalized = resample_audio(to_float32_mono(audio), int(sr), TARGET_SAMPLE_RATE)
        write_wav(case_audio_path, normalized, TARGET_SAMPLE_RATE)
        normalized_duration = duration_seconds(normalized, TARGET_SAMPLE_RATE)

        cases.append(
            BenchResult(
                name=case["id"],
                text=case["text"],
                text_chars=len(case["text"]),
                kind=str(case.get("kind", "base")),
                lang=lang,
                voice_name=str(voice_name) if voice_name is not None else None,
                speed=float(speed) if speed is not None else None,
                total_step=int(total_step) if total_step is not None else None,
                chunking=chunking,
                chunk_count=len(split_text_for_chunking(case["text"], max_chars=chunk_max_chars)) if chunking else 1,
                source_sample_rate=int(sr),
                normalized_sample_rate=TARGET_SAMPLE_RATE,
                output_audio_path=str(case_audio_path.relative_to(model_dir)),
                generation_time_sec=round(generation_time_sec, 6),
                audio_duration_sec=round(normalized_duration, 6),
                rtf=round(generation_time_sec / normalized_duration if normalized_duration else 0.0, 6),
                chars_per_sec=round(chars_per_second(case["text"], generation_time_sec), 6),
            )
        )

    memory_stats = sampler.stop()
    total_generation_sec = sum(item.generation_time_sec for item in cases)
    total_audio_duration_sec = sum(item.audio_duration_sec for item in cases)
    total_chars = sum(item.text_chars for item in cases)

    summary = {
        "avg_generation_time_sec": round(total_generation_sec / len(cases), 6) if cases else 0.0,
        "avg_audio_duration_sec": round(total_audio_duration_sec / len(cases), 6) if cases else 0.0,
        "avg_rtf": round(
            (total_generation_sec / total_audio_duration_sec) if total_audio_duration_sec else 0.0,
            6,
        ),
        "avg_chars_per_sec": round(
            (total_chars / total_generation_sec) if total_generation_sec else 0.0,
            6,
        ),
    }

    payload = {
        "phase": phase,
        "model": model_name,
        "display_name": adapter.display_name,
        "model_id": getattr(adapter, "model_id", None),
        "runtime_device": getattr(adapter, "target_device", None),
        "runtime_dtype": getattr(adapter, "dtype_name", None),
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_sets": test_sets,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cold_start": {
            "load_ready_sec": round(load_ready_sec, 6),
            "warmup_text": warmup_text,
            "warmup_generation_sec": round(warmup_generation_sec, 6),
        },
        "memory": memory_stats,
        "cases": [dataclasses.asdict(item) for item in cases],
        "summary": summary,
    }

    output_path = model_dir / "benchmark.json"
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def run_parent(args: argparse.Namespace) -> int:
    models = ["melotts", "supertonic2", "qwen3-tts"] if args.model == "all" else [args.model]
    python_by_model = {
        "melotts": args.python_melotts,
        "supertonic2": args.python_supertonic2,
        "qwen3-tts": args.python_qwen3,
    }
    for model_name in models:
        python_path = Path(python_by_model[model_name])
        if not python_path.exists():
            print(f"[error] python executable not found for {model_name}: {python_path}", file=sys.stderr)
            return 2
        cmd = [
            str(python_path),
            str(Path(__file__).resolve()),
            "--worker",
            "--phase",
            args.phase,
            "--model",
            model_name,
            "--output-root",
            args.output_root,
            "--warmup-text",
            args.warmup_text,
        ]
        for test_set in args.test_set or ["base"]:
            cmd.extend(["--test-set", test_set])
        for case_name, case_text in args.text_case or []:
            cmd.extend(["--text-case", case_name, case_text])
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def run_worker(args: argparse.Namespace) -> int:
    if args.phase == "phase1" and args.model == "supertonic2":
        cases = build_supertonic_phase1_cases()
        test_sets = ["phase1-supertonic2"]
        warmup_text = args.warmup_text
    elif args.phase == "phase2" and args.model == "melotts":
        cases = build_melotts_phase2_cases()
        test_sets = ["phase2-melotts"]
        warmup_text = cases[0]["text"]
    else:
        cases = build_cases(args.test_set, args.text_case)
        test_sets = expand_test_sets(args.test_set)
        warmup_text = args.warmup_text
    output_root = Path(args.output_root)
    benchmark_path = benchmark_model(
        model_name=args.model,
        phase=args.phase,
        output_root=output_root,
        warmup_text=warmup_text,
        test_sets=test_sets,
        text_cases=cases,
    )
    print(str(benchmark_path))
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
