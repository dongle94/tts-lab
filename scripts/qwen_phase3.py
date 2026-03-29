#!/usr/bin/env python3
"""Phase 3 Qwen3-TTS execution helper.

This script runs the Qwen3-TTS mode checks requested in docs/WORK_INSTRUCTION.md:
- CustomVoice no-instruct
- CustomVoice + instruct sweep for STY_01~STY_04
- Base voice cloning using a local reference clip
- VoiceDesign

It records timing and GPU memory stats where available and saves outputs under
results/phase3/qwen3-tts/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3" / "qwen3-tts"
REF_AUDIO = PROJECT_ROOT / "assets" / "ref_audio" / "qwen_ref_kr.wav"
REF_TEXT = "안녕하세요. 멜로TTS Phase 2 벤치마크를 시작합니다."

STY_CASES = {
    "STY_01": "정말 고마워! 네 덕분에 살았어.",
    "STY_02": "미안해, 내가 다 잘못했어. 한 번만 용서해 줘.",
    "STY_03": "조용히 해! 지금 공부하는 중이잖아.",
    "STY_04": "옛날 옛적 어느 마을에, 아주 착한 나무꾼이 살고 있었어요.",
}


def save_audio(name: str, wav: Any, sr: int) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.wav"
    sf.write(path, wav, sr)
    return str(path)


def gpu_stats() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 2),
    }


def load_model(model_id: str) -> tuple[Qwen3TTSModel, float]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    return model, time.perf_counter() - start


def run() -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "streaming": {
            "supported": False,
            "first_packet_latency_sec": None,
            "note": (
                "Installed qwen_tts wrapper exposes non_streaming_mode for simulated streaming text input only; "
                "it does not expose true streaming generation or packet callbacks."
            ),
        },
        "custom_voice": {},
        "base": {},
        "voice_design": {},
    }

    custom, load_sec = load_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    custom_entry: dict[str, Any] = {"load_sec": round(load_sec, 6), "gpu": gpu_stats(), "samples": []}

    start = time.perf_counter()
    wavs, sr = custom.generate_custom_voice(
        text="안녕하세요. Qwen3-TTS 커스텀 보이스 스모크입니다.",
        language="Korean",
        speaker="Sohee",
    )
    custom_entry["samples"].append(
        {
            "case": "NO_INSTRUCT",
            "generate_sec": round(time.perf_counter() - start, 6),
            "audio_seconds": round(len(wavs[0]) / sr, 6),
            "sample_rate": sr,
            "path": save_audio("qwen_custom_no_instruct", wavs[0], sr),
        }
    )

    for case_id, instruct in STY_CASES.items():
        start = time.perf_counter()
        wavs, sr = custom.generate_custom_voice(
            text=instruct,
            language="Korean",
            speaker="Sohee",
            instruct=instruct,
        )
        custom_entry["samples"].append(
            {
                "case": case_id,
                "generate_sec": round(time.perf_counter() - start, 6),
                "audio_seconds": round(len(wavs[0]) / sr, 6),
                "sample_rate": sr,
                "path": save_audio(f"qwen_custom_{case_id.lower()}", wavs[0], sr),
            }
        )
    report["custom_voice"] = custom_entry

    base, load_sec = load_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    start = time.perf_counter()
    wavs, sr = base.generate_voice_clone(
        text="이 목소리로 자연스럽게 읽어주세요.",
        language="Korean",
        ref_audio=str(REF_AUDIO),
        ref_text=REF_TEXT,
    )
    report["base"] = {
        "load_sec": round(load_sec, 6),
        "generate_sec": round(time.perf_counter() - start, 6),
        "audio_seconds": round(len(wavs[0]) / sr, 6),
        "sample_rate": sr,
        "path": save_audio("qwen_base_clone", wavs[0], sr),
        "gpu": gpu_stats(),
    }

    design, load_sec = load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    start = time.perf_counter()
    wavs, sr = design.generate_voice_design(
        text="오늘 회의는 여기서 마치겠습니다.",
        language="Korean",
        instruct="30대 여성 아나운서, 또렷하고 낮은 피치",
    )
    report["voice_design"] = {
        "load_sec": round(load_sec, 6),
        "generate_sec": round(time.perf_counter() - start, 6),
        "audio_seconds": round(len(wavs[0]) / sr, 6),
        "sample_rate": sr,
        "path": save_audio("qwen_voice_design", wavs[0], sr),
        "gpu": gpu_stats(),
    }

    out_path = OUTPUT_DIR / "phase3_modes.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    return report


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
