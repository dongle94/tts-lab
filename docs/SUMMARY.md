## Phase 0 Codex 완료 항목

- `scripts/benchmark_tts.py` 추가: MeloTTS, Supertonic2, Qwen3-TTS 공통 벤치마크 CLI
- 모델별 uv 환경 파이썬 분리 반영: `envs/melotts/bin/python`, `envs/supertonic2/bin/python`, `envs/qwen3-tts/bin/python`
- 측정 항목 포함: 생성 시간, 오디오 길이, RTF, chars/sec, cold start time, 메모리 사용량
- `results/phase0/test_scripts.md` 기준 케이스 ID 반영: `KO_S_01`, `KO_L_01`, `KO_N_01`, `KO_C_01`, `EN_S_01`, `EN_L_01`, `MIX_01`, `MIX_02`
- 기능별 세트 선택 지원: `--test-set base|SPD|LNG|STY|CLN|all`
- 공통 샘플레이트 정규화 로직 포함: 모든 출력 WAV를 24 kHz로 저장
- 결과 저장 경로 고정: `results/{phase}/{model}/benchmark.json`
- 모델별 출력 오디오도 `results/{phase}/{model}/audio/` 아래에 함께 저장

## Phase 0 Gemini 완료 항목

- `results/phase0/test_scripts.md` 작성: 세 TTS 모델 비교를 위한 표준 테스트 문장 세트 설계
- 테스트 구성:
    - 한국어/영어 기본 테스트 (짧은 문장, 긴 문장, 숫자+기호, 구어체)
    - 한영 혼합 문장 테스트 (외래어 및 전문 용어)
    - 기능별 특화 테스트 (속도 제어, 장문 합성, 스타일 지시, 보이스 클로닝)
- 정성적 평가 기준(MOS) 정의: 자연성, 명료성, 감정 표현력, 언어 전환 품질

## Phase 1 Codex 완료 항목

- `scripts/benchmark_tts.py`에 Supertonic2 전용 Phase 1 프로파일 반영 및 실행 완료
- 포함 범위: 한국어/영어 기본 합성, 스타일(M1-M5, F1-F5) 스윕, total-step(2/5/10) 비교, speed(0.9~1.5) 스윕
- 주요 지표:
    - 평균 RTF: 0.024 (매우 빠름)
    - 평균 chars/sec: 142.8
    - 결과 저장: `results/phase1/supertonic2/benchmark.json` 및 `audio/*.wav`

## Phase 2 Codex 완료 항목

- `scripts/benchmark_tts.py --worker --model melotts --phase phase2` 실행 완료
- 포함 범위: 한국어 단일 화자(KR), 영어 액센트별 화자(EN-US, EN-BR, EN_INDIA, EN-AU), speed 0.5/1.0/1.5/2.0, cold start 반복, 긴 문장 forward-only 안정성
- 주요 지표:
    - 케이스 수: 11
    - cold start load_ready_sec: 37.756s
    - warmup generation: 3.053s
    - 평균 generation: 1.562s
    - 평균 RTF: 0.223
    - 평균 chars/sec: 42.1
    - 결과 저장: `results/phase2/melotts/benchmark.json`
- 긴 문장 청킹 검증 메모:
    - `ph2_long_chunk_chunk1.wav` 8.413s, `ph2_long_chunk_chunk2.wav` 13.501s, `ph2_long_chunk_chunk3.wav` 6.610s, 합계 28.524s
    - `ph2_long_forward.wav` 28.443s, `ph2_long_chunk.wav` 15.523s
    - 청크 텍스트는 원문과 누락 없이 일치했음
    - 차이는 `scripts/benchmark_tts.py`의 청킹 경로에서 청크별로 이미 24 kHz로 정규화한 뒤, 최종 저장 단계에서 `chunk_source_sr=44100` 기준으로 한 번 더 리샘플한 데서 발생했음
    - 따라서 Phase 4의 MeloTTS 청킹 비교 시에는 현재 `LONG_CHUNK` 수치를 그대로 신뢰하지 말고, 청킹 경로의 이중 리샘플링을 제거한 뒤 재측정해야 함

## Phase 1 Gemini 완료 항목

- `results/phase1/supertonic2/mos_eval.md` 작성: Supertonic2 청취 평가 수행
- 주요 평가 결과:
    - **Step 품질**: step 2(거친 질감) -> 5(실용적) -> 10(매끄러움)로 품질 차이 뚜렷. 기본값 5 권장.
    - **속도 제어**: 0.9x ~ 1.5x 구간에서 피치 왜곡 없이 안정적. 1.2x가 가장 자연스러움.
    - **스타일 다양성**: M/F 각 5종의 개성이 뚜렷하며 명료성이 우수함. 용도별(아나운서, 일상 등) 최적 보이스 가이드 제시.
    - **장문 안정성**: 100자 청킹 방식이 매우 안정적이며 연결이 매끄러움.

## Phase 2 Gemini 완료 항목

- `results/phase2/melotts/mos_eval.md` 작성: MeloTTS 청취 평가 및 Supertonic2와의 비교 분석
- 주요 평가 결과:
    - **한국어 품질**: 안정적이나 Supertonic2에 비해 다소 기계적인 음색.
    - **글로벌 강점**: 영어 액센트(US, BR, India, AU 등) 지원이 매우 강력하며 품질이 우수함.
    - **속도 제어**: 0.5x~2.0x 범위에서 작동하나, 1.5x 이상 고속 합성 시 Supertonic2보다 발음 뭉침이 심함.
    - **비교 인사이트**: 국내 전용 서비스에는 Supertonic2가, 글로벌 액센트 대응이 필요한 서비스에는 MeloTTS가 유리함.

## Phase 3 Codex 완료 항목

- `scripts/benchmark_tts.py`에 Qwen3-TTS 전용 Phase 3 프로파일 반영 및 실행 완료
- 포함 범위: CustomVoice(Sohee), Instruct(STY_01~04), Base Clone, VoiceDesign, 0.6B smoke test
- 결과 저장: `results/phase3/qwen3-tts/benchmark.json` 및 `audio/*.wav` 등

## Phase 3 Gemini 완료 항목

- `results/phase3/instruct_cases.md` 작성: Qwen3-TTS용 복합 지시문(감정+피치+속도) 테스트 케이스 설계
- `results/phase3/qwen3-tts/mos_eval.md` 작성: Qwen3-TTS 청취 평가 및 3사 모델 최종 비교
- 주요 평가 결과:
    - **한국어 자연성**: 4.5점으로 3사 중 최고점. 가장 인간적인 억양과 감정 표현력 확인.
    - **지시어 반영**: 자연어 지시에 따른 피치/속도/감정 변화가 매우 정교하게 반영됨. (구연동화 스타일 4.7점)
    - **클로닝/디자인**: 3초 레퍼런스로 높은 유사도의 클론 생성 및 텍스트 묘사 기반 목소리 생성(VoiceDesign) 능력 입증.
    - **최종 인사이트**: 고품질/개인화 서비스에는 Qwen3-TTS, 저비용/고속 운영에는 Supertonic2가 최적임을 확정.

## Phase 3 Codex 완료 항목

- `envs/qwen3-tts` 구축 및 Qwen3-TTS 실행 확인: Python 3.12, `qwen-tts`, `soundfile`
- 0.6B smoke test 성공: `results/phase3/qwen3-tts/smoke_0_6b.wav`
- `scripts/benchmark_tts.py --worker --model qwen3-tts --phase phase3` 실행 완료
- `results/phase3/qwen3-tts/benchmark.json` 저장
- benchmark 요약:
    - 케이스 수: 8
    - cold start `load_ready_sec`: 126.974s
    - warmup generation: 7.641s
    - 평균 generation: 6.098s
    - 평균 audio duration: 7.280s
    - 평균 RTF: 0.838
    - 평균 chars/sec: 9.511
- GPU 메모리 요약:
    - peak allocated: 4.656 GB
    - peak reserved: 4.865 GB
    - OOM 발생 없음
- 1.7B 모드 실행 및 산출물 저장:
    - CustomVoice no-instruct + STY_01~STY_04: `results/phase3/qwen3-tts/phase3_modes.json`
    - Base voice clone: `results/phase3/qwen3-tts/qwen_base_clone.wav`
    - VoiceDesign: `results/phase3/qwen3-tts/qwen_voice_design.wav`
- 스트리밍 메모:
    - 설치된 `qwen_tts` 래퍼는 true streaming generation / packet callback을 노출하지 않고, `non_streaming_mode`는 simulated streaming text input만 지원함
    - 따라서 first packet latency는 패키지 레벨에서 직접 측정하지 못했고, `phase3_modes.json`에 제한 사항을 기록함
- Qwen3-TTS 벤치마크 주의사항:
    - `scripts/benchmark_tts.py`의 `QwenAdapter`가 현재 `language="Korean"`으로 하드코딩되어 있어 `results/phase3/qwen3-tts/benchmark.json`의 `EN_S_01`, `EN_L_01`, `MIX_01`, `MIX_02` 케이스가 `lang: "ko"`로 기록됨
    - 따라서 Phase 4 비교에서 Qwen3-TTS 영어 및 혼합 언어 케이스 수치는 신뢰하지 말고, 언어 인자 전달을 수정한 뒤 재측정해야 함

## Phase 4 Codex 완료 항목

- `results/phase4/technical_comparison.md` 작성: Phase 1~3 `benchmark.json` 기반 기술적 통합 비교표
- 핵심 비교 항목: cold start, avg RTF, avg chars/sec, peak memory, GPU 필요 여부
- 유스케이스별 최적 모델 추천 매트릭스 및 운영 비용/난이도 평가 완료
- 기술적 제한사항 요약: Supertonic2(CPU 성능 평탄화), MeloTTS(청킹 리샘플링 버그), Qwen3-TTS(언어 코드 하드코딩 및 스트리밍 측정 불가) 기록

## Phase 4 Gemini 완료 항목

- `results/phase4/final_report.md` 작성: 3사 모델 통합 비교 및 최종 서비스 권장안 도출
- 주요 내용:
    - **MOS 종합 비교표**: 자연성(Qwen3 우위), 생성 속도(Supertonic 우위), 글로벌 지원(MeloTTS 우위) 등 다각도 비교
    - **유스케이스별 최종 추천**: 챗봇/안내(Supertonic2), 고속 배치(Supertonic2), 글로벌 지원(MeloTTS), 감정/스토리텔링(Qwen3), 클로닝(Qwen3)
    - **안티 패턴 가이드**: 모델별 부적합한 사용 상황 명시
    - **최종 아키텍처 제언**: Supertonic2를 메인으로 하고 Qwen3-TTS를 프리미엄 서브 엔진으로 활용하는 하이브리드 전략 제시
