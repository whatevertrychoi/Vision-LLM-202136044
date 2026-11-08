# GilBeot (길벗) — 시각·청각장애인을 위한 온디바이스 보행 안내 시스템

> 이마에 단 카메라로 앞을 보고, **Jetson 안에서** 판단해서, **바로 목소리로** 알려준다.
> 인터넷 연결 없이 NVIDIA Jetson Orin Nano 한 대에서 모든 처리가 이루어진다.

| 항목 | 내용 |
|---|---|
| 과목 | 가천대학교 온디바이스 AI (Physical AI · Vision-LLM) 최종 프로젝트 |
| 작성자 | 최민석 (202136044) |
| 하드웨어 | NVIDIA Jetson Orin Nano 8GB, USB 카메라, USB 마이크, USB 스피커 |
| 사용 모델 | YOLO11n (TensorRT INT8), Gemma 4 E2B (llama.cpp, Q4_K_M), Whisper base (whisper.cpp), Piper `ko_KR-kss-medium` |
| 실행 환경 | `~/vision-llm` 수업 실행 폴더의 가상환경 (모델·도구 공용) |

**시연 한 줄 요약**: 카메라 앞에 사람이 서면 → *"전방에 사람이 있으니 주의하세요."*

---

## 목차

1. [문제 정의와 목표](#1-문제-정의와-목표)
2. [기능 목록](#2-기능-목록)
3. [시스템 설계](#3-시스템-설계)
4. [프로젝트 구조](#4-프로젝트-구조)
5. [핵심 알고리즘](#5-핵심-알고리즘)
6. [수업 내용(01~06) 활용](#6-수업-내용0106-활용)
7. [설치 및 환경](#7-설치-및-환경)
8. [실행 방법](#8-실행-방법)
9. [설정 값과 현장 튜닝](#9-설정-값과-현장-튜닝)
10. [테스트 시나리오](#10-테스트-시나리오)
11. [문제 해결 (Troubleshooting)](#11-문제-해결-troubleshooting)
12. [한계와 확장 방향](#12-한계와-확장-방향)
13. [개발 기록](#13-개발-기록)

---

## 1. 문제 정의와 목표

### 문제

| 대상 | 어려움 |
|---|---|
| 시각장애인 | 흰지팡이는 **닿아야** 알 수 있다. 다가오는 사람·자전거, 앞에 놓인 의자, 떨어뜨린 물건은 미리 알기 어렵다. |
| 청각장애인 | 자전거 벨, 부르는 소리 같은 **청각 경고**를 놓친다. |
| 기존 클라우드 AI | 영상을 서버로 보내야 해서 **지연**이 생기고, **통신이 끊기면 안내도 멈춘다**. 길에서 찍은 영상이 외부로 나가는 개인정보 문제도 있다. |

### 목표

1. 전방 위험을 **0.5초 안에** 음성으로 알린다.
2. 사용자가 **말로 물어보면** 주변 상황을 설명하고, 원하는 물건을 찾아 준다.
3. 같은 경고를 **화면 색·자막**으로도 보여 청각장애인도 쓸 수 있게 한다.
4. 인터넷 없이 **Jetson 한 대**에서 모든 처리를 끝낸다.

---

## 2. 기능 목록

### 2-1. 보행 경고 (자동, 매 프레임)

| 기능 | 조건 | 음성 | 화면 |
|---|---|---|---|
| 전방 사람 | 정면(화면 가로 30~70%) + 키가 화면 높이의 25% 이상 + 신뢰도 0.40 이상 | "전방에 사람이 있으니 주의하세요." | 주황 박스·테두리 |
| 근접 사람 | 정면 + 키 65% 이상 (`--simple`이면 꺼짐) | "바로 앞에 사람이 있습니다. 멈추세요." | 빨강 |
| **접근 감지** | 같은 사람의 크기 증가율 r > 0.35/s (충돌까지 약 2.9초 미만) | "사람이 빠르게 다가옵니다. 조심하세요." | 빨강 + 라벨 `TTC 2.5s` |
| 멀어지는 사람 | r < −0.25/s | 전방 경고 생략 | 라벨 `away` |
| 차량·자전거 | car / bus / truck / motorcycle / bicycle 정면 | "전방에 차량이나 자전거가 있습니다. 조심하세요." | 빨강 |
| **의자** | 정면(또는 화면 중앙을 가로지름) + 박스 아래쪽이 화면 하단 25% 안 + 신뢰도 0.35 이상 | "전방에 의자가 있습니다. 피해 가세요." | 주황 |
| **책상** | 위와 같음 (YOLO 클래스 `dining table`) | "전방에 책상이 있습니다. 피해 가세요." | 주황 |
| **카메라 가림** | 밝기 평균 < 35, 또는 (밝기 표준편차 < 12 그리고 Laplacian 분산 < 25) | "카메라가 가려졌습니다. 안내를 할 수 없습니다." | 빨강 + `CAMERA BLOCKED` |
| 가림 해제 | 가림 경고가 사라질 때 | "카메라가 다시 보입니다." | — |

### 2-2. 물건 찾기 모드

`s` 키 → *"컵 찾아줘"*, *"핸드폰 어디 있어?"*처럼 **찾아 / 찾기 / 어디**가 들어간 문장을 말하면 시작한다.

| 상태 | 음성 |
|---|---|
| 시작 | "컵 찾기를 시작합니다." |
| 물체가 화면 왼쪽 (중심 x < 38%) | "왼쪽으로 조금 돌리세요." |
| 물체가 화면 오른쪽 (중심 x > 62%) | "오른쪽으로 조금 돌리세요." |
| 정면 | "정면에 있습니다. 천천히 앞으로 가세요." |
| 정면 + 충분히 큼 (면적 ≥ 8% 또는 높이 ≥ 50%) | "바로 앞에 있습니다. 손을 뻗어 보세요." → **모드 종료** |
| 2.5초 이상 안 보임 | "보이지 않습니다. 천천히 고개를 돌려 보세요." |
| 대상을 알아듣지 못함 | "어떤 물건인지 알아듣지 못했습니다." |
| "그만/종료/취소/멈춰", `x` 키, 60초 경과 | "물건 찾기를 종료합니다." |

찾을 수 있는 물건 (한국어 키워드 → YOLO 클래스):

| 말 | 클래스 | 말 | 클래스 |
|---|---|---|---|
| 컵, 머그, 텀블러, 잔 | cup | 의자 | chair |
| 물병, 페트병, 병, 생수 | bottle | 가방, 백팩 / 핸드백 | backpack / handbag |
| 휴대폰, 핸드폰, 스마트폰, 전화기, 폰 | cell phone | 책 | book |
| 노트북, 랩탑 | laptop | 시계 | clock |
| 키보드 / 마우스 | keyboard / mouse | 가위 / 리모컨 / 우산 | scissors / remote / umbrella |
| 테이블, 책상, 식탁 | dining table | 그릇 / 화분 / 소파 | bowl / potted plant / couch |
| 티비, 텔레비전 | tv | 인형 / 칫솔 / 캐리어 | teddy bear / toothbrush / suitcase |
| 사람 | person | | |

사전에 없는 단어는 **Gemma가 COCO 클래스 이름 하나로** 바꿔 시도한다.

### 2-3. 음성 질문 / 장면 설명 (Gemma)

| 입력 | 동작 |
|---|---|
| `s` → *"앞에 뭐 있어?"* | 녹음 4초 → Whisper STT → Gemma → 음성 답변 |
| `d` | 질문 없이 "지금 주변 상황을 알려줘."로 Gemma 호출 |

- 탐지 결과를 `"1. 사람: 정면, 가까이, 다가오는 중 (신뢰도 0.89)"`처럼 자연어로 바꿔 Gemma에 넘긴다 (가까운 것부터 최대 8개).
- 최근 2턴의 대화를 기억한다 (Sliding Window).
- `--vlm` 옵션을 주면 카메라 이미지(512px 축소)도 함께 넘긴다 (mmproj 필요).
- 로딩 전 질문 → "언어 모델이 아직 준비되지 않았습니다.", `--no-llm` → "현재 대화 기능을 사용할 수 없습니다."

### 2-4. 화면 출력 (청각장애인·관람자용)

| 요소 | 의미 |
|---|---|
| 박스 색 | 초록 = 경고 대상 아님, 주황 = 정면 주의, 빨강 = 근접·접근·차량, 하늘색 굵은 박스 = 찾는 물건 |
| 라벨 | `#추적ID 클래스 신뢰도 [TTC x.xs / away]` |
| 흰 세로선 2개 | 전방 구역 (30% ~ 70%) |
| 두꺼운 테두리 + 하단 자막 | 가장 급한 활성 경고 (예: `CAUTION - PERSON AHEAD`) |
| 상단 띠 | 물건 찾기 상태 (예: `FIND: cup  << TURN LEFT`) |
| 좌상단 | FPS, 상태 (`LISTENING...` / `THINKING...` / `MUTED`) |

글자 크기는 해상도에 비례해 자동 조절된다 (640×480에서도 잘리지 않음).

---

## 3. 시스템 설계

### 3-1. 핵심 설계: 경고(Fast Path)와 대화(Slow Path)의 분리

보행 경고는 **지연 = 위험**이다. 그런데 Gemma는 Jetson에서 답을 만드는 데 수 초가 걸리고, 가끔 틀린 말(환각)도 한다.

만약 경고를 LLM에 맡기면:
```
0.0s  사람 보임 → YOLO → Gemma에게 "어떻게 말할까?"
3.0s  Gemma 답변 생성 → 음성 합성
3.5s  "전방에 사람이 있습니다"   ← 이미 부딪혔을 수 있음
```

그래서 두 경로로 나눴다.

| | Fast Path (안전 경고) | Slow Path (대화) |
|---|---|---|
| 언제 | 매 프레임 자동 | 사용자가 `s` / `d`를 누를 때만 |
| 판단 | YOLO + IoU 추적 + **결정적 규칙** | Whisper → 의도 분기 → 물건 찾기 / **Gemma** |
| 음성 | **미리 만든 wav**를 즉시 재생 | 동적 문장을 Piper로 실시간 합성 |
| 지연 | 약 0.3초 (확정 프레임 포함) | 수 초 (사용자가 기다릴 수 있음) |
| 원칙 | LLM은 **안전 판단에 쓰지 않는다** | 느려도 되는 일에만 LLM 사용 |

두 경로는 하나의 **우선순위 음성 큐**에서 만나고, 급한 경고가 오면 Gemma 답변의 합성·재생을 **끊고** 경고를 먼저 말한다.

### 3-2. 전체 데이터 흐름

```
                               config.py (모든 숫자·문구·경로)
                                         │
┌──────────────────────── 메인 스레드 (Fast Path, 매 프레임) ────────────────────────┐
│                                                                                  │
│  camera.py ──► check_camera() ──► detector.py ──► tracker.py ──► frame_events() │
│  (GStreamer)   (밝기·Laplacian)    (YOLO INT8)    (IoU + TTC)     (원시 사건)     │
│                      │가려짐                                          │          │
│                      └──────────────► camera_blocked ─────────────────┤          │
│                                                                       ▼          │
│                                                        guide.py (Debounce·Cooldown)
│                                                                       │          │
│  finder.py (물건 찾기 상태) ◄── scene ───────────────────────────────── │          │
│        │                                                              │          │
└────────┼──────────────────────────────────────────────────────────────┼──────────┘
         │ say_key("find_left")                         say_key("person_front")
         ▼                                                              ▼
   ┌──────────────── audio.Speaker : 우선순위 음성 큐 (스피커 전담 스레드) ─────────────┐
   │  heap[(priority, seq, item)] → piper 합성(필요 시) → aplay 재생                  │
   │  더 급한 항목이 들어오면 현재 piper/aplay 프로세스 종료 (선점)                     │
   └─────────────────────────────────────────────────────────────────────────────────┘
         ▲ say_text(answer)
         │
┌────────┴──────────── 질문 스레드 (Slow Path, s/d 키를 누를 때만) ────────────────────┐
│ "말씀하세요" → arecord 4초 → whisper-cli -l ko → 의도 분기                          │
│     ├ 찾기 중 + "그만"   → finder.stop()                                            │
│     ├ "찾아/찾기/어디"   → 키워드 사전 → (실패 시) Gemma 클래스 추출 → finder.start() │
│     └ 그 외             → 최신 scene(락으로 복사) → assistant.py (Gemma) → 답변      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 3-3. 스레드 구성

| 스레드 | 하는 일 | 왜 따로 두는가 |
|---|---|---|
| 메인 | 카메라 → 가림 점검 → YOLO → 추적 → 판단 → 화면 | **절대 멈추면 안 됨**. 말은 큐에 넣기만 하고 바로 다음 프레임으로 |
| 스피커 전담 | 큐에서 꺼내 합성·재생 | 재생 3초 동안 메인 루프가 기다리지 않게. 스피커는 하나뿐이므로 한 스레드만 담당 |
| 질문 처리 (`s`/`d`) | 녹음 → STT → Gemma | 녹음 4초 + Gemma 수 초 동안에도 경고가 계속 나오게 |
| Gemma 로딩 (1회) | 모델 로드 | YOLO 준비가 끝난 **뒤에** 시작. 카메라·경고는 그동안 이미 동작 |

공유 상태(`latest_scene`, `latest_frame`)는 메인 스레드가 쓰고 질문 스레드가 읽으므로 `threading.Lock`으로 보호한다.

### 3-4. 실행 순서 (`App.run`)

```
1. speaker.prerender()   고정 문구 19개 wav 생성 (이미 있으면 건너뜀)
2. speaker.start()       스피커 전담 스레드 시작
3. camera.read()         첫 프레임
4. detector.warmup()     TensorRT 첫 추론 (느리므로 미리)
5. assistant.load_async() ← YOLO 준비 후에 Gemma 로드 시작 (GPU 메모리 경합 방지)
6. "보행 안내를 시작합니다."
7. 메인 루프
```

> **5번 순서가 중요한 이유**: Orin Nano는 CPU와 GPU가 메모리 8GB를 공유한다. Gemma와 TensorRT 엔진이 동시에 GPU 메모리를 잡으면 시스템이 멈출 수 있어서 (실제로 warm-up에서 멈춘 적 있음) 순서를 나눴다.

---

## 4. 프로젝트 구조

```
Project02_GilBeot/
├── main.py            # 지휘자: 부품 조립, 메인 루프(Fast Path), 질문 스레드(Slow Path), 키 입력
├── config.py          # 설정판: 경로, 오디오 장치, 카메라 파이프라인, 모든 임계값, 안내 문구, 프롬프트, 키워드 사전
├── camera.py          # 눈: USB(GStreamer v4l2src) / CSI / 웹캠 / 영상 파일 입력
├── detector.py        # 시각: YOLO → scene dict, 방향·거리, 카메라 가림 점검, 원시 사건, 자연어 변환, 화면 그리기
├── tracker.py         # 움직임: IoU 추적기, 크기 증가율 → TTC
├── guide.py           # 판단: Debounce + Cooldown + 경고 숨김 + 최우선 1개 선택
├── finder.py          # 길 안내: 물건 찾기 상태 머신, 한국어 → COCO 매핑
├── audio.py           # 입과 귀: Speaker(우선순위 큐·선점·캐시 wav) / Listener(arecord + whisper-cli)
├── assistant.py       # 대화 두뇌: Gemma 상주 로드, 질의 응답, 찾기 대상 클래스 추출
├── prerender_tts.py   # 도구: 고정 문구 wav 미리 생성 + 하나씩 재생해 스피커 점검
├── README.md
├── .gitignore         # runtime/, __pycache__/, *.engine, *.gguf, *.onnx 제외
└── runtime/           # (실행 시 자동 생성, git 제외)
    ├── tts_cache/     #   고정 문구 wav (예: person_front_1a2b3c4d.wav)
    ├── scene.json     #   1초마다 저장되는 최신 장면
    ├── input.wav      #   마지막 녹음
    └── llm_answer.wav #   마지막 Gemma 답변 음성
```

### 4-1. 파일별 상세

| 파일 | 주요 클래스/함수 | 역할 |
|---|---|---|
| `main.py` | `App.run()` | 매 프레임 9단계 파이프라인 실행 (§5-1) |
| | `App.start_qa()`, `_qa_worker()` | 질문 스레드 생성, 녹음 → STT → 의도 분기 → Gemma |
| | `App.start_find()` | 키워드 사전 → Gemma → 실패 안내의 3단계 대상 결정 |
| `config.py` | `first_existing()` | 모델 후보 경로 중 존재하는 첫 번째 선택 |
| | `_pick()` | `aplay -l` / `arecord -l` 결과에서 장치 이름으로 오디오 장치 탐지 |
| | `v4l2_pipeline()`, `csi_pipeline()` | GStreamer 카메라 파이프라인 문자열 |
| `camera.py` | `Camera` | `--source` 값에 따라 입력 열기, 영상 파일은 끝나면 반복 |
| `detector.py` | `Detector.detect()` | YOLO 추론 → `scene` dict |
| | `check_camera()` | 밝기·표준편차·Laplacian으로 가림 판정 |
| | `obstacle_ahead()` | 의자·책상 전방 근접 판정 (바닥 접점 기준) |
| | `frame_events()` | 장면 → 원시 사건 집합 |
| | `scene_to_text()` | 장면 → Gemma용 자연어 |
| | `save_scene_json()` | 임시 파일 → `os.replace` 원자적 저장 |
| | `draw()` | 박스·테두리·자막·찾기 상태·FPS 그리기 |
| `tracker.py` | `iou()`, `IoUTracker.update()` | IoU 탐욕 매칭으로 track_id 유지 |
| | `Track.growth_rate()` | 최근 0.8초 선형회귀 → r = (dh/dt)/h |
| `guide.py` | `Guide.update()` | 사건 확정·반복 제어·숨김·최우선 선택 |
| `finder.py` | `parse_find_target()` | 문장 → (COCO 클래스, 한국어 이름) |
| | `Finder.update()` | 물체 위치 → 안내 상태, 3프레임 안정화 |
| `audio.py` | `Speaker.say_key()` / `say_text()` | 고정 문구 / 동적 문장을 큐에 넣기 |
| | `Speaker._push()`, `_kill_current()`, `_run()` | 선점 로직 (§5-8) |
| | `Listener.listen()` | `pasuspender -- arecord` → `whisper-cli` |
| | `clean_for_tts()` | Gemma 출력의 마크다운·이모지 제거 |
| `assistant.py` | `Assistant.load_async()` | 백그라운드 로드 |
| | `Assistant.ask()` | system 프롬프트 + 최근 2턴 + (장면 + 질문) |
| | `Assistant.extract_find_target()` | 클래스 이름 하나만 답하게 하는 제약 프롬프트 |

### 4-2. 모든 파일이 주고받는 데이터: `scene` dict

06의 "탐지 결과 → Dictionary → JSON" 방식이다. 각 단계가 같은 dict에 정보를 **덧붙인다**.

```python
scene = {
  "timestamp": 1762570541.2,
  "image_width": 640, "image_height": 480,
  "camera": {"blocked": False, "mean": 112.4, "std": 48.1, "laplacian": 310.5},   # main.py
  "objects": [
    {
      "class": "person", "class_ko": "사람", "confidence": 0.89,                  # detector.py
      "bbox": {"x1": 250, "y1": 80, "x2": 390, "y2": 470},
      "cx_ratio": 0.5, "h_ratio": 0.81, "bottom_ratio": 0.98,
      "x1_ratio": 0.39, "x2_ratio": 0.61,
      "direction": "front",          # left / front / right
      "distance": "near",            # far / mid / near
      "track_id": 3,                                                               # tracker.py
      "growth": 0.41,                # r [1/s]
      "motion": "approach",          # approach / recede / steady / unknown
      "ttc": 2.4                     # 초, approach일 때만
    }
  ]
}
```

---

## 5. 핵심 알고리즘

### 5-1. 메인 루프: 한 프레임의 9단계

```python
while True:
    ok, frame = camera.read()                              # ① 프레임
    blocked, stat = check_camera(frame)                    # ② 가림 점검
    if blocked:
        scene = {"objects": [], ...}                       #    가려지면 YOLO 생략
    else:
        scene = detector.detect(frame)                     # ③ YOLO
        tracker.update(scene["objects"], now)              # ④ 추적 + TTC
    to_say, active = guide.update(frame_events(scene, blocked), now)   # ⑤ 판단
    for key in to_say: speaker.say_key(key)                #    큐에 넣기만 (즉시 반환)
    find_key = finder.update(scene, now)                   # ⑥ 물건 찾기
    with state_lock: latest_scene, latest_frame = scene, frame   # ⑦ 공유 상태
    save_scene_json(scene)  # 1초마다                     # ⑧ JSON
    cv2.imshow(draw(...)); handle_key()                    # ⑨ 화면·키
```

- 시간은 프레임 번호가 아니라 **실제 초(`time.time()`)** 기준이다. FPS가 흔들려도 "0.8초", "5초"의 의미가 같다.
- FPS는 지수 이동 평균으로 표시한다: `fps = 0.9·fps + 0.1·(1/dt)`.

### 5-2. 방향 판정

```python
cx_ratio = ((x1 + x2) / 2) / W
direction = "left" if cx_ratio < 0.30 else "right" if cx_ratio > 0.70 else "front"
```

이마 카메라에서는 **화면 가운데 = 걸어가는 방향**이다. 가운데 40%는 사람 어깨너비 통로를 걸어갈 때 부딪힐 수 있는 범위로 잡았다.

### 5-3. 거리 판정 (단안 카메라 근사)

**사람·차량: 화면 속 키**
```python
h_ratio = (y2 - y1) / H
distance = "near" if h_ratio >= 0.65 else "mid" if h_ratio >= 0.25 else "far"
```
가까울수록 크게 보인다는 원리. 미터 단위가 아니라 3단계로만 쓴다.

**의자·책상: 바닥 접점 (박스 아래쪽 y)**

의자나 책상은 키가 낮아서 바로 앞에 있어도 박스가 작다. 그래서 높이 대신 **바닥에 닿은 지점**을 쓴다. 이마 카메라는 앞쪽 바닥을 내려다보므로, 바닥 접점이 화면 아래쪽일수록 가깝다.
```python
def obstacle_ahead(obj):
    in_front = obj["direction"] == "front" or (obj["x1_ratio"] < 0.5 < obj["x2_ratio"])  # 넓은 책상
    return obj["confidence"] >= 0.35 and in_front and obj["bottom_ratio"] >= 0.75
```

### 5-4. 원시 사건 → 확정 경고 (`guide.py`)

`frame_events()`가 이번 프레임만 보고 사건을 만든다. 예: `{"person_front", "obstacle_chair"}`
`Guide.update()`가 여러 프레임을 보고 거른다.

```python
history.append(raw_events)                                  # 최근 6프레임
active = {e for e, c in counts.items() if c >= 4}           # ① Debounce: 6중 4 이상
active -= HIDES[...]                                        # ② 숨김 규칙
for e in active:
    if e not in prev_active or now - last_spoken[e] >= 5.0: # ③ Cooldown
        candidates.append(e)
for e in prev_active - active:
    if e in CLEAR_PHRASES: candidates.append(CLEAR_PHRASES[e])   # 해제 안내 (가림 해제)
top = min(candidates, key=priority)                         # ④ 가장 급한 하나만
```

| 장치 | 막는 문제 |
|---|---|
| Debounce (6중 4) | YOLO가 한두 프레임 깜빡이는 오탐·미탐으로 말이 튀는 것 |
| Cooldown (5초) | 서 있는 사람에게 매 프레임 경고가 쏟아지는 것 |
| 단계 상승 즉시 | "주의" → "멈추세요"처럼 위험이 커지면 5초를 기다리지 않음 (새 사건이므로) |
| 숨김 규칙 | 가림 → 모든 경고 숨김, 근접 → 전방·접근 숨김, 접근 → 전방 숨김 |
| 최우선 1개 | 여러 문장을 동시에 말하면 오히려 혼란 |

### 5-5. IoU 추적 (`tracker.py`)

**문제**: YOLO는 프레임마다 박스만 줄 뿐, 이 박스가 직전의 그 사람인지 모른다 (기억이 없다).

**원리**: 15fps면 프레임 간격이 약 0.07초라 사람은 거의 움직이지 않는다. 따라서 **이전 박스와 많이 겹치는 박스 = 같은 사람**.

```python
IoU = 겹친 넓이 / 합친 넓이      # 04에서 배운 IoU

pairs = [(iou(track.box, box), tid, i)                   # 1) 모든 (트랙, 박스) 쌍
         for 같은 클래스끼리만 if iou >= 0.3]
pairs.sort(reverse=True)                                 # 2) 큰 IoU부터 탐욕 매칭
for v, tid, i in pairs:
    if tid, i 둘 다 미사용: assign(tid, i)
새 박스 → 새 ID                                          # 3)
안 보인 트랙 → miss += 1, 8프레임 초과 시 삭제            # 4)
```

| 설계 결정 | 이유 |
|---|---|
| 같은 클래스끼리만 비교 | 사람 박스가 옆 의자 박스로 넘어가지 않게 |
| 헝가리안 대신 탐욕 매칭 | 화면 속 사람이 몇 명뿐이라 결과가 거의 같고 코드가 단순 |
| IoU 임계값 0.3 | 같은 사람의 IoU는 보통 0.7 이상. 빠른 움직임에도 놓치지 않도록 넉넉하게 |
| 8프레임 유예 (약 0.5초) | 잠깐 가려지거나 YOLO가 놓쳐도 ID 유지 |
| ByteTrack 미사용 | ultralytics 추적기는 `lap` 패키지를 인터넷으로 자동 설치하려 해서 오프라인 Jetson에서 실패할 수 있음 |

**한계**: 두 사람이 교차하면 ID가 바뀔 수 있고, 오래 가려졌다 나타나면 새 ID가 된다. 이 프로젝트에서는 "누구인지"보다 "앞의 박스가 커지는 중인지"가 중요해 영향이 작다.

### 5-6. 접근 감지: 크기 증가율 → TTC

**유도** (핀홀 카메라 모델):
```
h = f · H / Z                 f: 초점거리, H: 실제 키, Z: 거리
dh/dt = f·H/Z² · v            v = −dZ/dt (다가오는 속도)
r = (dh/dt) / h = v / Z = 1 / TTC
```
**f와 H가 약분되어 사라진다.** 카메라 보정도, 사람 키도, 실제 거리도 몰라도 "몇 초 뒤 부딪히는지"는 알 수 있다.

**구현** (노이즈 대응):
```python
pts = 최근 0.8초의 (시간, h_ratio)                # 최소 5개, 구간 0.4초 이상
slope = np.polyfit(t - t[0], h, 1)[0]           # 1차 직선 기울기 = dh/dt
r = slope / h.mean()

if   r >  0.35: motion = "approach"; ttc = 1/r   # "빠르게 다가옵니다"
elif r < -0.25: motion = "recede"                # 전방 경고 생략
else:           motion = "steady"                # 일반 전방 경고
```

| 설계 결정 | 이유 |
|---|---|
| 선형회귀 (0.8초) | YOLO 박스는 서 있어도 몇 %씩 떨린다. 두 프레임 차분은 떨림을 속도로 착각. 회귀로 떨림은 평균으로 지우고 추세만 남김 |
| 임계값 0.35/s | 4m 앞에서 보통 걸음(1.4m/s)으로 다가오면 TTC = 4/1.4 ≈ 2.9초, r ≈ 0.35 |
| 실제 시간 기준 | FPS가 흔들려도 r 단위가 항상 1/초 |

**트레이드오프**: 합성 영상 테스트에서 걷기 시작 후 약 1초 뒤 경고가 나왔다 (회귀 창 + Debounce). 창을 줄이면 빨라지지만 오경고가 늘어난다.

**참고**: 이마 카메라 기준 **상대** 접근이므로 사용자가 걸어가서 가까워지는 경우도 잡힌다. 충돌 위험은 같으므로 의도한 동작이다.

### 5-7. 카메라 가림 감지 (fail-safe)

**문제**: 안내기가 조용하면 사용자는 "앞이 비었다"고 착각한다. 렌즈가 덮이면 탐지가 안 되는데도 침묵하게 된다.

```python
gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
mean, std = gray.mean(), gray.std()
lap = cv2.Laplacian(gray, cv2.CV_64F).var()          # 에지(선명도) 양
blocked = mean < 35 or (std < 12 and lap < 25)
```

| 상황 | 특징 |
|---|---|
| 어두움·완전히 가림 | 평균 밝기가 매우 낮음 |
| 손바닥·천으로 덮음 | 밝기는 있어도 균일하고 뿌옇다 → 표준편차·Laplacian 둘 다 낮음 |

가려진 동안은 YOLO를 건너뛴다 (오탐 방지 + 연산 절약).

### 5-8. 우선순위 음성 큐와 선점 (`audio.py`)

| 우선순위 | 문구 |
|---|---|
| 0 | 멈추세요(근접), 카메라 가림 |
| 1 | 빠르게 다가옴, 차량, 가림 해제 |
| 2 | 사람 주의, 의자, 책상 |
| 3 | 물건 찾기 안내, Gemma 답변, 시스템 안내 |
| 4 | 전방이 비었습니다 (기본 꺼짐) |

**구조**
```
say_key / say_text ─► _push() ─► heap[(priority, seq, item)] ─► 전담 스레드 _worker()
                         │                                       └► _play(): piper → aplay
                         └─ priority < 재생 중 priority 이면 _kill_current()
```

```python
def _push(self, priority, item):
    with self._cv:                                         # 락
        heapq.heappush(self._heap, (priority, next(self._seq), item))
        if self._playing_priority is not None and priority < self._playing_priority:
            self._kill_current()                           # 더 급하면 끊기
        self._cv.notify()

def _kill_current(self):
    self._preempt.set()                                    # "그만" 깃발
    if self._proc and self._proc.poll() is None:
        self._proc.terminate()                             # piper 또는 aplay 종료

def _run(self, cmd):                                       # piper·aplay 모두 이걸로 실행
    with self._cv:
        if self._preempt.is_set(): return False            # 이미 끊으라는 신호가 왔으면 시작 안 함
        self._proc = subprocess.Popen(cmd, ...)
    self._proc.communicate()
    ...
```

| 설계 결정 | 이유 |
|---|---|
| `heapq` + 순번 | 숫자 작은 것부터, 같으면 먼저 들어온 것부터 |
| 전담 스레드 1개 | 스피커는 하나. 메인 루프는 넣기만 하고 바로 반환 |
| 외부 프로세스 `terminate()` | piper 합성 중이든 aplay 재생 중이든 즉시 끊을 수 있음 |
| 깃발 + 같은 락 | "합성은 끝났고 aplay는 아직 시작 전"인 틈에 경고가 와도 놓치지 않음 |
| 2초 넘게 밀린 보행 경고 폐기 | 이미 상황이 바뀌었을 수 있음 (찾기 안내는 버리지 않음) |
| 같은 경고 중복 방지 | 이미 큐에 있으면 추가하지 않음 |
| 끊긴 Gemma 답변은 버림 | 상황이 바뀌었으니 옛 설명은 의미 없음. 필요하면 다시 질문 |

**지연 줄이기**: Piper는 호출마다 모델을 로드해 2~3초가 걸린다 (Jetson 실측 2.5~3.4초). 그래서 고정 문구 19개는 시작 시 **wav로 캐시**하고, 경고 때는 `aplay`만 실행한다. 파일 이름에 문구의 MD5 해시를 넣어 **문구를 고치면 자동 재생성**된다.

**검증** (가짜 piper 2초 / aplay 3초로 측정):
```
0.0s Gemma 답변 시작 → 1.0s 경고 push → 즉시 합성 중단 → 경고 재생 → 4.0s 종료
0.0s Gemma 답변 시작 → 3.0s(재생 중) 경고 push → 즉시 재생 중단 → 경고 재생 → 6.0s 종료
```

### 5-9. 물건 찾기 상태 머신 (`finder.py`)

```
            ┌───────── 3프레임 연속 같은 상태일 때만 전환 ─────────┐
 시작 ──► [left] ⇄ [center] ⇄ [right]                               │
             │         │                                           │
             │         └─ 면적 ≥ 8% 또는 높이 ≥ 50% ──► [reach] ──► 종료
             └─ 2.5초 안 보임 ──► [lost]
 "그만" / x / 60초 ──► 종료
```

- 3프레임 안정화: 경계에서 왼쪽/정면이 왔다 갔다 하며 말이 튀는 것 방지
- 같은 안내는 3초마다 반복
- 찾기 안내 우선순위는 3이라 보행 경고(0~2)가 항상 먼저 나온다

**대상 결정 3단계** (`App.start_find`):
1. 한국어 키워드 사전 (빠르고 확실, 긴 단어부터 매칭)
2. 실패 시 Gemma에게 **목록 중 영어 클래스 이름 하나로만** 답하게 함 (05: 출력 형식 제약)
3. 둘 다 실패 → "어떤 물건인지 알아듣지 못했습니다."

### 5-10. Gemma 프롬프트와 메모리 (`assistant.py`)

**System 프롬프트** (05의 Instruction / Constraint / Output Format 구조)
```
Instruction: 당신은 시각장애인의 보행을 돕는 안내 도우미입니다. ...
Constraint : 탐지 정보에 없는 객체는 추측하지 마시오.
             방향은 '왼쪽/정면/오른쪽', 거리는 '가까이/조금 떨어져/멀리'로.
             위험한 것이 있으면 가장 먼저 말하시오.
Output     : 한국어 두 문장 이내. 특수문자·이모지 없이 소리 내어 읽기 쉬운 문장.
```

**메시지 구성**
```
[system] + [최근 2턴 user/assistant] + [user: "Context:\n1. 사람: 정면, 가까이, 다가오는 중 ...\n\nQuestion:\n앞에 뭐 있어?"]
```

| 설정 | 값 | 이유 |
|---|---|---|
| `n_gpu_layers` | -1 | 전 레이어 GPU |
| `n_ctx` | 2048 | 06과 동일, 메모리 절약 |
| `n_batch`, `n_ubatch` | 32 | 06과 동일 |
| `max_tokens` | 120 | 두 문장이면 충분, 응답 시간 단축 |
| `temperature` | 0.3 | 안내는 창의성보다 일관성 |
| 메모리 | 최근 2턴 (Sliding Window) | 이미지 없이 텍스트만 저장해 컨텍스트 절약 |
| 동시 호출 | `threading.Lock` | llama.cpp 인스턴스는 동시 호출 불가 |

출력은 `clean_for_tts()`로 마크다운·이모지를 제거한 뒤 Piper로 읽는다.

### 5-11. STT와 의도 분기

```bash
pasuspender -- arecord -D <MIC> -f S16_LE -r 16000 -c 1 -d 4 runtime/input.wav
whisper-cli -m ggml-base.bin -f runtime/input.wav -l ko --no-timestamps
```
- `pasuspender`: 녹음 동안 PulseAudio가 마이크를 점유하지 않게 (06과 동일)
- 녹음 전 "말씀하세요"가 **끝날 때까지 기다린 뒤** 녹음 (안내 음성이 녹음에 섞이지 않게)
- `[BLANK_AUDIO]`, `(음악)` 같은 태그 제거
- 의도 분기는 **규칙**(단어 포함 여부)으로 한다. LLM을 부를 필요가 없어 빠르다

---

## 6. 수업 내용(01~06) 활용

| 단원 | 프로젝트에서 사용한 곳 | 파일 |
|---|---|---|
| 01 Linux & Python | `subprocess`로 arecord·aplay·whisper·piper 제어, 스레드, venv 분리 실행 (`.piper_venv`) | `audio.py`, `main.py` |
| 02 Computer Vision | GStreamer 카메라 파이프라인(`v4l2src`), 회색조 변환, 밝기 통계, **Laplacian**, 화면 오버레이 | `camera.py`, `detector.py` |
| 03 DL & GPU | GPU 추론 기반 실시간 처리 구조 | `detector.py` |
| 04 Object Detection | YOLO11n, **TensorRT INT8 engine**, warm-up, FPS, **IoU**로 직접 만든 추적기 | `detector.py`, `tracker.py` |
| 05 LLM & Gemma | System 프롬프트(Instruction/Constraint/Output Format), 출력 형식 제약으로 클래스 추출, **Sliding Window** 메모리, llama.cpp 로컬 추론 | `assistant.py`, `config.py` |
| 06 Multimodal | 탐지 → Dict → **JSON(원자적 저장)** → 자연어 → Gemma, (옵션) 이미지 입력, **Whisper STT**, **Piper TTS** | `detector.py`, `assistant.py`, `audio.py` |

---

## 7. 설치 및 환경

### 7-1. 폴더 구성

모델과 도구는 레포에 넣지 않고 **수업 실행 폴더 `~/vision-llm`** 에 있는 것을 쓴다. 다른 위치면 `export VISION_LLM_HOME=/경로`.

```
~/vision-llm/
├── .venv/                                   # ultralytics, opencv(GStreamer), llama-cpp-python, numpy
├── .piper_venv/bin/python                   # piper-tts
├── whisper.cpp/
│   ├── build/bin/whisper-cli                # CUDA 빌드 (있으면 우선)
│   ├── build-cpu/bin/whisper-cli            # CPU 빌드
│   └── models/ggml-base.bin
├── src/models/
│   ├── YOLO/yolo11n_int8.engine             # 우선순위: int8.engine → .engine → .pt
│   ├── Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
│   ├── Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf   # --vlm 일 때만
│   └── Piper/ko_KR-kss-medium.onnx
└── Project02_GilBeot/                       # ← 이 프로젝트 (git pull 위치)
```

### 7-2. 필요한 시스템 도구

`aplay`, `arecord`, `pasuspender` (ALSA / PulseAudio), GStreamer (`v4l2src`, `videoconvert`).
모두 수업 06 실습 환경에 이미 설치되어 있다.

### 7-3. 오디오 장치

| 장치 | 기본 동작 | 수동 지정 |
|---|---|---|
| 스피커 | **06과 동일**: `aplay 파일` (장치 지정 없음 → Ubuntu 기본 출력 장치) | `export GILBEOT_SPEAKER=plughw:0,0` 또는 `auto`(USB 자동 탐지) |
| 마이크 | `arecord -l`에서 이름에 Microphone이 있는 장치 → 없으면 USB 장치 → 없으면 `plughw:3,0` | `export GILBEOT_MIC=plughw:3,0` |

스피커 출력 장치는 **Ubuntu 설정 → 소리 → 출력 장치**에서 고른다.
장치 이름으로 찾으므로 USB 재연결·재부팅으로 카드 번호가 바뀌어도 동작한다.

---

## 8. 실행 방법

### 8-1. Jetson

```bash
# 1) 코드 받기
cd ~/vision-llm/Project02_GilBeot          # 레포를 받은 위치
git pull

# 2) 가상환경
source ~/vision-llm/.venv/bin/activate

# 3) 장치 확인 (처음 한 번)
ls /dev/video*        # 카메라 번호
arecord -l            # 마이크
aplay -l              # 스피커

# 4) 경고 음성 생성 + 스피커 확인 (처음 한 번, 약 1분)
python prerender_tts.py --play

# 5) 실행
python main.py --simple
```

정상 시작 로그:
```
[Camera] USB /dev/video0 (GStreamer v4l2src 640x480)
[Detector] YOLO model: /home/<user>/vision-llm/src/models/YOLO/yolo11n_int8.engine
[Speaker] Piper TTS: ko_KR-kss-medium.onnx, speaker: aplay 기본 장치
[Listener] whisper-cli: .../whisper-cli, mic: plughw:3,0
[App] YOLO warm-up...
[App] YOLO ready
[App] 시작  q:종료  s:음성명령  d:장면설명  x:찾기취소  m:음소거
[Assistant] Gemma 로드 완료 (텍스트, xx.xs)
```

### 8-2. 노트북 (Windows, 비전·규칙 검증용)

```bash
python main.py --simple --source 0 --no-llm
```
Piper/aplay가 없으면 **콘솔 출력**으로, Whisper가 없으면 **키보드 입력**으로 자동 대체된다. YOLO 엔진이 없으면 `yolo11n.pt`를 사용한다.

### 8-3. 실행 옵션

| 옵션 | 설명 |
|---|---|
| `--source v4l2` | (Jetson 기본) USB `/dev/video0`, GStreamer v4l2src 640×480 YUY2 |
| `--source v4l2:N` | `/dev/videoN` |
| `--source csi` / `csi:N` | CSI 카메라 (nvarguscamerasrc, 1280×720) |
| `--source 0` | OpenCV 기본 웹캠 (노트북 기본) |
| `--source 영상.mp4` | 영상 파일, 끝나면 반복 (시연 백업용) |
| `--model 경로` | YOLO 모델 직접 지정 |
| `--simple` | 시연 모드: "멈추세요" 단계 없이 전방 사람은 모두 "주의하세요" |
| `--no-approach` | 접근 감지 경고 끄기 |
| `--no-obstacle` | 의자·책상 경고 끄기 (책상 앞 시연 시) |
| `--no-llm` | Gemma 끄기 (메모리 절약, 경고 기능만) |
| `--vlm` | Gemma에 카메라 이미지도 전달 (mmproj 필요) |
| `--find 컵` | 시작하자마자 물건 찾기 (테스트용) |
| `--mute` | 음성 끄기 |
| `--headless` | 화면 없이 실행 |
| `--scale N` | 카메라 화면 표시 배율 (기본 2.0 = 200%, 1.0 = 원본). 표시용 확대만 하고 탐지는 원본 해상도 |
| `--save-json-sec N` | 장면 JSON 저장 주기 (기본 1초, 0이면 끔) |

### 8-4. 키

| 키 | 동작 |
|---|---|
| `s` | 음성 명령/질문 ("컵 찾아줘", "앞에 뭐 있어?", "그만") |
| `d` | 질문 없이 현재 장면 설명 |
| `x` | 물건 찾기 취소 |
| `m` | 음소거 토글 (화면 경고는 유지) |
| `q` | 종료 (영상 창에 포커스가 있어야 함) |

### 8-5. 강제 종료

```bash
Ctrl + C            # 실행한 터미널에서
Ctrl + \            # Ctrl+C가 안 먹을 때 (SIGQUIT)
pkill -9 -f main.py # 다른 터미널에서
pkill -9 aplay      # 남아서 계속 말하는 음성 정리
fuser -k /dev/video0   # 종료 후 카메라가 안 열릴 때
```

---

## 9. 설정 값과 현장 튜닝

모든 값은 `config.py` 한 곳에 있다. 현장에서 코드를 뒤질 필요 없이 이 파일만 고치면 된다.

### 9-1. 주요 값

| 분류 | 이름 | 값 | 의미 |
|---|---|---|---|
| 탐지 | `YOLO_CONF` | 0.25 | YOLO 탐지 임계값 (물건 찾기용으로 낮게) |
| | `ALERT_CONF` | 0.40 | 보행 경고에 쓰는 최소 신뢰도 |
| | `OBSTACLE_CONF` | 0.35 | 의자·책상 경고 최소 신뢰도 |
| 방향 | `FRONT_X_MIN / MAX` | 0.30 / 0.70 | 전방 구역 |
| 거리 | `MID_RATIO` | 0.25 | 사람 키 비율, 이상이면 "주의" |
| | `NEAR_RATIO` | 0.65 | 사람 키 비율, 이상이면 "멈추세요" |
| | `OBSTACLE_BOTTOM_RATIO` | 0.75 | 의자·책상 바닥 접점 비율 |
| 안정화 | `DEBOUNCE_WINDOW / HITS` | 6 / 4 | 6프레임 중 4프레임 |
| | `REPEAT_SEC` | 5.0 | 같은 경고 반복 간격 |
| | `ALERT_STALE_SEC` | 2.0 | 이보다 늦은 보행 경고는 버림 |
| 추적 | `TRACK_IOU` | 0.30 | 같은 객체 판정 IoU |
| | `TRACK_MAX_MISS` | 8 | 트랙 유지 프레임 |
| | `TRACK_WINDOW_SEC` | 0.8 | 회귀 구간 |
| | `APPROACH_RATE` | 0.35 | 접근 판정 r (1/s) |
| | `RECEDE_RATE` | −0.25 | 멀어짐 판정 r |
| 가림 | `BLOCK_MEAN` | 35 | 평균 밝기 |
| | `BLOCK_STD`, `BLOCK_LAPLACIAN` | 12, 25 | 균일함·선명도 |
| 찾기 | `FIND_CENTER_MIN / MAX` | 0.38 / 0.62 | 정면 판정 |
| | `FIND_REACH_AREA / HEIGHT` | 0.08 / 0.50 | "손을 뻗으세요" 판정 |
| | `FIND_STABLE_FRAMES` | 3 | 방향 안정화 |
| | `FIND_LOST_SEC / TIMEOUT_SEC` | 2.5 / 60 | 안 보임 / 자동 종료 |
| Gemma | `LLM_MAX_TOKENS / TEMPERATURE` | 120 / 0.3 | |
| | `LLM_HISTORY_TURNS` | 2 | 대화 기억 |
| 녹음 | `RECORD_SECONDS` | 4 | |

### 9-2. 증상별 조정

| 증상 | 조정 |
|---|---|
| 사람이 멀리 있으면 경고가 안 나옴 | `MID_RATIO` 0.25 → 0.18 |
| 옆 사람에게도 경고 | `FRONT_X_MIN / MAX`를 0.35 / 0.65로 좁히기 |
| 서 있는 사람에게 "다가옵니다" | `APPROACH_RATE` 0.35 → 0.5 |
| 빠르게 와도 접근 경고 없음 | `APPROACH_RATE` → 0.25, `TRACK_WINDOW_SEC` → 0.5 |
| 흰 벽·천장을 보면 "가려졌습니다" | `BLOCK_STD`, `BLOCK_LAPLACIAN` 낮추기 |
| 손으로 가려도 반응 없음 | `BLOCK_MEAN` 35 → 50 |
| 의자가 너무 늦게/일찍 경고 | `OBSTACLE_BOTTOM_RATIO` 내리기/올리기 |
| 의자를 잘 못 잡음 | `OBSTACLE_CONF` 0.35 → 0.25 |
| 컵이 가까운데 "손을 뻗으세요"가 안 나옴 | `FIND_REACH_AREA` 0.08 → 0.05 |
| 경고가 너무 자주 반복 | `REPEAT_SEC` 5 → 8 |
| 경고 반응이 느림 | `DEBOUNCE_HITS` 4 → 3 |
| 안내 문구 변경 | `PHRASES`의 `text` 수정 → 다음 실행 시 wav 자동 재생성 |

---

## 10. 테스트 시나리오

`A·B`는 `python main.py --simple --no-llm`, `C·D`는 `python main.py --simple`로 실행.

### A. 보행 경고

| # | 할 것 | 기대 결과 |
|---|---|---|
| A1 | 실행 | "보행 안내를 시작합니다" |
| A2 | 정면 2~3m에 서기 | 약 0.3초 뒤 "전방에 사람이 있으니 주의하세요", 주황 테두리 |
| A3 | 계속 서 있기 | 5초마다 한 번만 반복 |
| A4 | 옆으로 비키기 | 테두리 사라지고 경고 중단 |
| A5 | 화면 왼쪽/오른쪽 끝에 서기 | 초록 박스, 경고 없음 |
| A6 | 5m 이상 뒤에 서기 | 경고 없음 |
| A7 | 화면을 빠르게 가로지르기 | 경고 없음 또는 1회 (오탐 방지) |
| A8 | (`--simple` 없이) 카메라 1m 앞 | "바로 앞에 사람이 있습니다. 멈추세요", 빨간 테두리 |

### B. 추가 경고

| # | 할 것 | 기대 결과 |
|---|---|---|
| B1 | 4~5m에서 빠르게 걸어오기 | "사람이 빠르게 다가옵니다", 라벨 `TTC 2.xs` |
| B2 | 정면에 가만히 서기 | "다가옵니다"가 나오면 안 됨 |
| B3 | 가까이 섰다가 뒤로 걸어가기 | 라벨 `away`, 전방 경고 생략 |
| B4 | 손바닥으로 렌즈 덮기 | "카메라가 가려졌습니다" |
| B5 | 손 떼기 | "카메라가 다시 보입니다" |
| B6 | 흰 벽 비추기 | 가림 경고가 나오면 안 됨 |
| B7 | 바닥의 의자를 내려다보기 | "전방에 의자가 있습니다" |
| B8 | 책상 비추기 | "전방에 책상이 있습니다" |
| B9 | 사람이 서 있는 상태에서 렌즈 가리기 | 가림 경고만, 사람 경고는 숨김 |
| B10 | `m` 키 | 소리만 끊기고 화면 경고 유지 |

### C. 음성 기능 (`Gemma 로드 완료` 로그 확인 후)

| # | 할 것 | 기대 결과 |
|---|---|---|
| C1 | `s` → 말하기 | 터미널에 `[STT] '...'` |
| C2 | `s` → "컵 찾아줘" | "컵 찾기를 시작합니다", 상단 `FIND: cup` |
| C3 | 컵을 왼쪽 → 가운데 → 오른쪽으로 | "왼쪽으로…" → "정면에 있습니다" → "오른쪽으로…" |
| C4 | 컵을 30cm 앞에 | "바로 앞에 있습니다. 손을 뻗어 보세요" → 모드 종료 |
| C5 | 컵 숨기기 | 2.5초 뒤 "보이지 않습니다…" |
| C6 | `x` 또는 `s` → "그만" | "물건 찾기를 종료합니다" |
| C7 | "핸드폰 찾아줘", "의자 어디 있어?" | 각각 찾기 시작 |
| C8 | "사과 찾아줘" | Gemma가 처리하거나 "알아듣지 못했습니다" |
| C9 | `s` → "앞에 뭐 있어?" | 두 문장 이내 장면 설명 |
| C10 | `d` | 질문 없이 설명 |
| C11 | C9 후 `s` → "그럼 왼쪽은?" | 이전 대화를 이어서 답함 |

### D. 핵심 설계 확인

| # | 할 것 | 기대 결과 |
|---|---|---|
| D1 | `d` 후 Gemma가 말하는 도중 사람이 빠르게 다가오기 | 설명이 끊기고 "빠르게 다가옵니다"가 먼저 |
| D2 | 컵 찾기 중 사람이 정면에 서기 | 사람 경고가 찾기 안내보다 먼저 |
| D3 | `s`로 녹음하는 동안 화면 보기 | FPS 유지, 박스 계속 움직임 |

### 시연 순서 (권장)

A2 → A3 → A4 → B1 → B4 → B5 → C2~C4 → D1

---

## 11. 문제 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|---|---|---|
| 로그(`[TTS] ...`)는 뜨는데 소리가 안 남 | 기본 출력 장치가 스피커가 아님 (HDMI 등) | 먼저 `python 06_TTS.py`로 확인. Ubuntu 설정 → 소리 → 출력 장치를 USB 스피커로. 또는 `export GILBEOT_SPEAKER=plughw:0,0` |
| `[Speaker] 'aplay' 실패 (code 1): ...` | 장치 지정 오류·장치 점유 | 에러 메시지 확인, `aplay -l`로 번호 재확인 |
| `[Speaker] 콘솔 출력 모드` | `.piper_venv` 또는 Piper 모델 경로가 다름 | `ls ~/vision-llm/.piper_venv/bin/python ~/vision-llm/src/models/Piper/` |
| `YOLO warm-up...`에서 멈춤, Ctrl+C 안 먹음 | Gemma와 TensorRT의 GPU 메모리 경합 | 다른 터미널에서 `pkill -9 -f main.py`. 최신 `main.py`는 YOLO 준비 후 Gemma 로드. `--no-llm`으로 확인 |
| `[TRT] [W] Using an engine plan file across different models of devices` | 다른 Jetson에서 만든 엔진 | 멈추거나 오류가 나면 `--model .../yolo11n.pt`로 확인 후 `04_04_INT8_TensorRT.py`로 이 Jetson에서 엔진 재생성 |
| `카메라를 열 수 없습니다` | 번호 다름 / 이전 프로세스가 점유 | `ls /dev/video*` → `--source v4l2:N`, `fuser -k /dev/video0`, USB 재연결 |
| `[STT] ''` (빈 결과) | 마이크 장치 다름 / 소리가 작음 | `arecord -l` 확인 → `export GILBEOT_MIC=plughw:3,0`, 마이크 가까이 |
| YOLO model이 `yolo11n.pt`로 잡히고 다운로드 시도 | 엔진 경로가 다름 | `--model ~/vision-llm/실제경로/yolo11n_int8.engine` |
| 화면 창이 안 뜸 | SSH 실행 | 모니터 연결된 데스크톱 터미널에서 실행, 또는 `export DISPLAY=:0` |
| 느려짐·메모리 부족 | Gemma + YOLO + Whisper 동시 | `--no-llm`, `tegrastats`로 확인 |
| 책상 경고가 계속 나옴 | 책상 앞에서 시연 | `--no-obstacle` |

---

## 12. 한계와 확장 방향

### 한계

- 거리와 TTC는 bbox 크기 기반 **근사치** (단안 카메라).
- 물건 찾기·장애물은 **COCO 80개 클래스** 범위. 기둥·볼라드·계단·턱은 탐지 불가.
- IoU 추적은 사람이 교차하거나 오래 가려지면 ID가 바뀔 수 있음.
- 균일한 흰 벽은 카메라 가림으로 오인될 수 있음.
- 이마 카메라 기준 상대 접근이라 사용자가 다가가는 것도 "다가옴"으로 판정 (충돌 위험은 같음).
- 본 시스템은 **보조 도구**이며 흰지팡이·안내견을 대체하지 않는다.

### 확장 계획

| 기능 | 방법 |
|---|---|
| 점자블록 안내 | 화면 하단 노란색 HSV 분할 → 컨투어 중심·기울기로 좌/우/이탈 안내 |
| 신호등 | YOLO `traffic light` ROI → HSV 빨강/초록 비율 판정 |
| 실제 거리 | 깊이 카메라 또는 단안 깊이 추정 모델 |
| 청각장애인 모드 강화 | 주변 말소리 상시 STT 자막, 큰 소리 감지 화면 경고 |
| 비프 모드 | 주차 센서식 거리 비프 + 좌우 스테레오 방향음 |
| 제스처 조작 | MediaPipe 손 제스처로 질문·음소거 (키보드 없이) |

---

## 13. 개발 기록

| 변경 | 내용 |
|---|---|
| 초기 버전 | YOLO + 규칙 기반 전방 사람 경고, Gemma 질의, Whisper/Piper, Fast/Slow Path 분리 |
| 기능 추가 | IoU 추적 + TTC 접근 감지, 물건 찾기 모드, 카메라 가림 감지 |
| 카메라 | 수업 USB 카메라 파이프라인(GStreamer `v4l2src` 640×480 YUY2)을 기본값으로, 해상도 비례 화면 출력 |
| 장애물 | 의자·책상 경고 추가 (바닥 접점 기반 거리 판정) |
| 음성 선점 보완 | Piper **합성 중**에도 경고가 끼어들 수 있도록 깃발 + 락 구조로 변경 |
| 오디오 장치 | aplay/piper 에러를 로그에 표시, 스피커는 06과 동일하게 기본 장치 사용, 마이크는 이름으로 자동 탐지 |
| 안정성 | Gemma 로드를 YOLO TensorRT warm-up **이후로** 이동 (GPU 메모리 경합으로 인한 멈춤 해결) |
| 화면 | 카메라 창 표시 배율 옵션 `--scale` (기본 200%) |
