# GilBeot (길벗) — 시각·청각장애인 보행 보조 Vision-LLM 시스템

이마에 단 카메라로 전방을 보고, **사람이 앞에 있으면 즉시 음성으로 경고**하는 Jetson 온디바이스 시스템.
다가오는 사람을 따로 구분해 경고하고, 카메라가 가려지면 알려주며, "컵 찾아줘"라고 말하면 물건 위치로 안내한다.
모든 처리는 인터넷 없이 Jetson 안에서 수행된다.

> 시연: 카메라 앞에 사람이 서면 → **"전방에 사람이 있으니 주의하세요."**

---

## 1. 기능

| 기능 | 트리거 | 음성 |
|---|---|---|
| 전방 사람 경고 | 정면(화면 가로 30~70%) + bbox 높이 ≥ 화면의 25% | "전방에 사람이 있으니 주의하세요." |
| 근접 경고 | 정면 + bbox 높이 ≥ 65% (`--simple`이면 꺼짐) | "바로 앞에 사람이 있습니다. 멈추세요." |
| **접근 감지** | IoU 추적 + bbox 크기 증가율 → TTC < 약 2.9초 | "사람이 빠르게 다가옵니다. 조심하세요." |
| 멀어지는 사람 무시 | 크기 감소율 < -0.25/s | (경고 생략) |
| 차량·자전거 경고 | car/bus/truck/motorcycle/bicycle 정면 | "전방에 차량이나 자전거가 있습니다. 조심하세요." |
| **장애물 경고 (의자·책상)** | 정면(또는 화면 중앙을 가로지름) + 박스 아래쪽이 화면 하단 25% 안 | "전방에 의자가 있습니다. 피해 가세요." / "전방에 책상이 있습니다. 피해 가세요." |
| **카메라 가림 감지** | 밝기 평균 < 35, 또는 밝기 분산·Laplacian 분산이 모두 작음 | "카메라가 가려졌습니다. 안내를 할 수 없습니다." → 해제 시 "카메라가 다시 보입니다." |
| **물건 찾기** | `s` → "컵 찾아줘" | "왼쪽으로/오른쪽으로 조금 돌리세요" → "정면에 있습니다" → "바로 앞에 있습니다. 손을 뻗어 보세요." |
| 음성 질문 | `s` → "앞에 뭐 있어?" | Gemma가 장면 설명 |
| 장면 설명 | `d` | Gemma가 장면 설명 |
| 화면 경고 | 경고 발생 시 | 테두리 색 + 자막 (청각장애인/관람자용) |

## 2. 설계 핵심: Fast Path / Slow Path 분리

보행 경고는 **지연 = 위험**이다. Gemma는 Jetson에서 답하는 데 수 초가 걸리고 틀린 말을 할 수도 있다.
그래서 "경고"와 "설명"을 다른 경로로 분리했다.

| | Fast Path (안전 경고) | Slow Path (대화형) |
|---|---|---|
| 트리거 | 매 프레임 자동 | 사용자 키 `s` / `d` |
| 판단 | YOLO + IoU 추적 + **규칙** | 음성 의도 분기 → 물건 찾기 / **Gemma** |
| 음성 | **미리 렌더링한 wav** 즉시 재생 | 동적 문장은 Piper 실시간 합성 |

```
Camera ─► 가림 점검(밝기·Laplacian) ─► YOLO11n INT8 ─► IoU 추적 → 크기 증가율(TTC)
                                                         │
                     ┌───────────────────────────────────┤
                     ▼                                   ▼
          frame_events → Guide(Debounce+Cooldown)    Finder (물건 찾기 상태)
                     │                                   │
                     └─────► Speaker 우선순위 큐 ◄────────┘
                                   ▲
 [s] arecord → Whisper STT ─► "찾아" → Finder 시작 / "그만" → 종료 / 그 외 → Gemma
```

## 3. 핵심 원리

**접근 감지 (Scale Expansion, TTC)**
물체가 다가오면 화면 속 크기 h가 커진다. 상대 변화율 r = (dh/dt)/h 는 거리 Z, 접근 속도 v일 때 v/Z 이므로
**충돌까지 남은 시간 TTC ≈ 1/r**. 단안 카메라라 실제 거리(미터)는 몰라도 "얼마나 빨리 부딪히는지"는 알 수 있다.
최근 0.8초 동안의 bbox 높이를 선형회귀해 기울기를 구한다.

**IoU 추적**
이전 프레임 박스와 IoU ≥ 0.3인 같은 클래스 박스를 같은 객체로 매칭(큰 IoU부터 탐욕적).
ultralytics ByteTrack은 `lap` 패키지 자동 설치가 필요해 오프라인 Jetson에서 실패할 수 있어 직접 구현했다.

**카메라 가림 (fail-safe)**
안내 기기가 조용하면 사용자는 "앞이 비었다"고 오해한다. 렌즈가 덮이면 어둡거나 균일하게 뿌옇게 되므로
밝기 평균·표준편차와 Laplacian 분산(에지 양)으로 판별하고, 그동안은 YOLO를 건너뛴다.

**장애물 거리 (의자·책상)**
의자·책상은 키가 낮아서 bbox 높이로는 거리를 알 수 없다. 이마 카메라에서는 물체가 바닥에 닿은 지점(bbox 아래쪽 y)이
화면 아래에 있을수록 가깝다. 그래서 `bottom_y / 화면 높이 ≥ 0.75`이면 가까운 장애물로 본다.
책상처럼 넓은 물체는 중심이 옆에 있어도 화면 가로 중앙을 가로지르면 전방으로 본다.

**안정화**
- Debounce: 최근 6프레임 중 4프레임 이상 감지돼야 확정
- Cooldown: 유지되면 5초마다 반복, 단계가 올라가면 즉시
- 상위 경고가 하위 경고를 숨김 (가림 > 근접 > 접근 > 전방), 카메라 가림은 장애물 경고도 숨김
- 물건 찾기 방향은 3프레임 연속 같아야 안내 변경

**음성 큐**
우선순위: 멈추세요·카메라 가림(0) > 접근·차량·가림 해제(1) > 사람 주의·의자·책상(2) > 찾기 안내·Gemma(3).
더 급한 음성이 오면 재생 중인 것을 끊고, 2초 넘게 밀린 보행 경고는 버린다.

## 4. 수업 내용(01~06) 활용

| 노트북 | 사용 위치 |
|---|---|
| 01 Linux & Python | subprocess(arecord/aplay/whisper/piper), 스레드, venv 분리 실행 |
| 02 Computer Vision | GStreamer 카메라 파이프라인(`v4l2src`), 회색조·밝기 통계·**Laplacian**(카메라 가림) |
| 03 DL & GPU | GPU 추론 기반 실시간 처리 |
| 04 Object Detection | YOLO11n **TensorRT INT8**, **IoU**로 직접 만든 추적기, FPS |
| 05 LLM & Gemma | 프롬프트(Instruction/Constraint/Output Format), 출력 형식 제약으로 클래스 추출, Sliding Window 메모리 |
| 06 Multimodal | Dict → JSON(원자적 저장) → 자연어 → Gemma, Whisper STT, Piper TTS |

## 5. 파일 구성

| 파일 | 역할 |
|---|---|
| `main.py` | 진입점. 카메라 루프 + 키 입력 + 음성 명령 스레드 |
| `config.py` | 경로·장치·**모든 임계값**·안내 문구·키워드 사전 |
| `camera.py` | USB(GStreamer v4l2src, 기본) / CSI / 웹캠 / 영상 파일 입력 |
| `detector.py` | YOLO → scene dict, 카메라 가림 점검, 원시 이벤트, 자연어 변환, 화면 |
| `tracker.py` | IoU 추적기 + 크기 증가율·TTC |
| `guide.py` | Debounce + Cooldown + 경고 숨김 규칙 |
| `finder.py` | 물건 찾기 상태 머신, 한국어 → COCO 매핑 |
| `audio.py` | TTS 우선순위 큐(캐시 wav, 선점), STT |
| `assistant.py` | Gemma 상주 로드, 질의, 찾기 대상 추출 |
| `prerender_tts.py` | 고정 문구 wav 생성 / 스피커 테스트 |

## 6. 실행

모델·도구는 레포에 넣지 않고 `~/vision-llm`에서 찾는다 (다른 위치면 `export VISION_LLM_HOME=/경로`).

```bash
cd ~/<레포 경로> && git pull
cd Project02_GilBeot
source ~/vision-llm/.venv/bin/activate

arecord -l && aplay -l          # 장치 확인 → config.MIC_DEVICE / export GILBEOT_MIC=plughw:X,0
python prerender_tts.py --play  # 최초 1회: 안내 음성 생성 + 스피커 확인
python main.py --simple         # 시연 (USB 카메라 /dev/video0, GStreamer v4l2src 640x480)
python main.py --simple --source v4l2:1   # /dev/video1 인 경우 (ls /dev/video* 로 확인)
python main.py --simple --source csi      # CSI 카메라인 경우
python main.py --simple --no-llm     # 메모리 부족 시
python main.py --find 컵             # 시작부터 물건 찾기 (테스트)
python main.py --simple --no-obstacle  # 책상 앞에서 시연할 때 책상 경고가 반복되면
```

| 키 | 동작 |
|---|---|
| `s` | 음성 명령/질문 ("컵 찾아줘", "앞에 뭐 있어?", "그만") |
| `d` | 현재 장면 설명 |
| `x` | 물건 찾기 취소 |
| `m` | 음소거 |
| `q` | 종료 |

찾을 수 있는 물건(한국어 키워드): 컵·텀블러, 물병·병, 휴대폰·핸드폰, 노트북, 키보드, 마우스, 의자, 가방, 책, 시계, 가위, 리모컨, 우산, 그릇, 화분, 테이블·책상, 인형, 칫솔, 캐리어, 사람 등.
사전에 없는 단어는 Gemma가 COCO 클래스로 바꿔 시도한다.

## 7. 현장 튜닝 (`config.py`)

| 증상 | 조정 |
|---|---|
| 서 있는 사람에게도 "다가옵니다" | `APPROACH_RATE` 올리기 (0.35 → 0.5) |
| 빠르게 와도 접근 경고 없음 | `APPROACH_RATE` 내리기, `TRACK_WINDOW_SEC` 줄이기 |
| 흰 벽을 보면 "가려졌습니다" | `BLOCK_STD`, `BLOCK_LAPLACIAN` 내리기 |
| 손으로 가려도 반응 없음 | `BLOCK_MEAN` 올리기 |
| 컵이 가까운데 "손을 뻗으세요"가 안 나옴 | `FIND_REACH_AREA` 내리기 |
| 경고 거리 | `MID_RATIO`, 전방 폭 `FRONT_X_MIN/MAX` |
| 의자·책상이 너무 일찍/늦게 경고 | `OBSTACLE_BOTTOM_RATIO` 올리기/내리기 (0.75) |
| 의자를 잘 못 잡음 | `OBSTACLE_CONF` 내리기 (0.35) |

## 8. 한계와 확장

- 거리는 bbox 크기 기반 근사치(단안). 깊이 카메라로 개선 가능
- 물건 찾기는 COCO 80개 클래스 범위, 장애물 경고는 의자·책상만 (기둥·볼라드·계단은 COCO에 없음)
- 점자블록(노란색 HSV 분할), 신호등 색 판정은 같은 이벤트 구조로 확장 가능
- 본 시스템은 보조 도구이며 흰지팡이·안내견을 대체하지 않는다
