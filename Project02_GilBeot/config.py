"""
GilBeot 설정 파일

- Jetson: 모델/도구는 ~/vision-llm (수업 실행 폴더)에서 찾는다.
- 노트북: 같은 경로가 없으면 레포의 src/ 나 ultralytics 자동 다운로드로 폴백한다.
- 실행 위치(cwd)와 무관하게 동작하도록 모든 경로를 절대경로로 만든다.
"""

import os
import platform
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1. 기본 경로
# ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent              # .../Project02_GilBeot
REPO_DIR = PROJECT_DIR.parent                              # .../Vision-LLM-202136044
VISION_LLM_HOME = Path(os.environ.get("VISION_LLM_HOME", Path.home() / "vision-llm"))

RUNTIME_DIR = PROJECT_DIR / "runtime"                      # 실행 중 생성 파일 (git 제외)
TTS_CACHE_DIR = RUNTIME_DIR / "tts_cache"
SCENE_JSON_PATH = RUNTIME_DIR / "scene.json"
MIC_WAV_PATH = RUNTIME_DIR / "input.wav"

IS_LINUX = platform.system() == "Linux"
IS_JETSON = IS_LINUX and platform.machine() == "aarch64"


def first_existing(*candidates):
    """후보 경로 중 처음으로 존재하는 것을 반환, 없으면 None"""
    for c in candidates:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


# ──────────────────────────────────────────────────────────────
# 2. 모델 경로
# ──────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = first_existing(
    VISION_LLM_HOME / "src/models/YOLO/yolo11n_int8.engine",   # 04: INT8 TensorRT (Jetson 최우선)
    VISION_LLM_HOME / "src/models/YOLO/yolo11n.engine",
    VISION_LLM_HOME / "src/models/YOLO/yolo11n.pt",
    REPO_DIR / "src/models/YOLO/yolo11n.pt",
) or "yolo11n.pt"                                              # 없으면 ultralytics가 자동 다운로드

GEMMA_MODEL_PATH = VISION_LLM_HOME / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
GEMMA_MMPROJ_PATH = VISION_LLM_HOME / "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

WHISPER_CLI = first_existing(
    VISION_LLM_HOME / "whisper.cpp/build/bin/whisper-cli",      # CUDA 빌드
    VISION_LLM_HOME / "whisper.cpp/build-cpu/bin/whisper-cli",  # CPU 빌드
)
WHISPER_MODEL = VISION_LLM_HOME / "whisper.cpp/models/ggml-base.bin"

PIPER_PYTHON = VISION_LLM_HOME / ".piper_venv/bin/python"
PIPER_MODEL = VISION_LLM_HOME / "src/models/Piper/ko_KR-kss-medium.onnx"

# ──────────────────────────────────────────────────────────────
# 3. 오디오 장치  (Jetson에서 `arecord -l`, `aplay -l` 로 확인 후 수정)
# ──────────────────────────────────────────────────────────────
MIC_DEVICE = os.environ.get("GILBEOT_MIC", "plughw:3,0")
SPEAKER_DEVICE = os.environ.get("GILBEOT_SPEAKER")           # None 이면 aplay 기본 장치
RECORD_SECONDS = 4

# ──────────────────────────────────────────────────────────────
# 4. 카메라
# ──────────────────────────────────────────────────────────────
# 기본: USB 카메라 GStreamer v4l2src (Jetson에서 실제 사용 중인 파이프라인)
USB_WIDTH, USB_HEIGHT, USB_FPS = 640, 480, 30
# CSI 카메라용 (06 노트북 파이프라인) — --source csi 일 때만 사용
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 1280, 720, 30


def v4l2_pipeline(device=0, width=USB_WIDTH, height=USB_HEIGHT, fps=USB_FPS):
    return (
        f"v4l2src device=/dev/video{device} ! "
        f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 "
        "max-size-bytes=0 max-size-time=0 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def csi_pipeline(sensor_id=0, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


# ──────────────────────────────────────────────────────────────
# 5. 탐지 / 판단 파라미터
# ──────────────────────────────────────────────────────────────
YOLO_CONF = 0.25          # 탐지 자체는 낮게 (물건 찾기에서 작은 물체도 잡도록)
ALERT_CONF = 0.40         # 보행 경고는 이 신뢰도 이상만 사용 (오경고 방지)
YOLO_IOU = 0.50
YOLO_IMGSZ = 640

PERSON_CLASS = "person"
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
# 정적 장애물: 클래스 → 이벤트 키 (의자, 책상)
OBSTACLE_EVENTS = {"chair": "obstacle_chair", "dining table": "obstacle_desk"}
TARGET_CLASSES = {PERSON_CLASS} | VEHICLE_CLASSES | set(OBSTACLE_EVENTS)

# 전방 구역: 화면 가로 비율 [FRONT_X_MIN, FRONT_X_MAX] 안에 bbox 중심이 있으면 "전방"
FRONT_X_MIN = 0.30
FRONT_X_MAX = 0.70

# 거리 근사: bbox 높이 / 화면 높이
#   >= NEAR_RATIO  : 바로 앞 (가까움)
#   >= MID_RATIO   : 전방 (주의)
#   그 미만        : 멀어서 무시
NEAR_RATIO = 0.65
MID_RATIO = 0.25
# False면 거리 단계를 나누지 않고 전방 사람은 모두 "전방에 사람이 있으니 주의하세요" (main.py --simple)
USE_NEAR_LEVEL = True

# 장애물 거리 판단: 의자·책상은 키가 낮아 bbox 높이로는 거리를 알 수 없다.
#   이마 카메라에서는 물체가 바닥에 닿은 지점(bbox 아래쪽 y)이 화면 아래일수록 가깝다.
#   bbox 하단 y / 화면 높이 ≥ OBSTACLE_BOTTOM_RATIO  → 가까운 장애물
OBSTACLE_BOTTOM_RATIO = 0.75
OBSTACLE_CONF = 0.35      # 의자·책상은 신뢰도가 낮게 나오는 편이라 별도 기준

# 시간적 안정화: 최근 WINDOW 프레임 중 HITS 프레임 이상 감지되어야 이벤트 확정
DEBOUNCE_WINDOW = 6
DEBOUNCE_HITS = 4

# ── 접근 감지 (IoU 추적 + bbox 높이 변화율) ──
#   r = (dh/dt) / h   [1/s]  →  TTC(충돌까지 시간) ≈ 1 / r
TRACK_IOU = 0.30          # 이전 프레임 박스와 IoU가 이 이상이면 같은 객체
TRACK_MAX_MISS = 8        # 이 프레임 수 이상 안 보이면 트랙 삭제
TRACK_WINDOW_SEC = 0.8    # 변화율 계산에 쓰는 최근 시간 구간
TRACK_MIN_SAMPLES = 5
APPROACH_RATE = 0.35      # r > 0.35/s  (TTC < 약 2.9초) 이면 "빠르게 다가옴"
RECEDE_RATE = -0.25       # r < -0.25/s 이면 "멀어지는 중"
SUPPRESS_RECEDING = True  # 멀어지는 사람에게는 "전방 주의" 경고 생략

# ── 카메라 가림/어두움 감지 (02: 밝기, Laplacian) ──
BLOCK_MEAN = 35           # 회색조 평균 밝기가 이 미만이면 어두움/가림
BLOCK_STD = 12            # 밝기 표준편차가 이 미만이고
BLOCK_LAPLACIAN = 25      #   Laplacian 분산(선명도)도 이 미만이면 렌즈가 덮인 것으로 판단

# ── 물건 찾기 모드 ──
FIND_CENTER_MIN = 0.38    # 물체 중심이 이 범위 안이면 "정면"
FIND_CENTER_MAX = 0.62
FIND_REACH_AREA = 0.08    # 물체 박스 면적 / 화면 면적 ≥ 이 값 + 정면 → "손을 뻗으세요"
FIND_REACH_HEIGHT = 0.50  #   또는 박스 높이 / 화면 높이 ≥ 이 값 (사람·의자처럼 가늘고 긴 물체)
FIND_STABLE_FRAMES = 3    # 같은 방향이 연속 N프레임 유지돼야 안내 변경
FIND_REPEAT_SEC = 3.0     # 같은 안내 반복 간격
FIND_LOST_SEC = 2.5       # 이 시간 동안 안 보이면 "보이지 않습니다"
FIND_TIMEOUT_SEC = 60.0   # 이 시간이 지나면 찾기 모드 자동 종료

# 같은 이벤트가 계속 유지될 때 다시 말하는 간격(초)
REPEAT_SEC = 5.0
# 큐에 오래 머문 경고는 버린다(초) — 이미 상황이 바뀌었을 수 있으므로
ALERT_STALE_SEC = 2.0     # (우선순위 2 이하 = 보행 경고에만 적용, 찾기 안내는 버리지 않음)

# ──────────────────────────────────────────────────────────────
# 6. 고정 안내 문구 (시작 시 wav로 미리 렌더링 → 즉시 재생)
#    priority: 숫자가 작을수록 급함
# ──────────────────────────────────────────────────────────────
PHRASES = {
    # 보행 경고
    "person_near":     {"text": "바로 앞에 사람이 있습니다. 멈추세요.",            "priority": 0},
    "camera_blocked":  {"text": "카메라가 가려졌습니다. 안내를 할 수 없습니다.",    "priority": 0},
    "person_approach": {"text": "사람이 빠르게 다가옵니다. 조심하세요.",           "priority": 1},
    "vehicle_front":   {"text": "전방에 차량이나 자전거가 있습니다. 조심하세요.",   "priority": 1},
    "person_front":    {"text": "전방에 사람이 있으니 주의하세요.",                "priority": 2},
    "obstacle_chair":  {"text": "전방에 의자가 있습니다. 피해 가세요.",            "priority": 2},
    "obstacle_desk":   {"text": "전방에 책상이 있습니다. 피해 가세요.",            "priority": 2},
    "camera_ok":       {"text": "카메라가 다시 보입니다.",                         "priority": 1},
    "path_clear":      {"text": "전방이 비었습니다.",                              "priority": 4},
    # 물건 찾기
    "find_left":       {"text": "왼쪽으로 조금 돌리세요.",                         "priority": 3},
    "find_right":      {"text": "오른쪽으로 조금 돌리세요.",                       "priority": 3},
    "find_center":     {"text": "정면에 있습니다. 천천히 앞으로 가세요.",           "priority": 3},
    "find_reach":      {"text": "바로 앞에 있습니다. 손을 뻗어 보세요.",           "priority": 3},
    "find_lost":       {"text": "보이지 않습니다. 천천히 고개를 돌려 보세요.",      "priority": 3},
    "find_unknown":    {"text": "어떤 물건인지 알아듣지 못했습니다.",               "priority": 3},
    "find_cancel":     {"text": "물건 찾기를 종료합니다.",                         "priority": 3},
    # 시스템
    "listening":       {"text": "말씀하세요.",                                     "priority": 3},
    "thinking":        {"text": "잠시만 기다려 주세요.",                           "priority": 3},
    "start":           {"text": "보행 안내를 시작합니다.",                         "priority": 3},
}
# 이벤트가 사라질 때 말할 문구 (예: 가림 해제 → "카메라가 다시 보입니다")
CLEAR_PHRASES = {"camera_blocked": "camera_ok"}
LLM_PRIORITY = 3          # Gemma 답변 음성 우선순위
SAY_PATH_CLEAR = False    # True면 전방 경고가 사라질 때 "전방이 비었습니다" 안내

# ──────────────────────────────────────────────────────────────
# 7. Gemma (05/06 노트북과 동일한 로딩 파라미터)
# ──────────────────────────────────────────────────────────────
LLM_CONTEXT = 2048
LLM_MAX_TOKENS = 120
LLM_TEMPERATURE = 0.3
LLM_HISTORY_TURNS = 2     # 05: Sliding Window 메모리 (최근 N턴 유지)

LLM_SYSTEM_PROMPT = """
Instruction:
당신은 시각장애인의 보행을 돕는 안내 도우미입니다.
주어진 카메라 탐지 정보와 사용자 질문을 바탕으로 짧고 명확하게 안내하시오.

Constraint:
탐지 정보에 없는 객체는 추측하지 마시오.
방향은 '왼쪽', '정면', '오른쪽'으로, 거리는 '가까이', '조금 떨어져', '멀리'로 표현하시오.
위험한 것이 있으면 가장 먼저 말하시오.

Output Format:
한국어 두 문장 이내. 특수문자와 이모지 없이 소리 내어 읽기 쉬운 문장.
""".strip()

# 클래스 영문 → 한국어 (Gemma 컨텍스트 / 안내 문장용)
CLASS_KO = {
    "person": "사람", "bicycle": "자전거", "car": "자동차", "motorcycle": "오토바이",
    "bus": "버스", "truck": "트럭", "traffic light": "신호등", "stop sign": "정지 표지판",
    "bench": "벤치", "dog": "개", "cat": "고양이", "chair": "의자", "fire hydrant": "소화전",
    "cup": "컵", "bottle": "병", "cell phone": "휴대폰", "laptop": "노트북", "keyboard": "키보드",
    "mouse": "마우스", "backpack": "가방", "handbag": "핸드백", "book": "책", "clock": "시계",
    "scissors": "가위", "remote": "리모컨", "umbrella": "우산", "couch": "소파",
    "potted plant": "화분", "bowl": "그릇", "tv": "텔레비전", "dining table": "책상",
    "teddy bear": "인형", "toothbrush": "칫솔", "suitcase": "캐리어", "bed": "침대",
}

# 물건 찾기: 사용자가 말한 한국어 → COCO 클래스 (긴 단어부터 매칭)
FIND_KEYWORDS = {
    "컵": "cup", "머그": "cup", "텀블러": "cup", "잔": "cup",
    "물병": "bottle", "페트병": "bottle", "병": "bottle", "생수": "bottle",
    "휴대폰": "cell phone", "핸드폰": "cell phone", "스마트폰": "cell phone", "전화기": "cell phone", "폰": "cell phone",
    "노트북": "laptop", "랩탑": "laptop", "키보드": "keyboard", "마우스": "mouse",
    "의자": "chair", "가방": "backpack", "백팩": "backpack", "핸드백": "handbag",
    "책": "book", "시계": "clock", "가위": "scissors", "리모컨": "remote", "우산": "umbrella",
    "소파": "couch", "화분": "potted plant", "그릇": "bowl", "티비": "tv", "텔레비전": "tv",
    "테이블": "dining table", "책상": "dining table", "식탁": "dining table",
    "인형": "teddy bear", "칫솔": "toothbrush", "캐리어": "suitcase", "사람": "person",
}
FIND_TRIGGER_WORDS = ("찾아", "찾기", "어디")
FIND_STOP_WORDS = ("그만", "종료", "취소", "멈춰")
