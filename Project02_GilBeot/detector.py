"""
YOLO 탐지 + 장면 해석 (Vision → 구조화 데이터)

04: YOLO / TensorRT INT8 engine
02: 밝기·Laplacian 기반 카메라 상태 점검
06: 탐지 결과 → Dictionary → JSON → 자연어
"""

import json
import os
import time

import cv2
from ultralytics import YOLO

import config


def direction_of(cx_ratio: float) -> str:
    if cx_ratio < config.FRONT_X_MIN:
        return "left"
    if cx_ratio > config.FRONT_X_MAX:
        return "right"
    return "front"


def distance_of(h_ratio: float) -> str:
    if h_ratio >= config.NEAR_RATIO:
        return "near"
    if h_ratio >= config.MID_RATIO:
        return "mid"
    return "far"


DIR_KO = {"left": "왼쪽", "front": "정면", "right": "오른쪽"}
DIST_KO = {"near": "가까이", "mid": "조금 떨어져", "far": "멀리"}
MOTION_KO = {"approach": "다가오는 중", "recede": "멀어지는 중", "steady": "", "unknown": ""}


class Detector:
    def __init__(self, model_path=None):
        self.model_path = str(model_path or config.YOLO_MODEL_PATH)
        print(f"[Detector] YOLO model: {self.model_path}")
        # .engine 파일은 task를 명시해야 경고 없이 로드됨
        self.model = YOLO(self.model_path, task="detect")

    def warmup(self, frame):
        # TensorRT 엔진은 첫 추론이 느리므로 시작 전에 한 번 돌려둔다
        self.model.predict(frame, imgsz=config.YOLO_IMGSZ, verbose=False)

    def detect(self, frame):
        """frame → scene dict"""
        h, w = frame.shape[:2]
        result = self.model.predict(
            source=frame,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            imgsz=config.YOLO_IMGSZ,
            verbose=False,
        )[0]
        names = result.names

        objects = []
        for box in result.boxes:
            cls_name = names[int(box.cls[0].item())]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().tolist()]
            cx_ratio = ((x1 + x2) / 2) / w
            h_ratio = (y2 - y1) / h
            objects.append({
                "class": cls_name,
                "class_ko": config.CLASS_KO.get(cls_name, cls_name),
                "confidence": round(conf, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "cx_ratio": round(cx_ratio, 3),
                "h_ratio": round(h_ratio, 3),
                "bottom_ratio": round(y2 / h, 3),
                "x1_ratio": round(x1 / w, 3),
                "x2_ratio": round(x2 / w, 3),
                "direction": direction_of(cx_ratio),
                "distance": distance_of(h_ratio),
            })

        return {
            "timestamp": time.time(),
            "image_width": w,
            "image_height": h,
            "objects": objects,
        }


# ──────────────────────────────────────────────────────────────
# 카메라 상태 점검 (fail-safe)
# ──────────────────────────────────────────────────────────────
def check_camera(frame):
    """
    렌즈가 손/천으로 덮이거나 너무 어두우면 True.
    - 평균 밝기가 매우 낮음 → 어두움/가림
    - 밝기 분산도 작고 Laplacian 분산(에지 양)도 작음 → 균일하게 뿌연 화면 = 렌즈 덮임
    안내 기기가 '조용하다'는 것을 '앞이 비었다'로 오해하지 않도록 알려준다.
    """
    small = cv2.resize(frame, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blocked = mean < config.BLOCK_MEAN or (std < config.BLOCK_STD and lap < config.BLOCK_LAPLACIAN)
    return blocked, {"mean": round(mean, 1), "std": round(std, 1), "laplacian": round(lap, 1)}


def obstacle_ahead(obj) -> bool:
    """
    의자·책상이 진행 방향 가까이에 있는가
      - 방향: 중심이 전방 구역 안이거나, 박스가 화면 가로 중앙(0.5)을 가로지름 (넓은 책상)
      - 거리: 박스 아래쪽(바닥 접점)이 화면 하단 25% 안
    """
    if obj["class"] not in config.OBSTACLE_EVENTS or obj["confidence"] < config.OBSTACLE_CONF:
        return False
    in_front = obj["direction"] == "front" or (obj["x1_ratio"] < 0.5 < obj["x2_ratio"])
    return in_front and obj["bottom_ratio"] >= config.OBSTACLE_BOTTOM_RATIO


# ──────────────────────────────────────────────────────────────
# 원시 이벤트
# ──────────────────────────────────────────────────────────────
def frame_events(scene, camera_blocked=False) -> set:
    """
    한 프레임에서 발생한 '원시 이벤트' 집합.
    (확정은 guide.Guide 에서 여러 프레임을 보고 결정)
    """
    events = set()
    if camera_blocked:
        events.add("camera_blocked")
        return events

    for obj in scene["objects"]:
        if obstacle_ahead(obj):
            events.add(config.OBSTACLE_EVENTS[obj["class"]])
            continue
        if obj["confidence"] < config.ALERT_CONF:
            continue
        if obj["direction"] != "front" or obj["distance"] == "far":
            continue
        motion = obj.get("motion", "unknown")

        if obj["class"] == config.PERSON_CLASS:
            if motion == "approach":
                events.add("person_approach")
            if config.USE_NEAR_LEVEL and obj["distance"] == "near":
                events.add("person_near")
            elif not (config.SUPPRESS_RECEDING and motion == "recede"):
                events.add("person_front")
        elif obj["class"] in config.VEHICLE_CLASSES:
            events.add("vehicle_front")
    return events


def scene_to_text(scene) -> str:
    """06: JSON → 자연어 (Gemma 컨텍스트). 가까운 것부터 정렬."""
    objects = sorted(scene["objects"], key=lambda o: -o["h_ratio"])
    if not objects:
        return "현재 카메라에 탐지된 객체가 없습니다."
    lines = []
    for i, o in enumerate(objects[:8], start=1):
        motion = MOTION_KO.get(o.get("motion", "unknown"), "")
        motion = f", {motion}" if motion else ""
        lines.append(
            f"{i}. {o['class_ko']}: {DIR_KO[o['direction']]}, {DIST_KO[o['distance']]}{motion} "
            f"(신뢰도 {o['confidence']:.2f})"
        )
    return "\n".join(lines)


def save_scene_json(scene, path=config.SCENE_JSON_PATH):
    """06: 임시 파일에 쓴 뒤 os.replace 로 원자적 교체 (읽는 쪽이 깨진 JSON을 보지 않도록)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────
# 시각화 (청각장애인/관람자용 화면 출력도 겸함)
# ──────────────────────────────────────────────────────────────
RED, ORANGE, GREEN, CYAN, WHITE = (0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 255, 0), (255, 255, 255)

ALERT_STYLE = {  # 이벤트 → (테두리 색, 자막)
    "camera_blocked":  (RED, "CAMERA BLOCKED - NO GUIDANCE"),
    "person_near":     (RED, "STOP - PERSON VERY CLOSE"),
    "person_approach": (RED, "WARNING - PERSON APPROACHING"),
    "vehicle_front":   (RED, "CAUTION - VEHICLE AHEAD"),
    "person_front":    (ORANGE, "CAUTION - PERSON AHEAD"),
    "obstacle_chair":  (ORANGE, "CAUTION - CHAIR AHEAD"),
    "obstacle_desk":   (ORANGE, "CAUTION - DESK AHEAD"),
}
FIND_LABEL = {"find_left": "<< TURN LEFT", "find_right": "TURN RIGHT >>",
              "find_center": "AHEAD - GO FORWARD", "find_reach": "REACH OUT", "find_lost": "NOT VISIBLE"}


def draw(frame, scene, active_events, fps, status="", finder=None):
    h, w = frame.shape[:2]
    out = frame
    k = w / 1280.0                       # 해상도 비례 스케일 (640x480 → 0.5)
    F = cv2.FONT_HERSHEY_SIMPLEX

    def px(v):
        return max(1, int(round(v * k)))

    def fit_text(text, max_w, scale, thick):
        """가로 폭을 넘으면 글자 크기를 줄인다"""
        tw = cv2.getTextSize(text, F, scale, thick)[0][0]
        return scale * min(1.0, max_w / max(tw, 1))

    # 전방 구역 표시
    x_min, x_max = int(w * config.FRONT_X_MIN), int(w * config.FRONT_X_MAX)
    cv2.line(out, (x_min, 0), (x_min, h), WHITE, 1)
    cv2.line(out, (x_max, 0), (x_max, h), WHITE, 1)

    for o in scene["objects"]:
        b = o["bbox"]
        if o["class"] in config.OBSTACLE_EVENTS:
            is_front = obstacle_ahead(o)
        else:
            is_front = (o["class"] == config.PERSON_CLASS or o["class"] in config.VEHICLE_CLASSES) \
                and o["direction"] == "front" and o["distance"] != "far" and o["confidence"] >= config.ALERT_CONF
        if o.get("motion") == "approach" and is_front:
            color = RED
        elif is_front:
            color = RED if o["distance"] == "near" else ORANGE
        else:
            color = GREEN
        thick = 2
        if finder is not None and finder.best is o:
            color, thick = CYAN, 4
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thick)
        label = f"#{o.get('track_id', '-')} {o['class']} {o['confidence']:.2f}"
        if o.get("ttc") is not None:
            label += f" TTC {o['ttc']:.1f}s"
        elif o.get("motion") == "recede":
            label += " away"
        cv2.putText(out, label, (b["x1"], max(px(40), b["y1"] - px(8))),
                    F, max(0.45, 0.6 * k), color, 1 if k < 0.75 else 2)

    # 가장 급한 활성 이벤트 → 테두리 색 + 큰 자막
    top = None
    for key in sorted(active_events, key=lambda kk: config.PHRASES[kk]["priority"]):
        if key in ALERT_STYLE:
            top = key
            break
    if top:
        color, caption = ALERT_STYLE[top]
        bar = px(70)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, px(18))
        cv2.rectangle(out, (0, h - bar), (w, h), (0, 0, 0), -1)
        thick = max(2, px(3))
        scale = fit_text(caption, w - px(40), 1.2 * k, thick)
        cv2.putText(out, caption, (px(20), h - int(bar * 0.32)), F, scale, color, thick)

    # 물건 찾기 상태
    if finder is not None and finder.active:
        guide = FIND_LABEL.get(finder.pending or "", "SEARCHING")
        text = f"FIND: {finder.target}  {guide}"
        y0, y1 = px(100), px(150)
        cv2.rectangle(out, (0, y0), (w, y1), (60, 40, 0), -1)
        scale = fit_text(text, w - px(40), 1.0 * k, 2)
        cv2.putText(out, text, (px(20), y1 - px(13)), F, scale, CYAN, 2)

    cv2.putText(out, f"FPS {fps:5.1f}", (px(20), px(40)), F, max(0.5, 1.0 * k), WHITE, 2)
    if status:
        cv2.putText(out, status, (px(20), px(80)), F, max(0.5, 0.8 * k), CYAN, 2)
    return out
