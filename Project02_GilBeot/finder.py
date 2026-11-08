"""
물건 찾기 모드

  사용자: (s 키) "컵 찾아줘"
    → STT 텍스트에서 '찾아/찾기/어디' 감지
    → 한국어 키워드 → COCO 클래스 매핑 (실패하면 Gemma에게 클래스 추출 요청)
    → 매 프레임 그 물체의 위치로 안내
         왼쪽에 있음  → "왼쪽으로 조금 돌리세요."
         오른쪽에 있음 → "오른쪽으로 조금 돌리세요."
         정면         → "정면에 있습니다. 천천히 앞으로 가세요."
         정면 + 충분히 큼 → "바로 앞에 있습니다. 손을 뻗어 보세요."  → 모드 종료
         안 보임       → "보이지 않습니다. 천천히 고개를 돌려 보세요."
    → 보행 경고(사람/차량/카메라 가림)는 찾기 중에도 계속 우선한다.
"""

import time

import config


def parse_find_target(text):
    """문장 → (coco_class, 한국어 이름) 또는 (None, None)"""
    for word in sorted(config.FIND_KEYWORDS, key=len, reverse=True):
        if word in text:
            cls = config.FIND_KEYWORDS[word]
            return cls, config.CLASS_KO.get(cls, word)
    return None, None


def is_find_request(text):
    return any(w in text for w in config.FIND_TRIGGER_WORDS)


def is_stop_request(text):
    return any(w in text for w in config.FIND_STOP_WORDS)


class Finder:
    def __init__(self):
        self.active = False
        self.target = None
        self.target_ko = None
        self.state = None          # 마지막으로 말한 상태
        self.pending = None        # 안정화 중인 상태
        self.pending_count = 0
        self.last_said_t = 0.0
        self.last_seen_t = 0.0
        self.start_t = 0.0
        self.best = None           # 화면 표시용

    def start(self, cls, ko):
        now = time.time()
        self.active = True
        self.target, self.target_ko = cls, ko
        self.state, self.pending, self.pending_count = None, None, 0
        self.last_said_t = 0.0
        self.last_seen_t = now
        self.start_t = now
        self.best = None
        print(f"[Finder] start: {ko} ({cls})")

    def stop(self):
        self.active = False
        self.best = None
        print("[Finder] stop")

    def update(self, scene, now=None):
        """return: 말할 PHRASES 키 또는 None"""
        if not self.active:
            return None
        now = time.time() if now is None else now

        if now - self.start_t > config.FIND_TIMEOUT_SEC:
            self.stop()
            return "find_cancel"

        cands = [o for o in scene["objects"] if o["class"] == self.target]
        if cands:
            W, H = scene["image_width"], scene["image_height"]
            best = max(cands, key=lambda o: (o["bbox"]["x2"] - o["bbox"]["x1"]) * (o["bbox"]["y2"] - o["bbox"]["y1"]))
            b = best["bbox"]
            area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) / float(W * H)
            cx = best["cx_ratio"]
            self.best = best
            self.last_seen_t = now
            if cx < config.FIND_CENTER_MIN:
                state = "find_left"
            elif cx > config.FIND_CENTER_MAX:
                state = "find_right"
            elif area >= config.FIND_REACH_AREA or best["h_ratio"] >= config.FIND_REACH_HEIGHT:
                state = "find_reach"
            else:
                state = "find_center"
        else:
            self.best = None
            if now - self.last_seen_t < config.FIND_LOST_SEC:
                return None        # 잠깐 놓친 것은 무시
            state = "find_lost"

        # 같은 상태가 N프레임 연속이어야 안내를 바꾼다 (경계에서 왼쪽/정면이 왔다갔다 하는 것 방지)
        if state == self.pending:
            self.pending_count += 1
        else:
            self.pending, self.pending_count = state, 1
        stable = self.pending_count >= config.FIND_STABLE_FRAMES or state == "find_lost"
        if not stable:
            return None

        changed = state != self.state
        due = now - self.last_said_t >= config.FIND_REPEAT_SEC
        if not (changed or due):
            return None

        self.state = state
        self.last_said_t = now
        if state == "find_reach":
            self.stop()            # 찾았으면 모드 종료
        return state
