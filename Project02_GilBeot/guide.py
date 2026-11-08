"""
규칙 기반 안내 판단 (Fast Path)

왜 LLM이 아니라 규칙인가?
  - 보행 경고는 지연 = 위험. Gemma는 응답에 수 초가 걸리고 환각 가능성이 있다.
  - 그래서 '무엇을 말할지'는 결정적 규칙으로, Gemma는 사용자가 질문할 때만 쓴다.

두 가지 안정화 장치
  1) Debounce : 최근 N프레임 중 M프레임 이상 감지된 이벤트만 '활성'으로 확정
                → YOLO가 한두 프레임 깜빡이는 오탐/미탐으로 말이 튀지 않게
  2) Cooldown : 활성 이벤트가 유지되면 REPEAT_SEC 마다 한 번만 반복 안내
                → 같은 말이 매 프레임 쏟아지지 않게
"""

import time
from collections import deque

import config

# 상위 이벤트가 활성이면 숨길 하위 이벤트
HIDES = {
    "camera_blocked": {"person_near", "person_approach", "person_front", "vehicle_front",
                       "obstacle_chair", "obstacle_desk"},
    "person_near": {"person_front", "person_approach"},
    "person_approach": {"person_front"},
}


class Guide:
    def __init__(self):
        self.history = deque(maxlen=config.DEBOUNCE_WINDOW)
        self.active = set()
        self.last_spoken = {}          # event -> time

    def update(self, raw_events: set, now=None):
        """
        raw_events: 이번 프레임의 원시 이벤트
        return: (지금 말해야 할 이벤트 리스트, 현재 활성 이벤트 set)
        """
        now = time.time() if now is None else now
        self.history.append(raw_events)

        counts = {}
        for evs in self.history:
            for e in evs:
                counts[e] = counts.get(e, 0) + 1
        new_active = {e for e, c in counts.items() if c >= config.DEBOUNCE_HITS}

        for upper, lowers in HIDES.items():
            if upper in new_active:
                new_active -= lowers

        candidates = []
        for e in new_active:
            newly = e not in self.active
            due = now - self.last_spoken.get(e, 0.0) >= config.REPEAT_SEC
            if newly or due:
                candidates.append(e)

        # 이벤트가 사라질 때 안내 (예: 카메라 가림 해제)
        for e in self.active - new_active:
            if e in config.CLEAR_PHRASES:
                candidates.append(config.CLEAR_PHRASES[e])
        if config.SAY_PATH_CLEAR and self.active and not new_active:
            candidates.append("path_clear")

        self.active = new_active

        # 가장 급한 경고 하나만 말한다 (동시에 여러 문장 → 오히려 혼란)
        to_say = []
        if candidates:
            top = min(candidates, key=lambda k: config.PHRASES[k]["priority"])
            self.last_spoken[top] = now
            to_say.append(top)
        return to_say, set(new_active)
