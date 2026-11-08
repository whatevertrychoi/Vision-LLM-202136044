"""
경량 IoU 추적기 + 접근 속도(TTC) 추정

04: IoU (Intersection over Union)
  - 이전 프레임의 박스와 IoU가 가장 큰 현재 박스를 같은 객체로 본다 (같은 클래스끼리만).
  - ultralytics의 ByteTrack은 'lap' 패키지를 자동 설치하려 해서 오프라인 Jetson에서 실패할 수 있어 직접 구현.

접근 판단 원리 (Scale Expansion)
  - 물체가 카메라에 다가오면 화면 속 크기 h 가 커진다.
  - 상대 변화율 r = (dh/dt) / h  [1/s]
  - 일정한 속도로 접근한다면 충돌까지 남은 시간 TTC ≈ 1 / r
    (거리 Z, 속도 v 일 때 h ∝ 1/Z  →  (dh/dt)/h = v/Z = 1/TTC)
  - 실제 거리(미터)를 몰라도 '얼마나 빨리 부딪히는지'는 알 수 있다.
  - 이마 카메라 기준 '상대' 접근이므로, 사용자가 걸어가서 가까워지는 경우도 똑같이 위험으로 잡힌다.
"""

from collections import deque

import numpy as np

import config


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


class Track:
    def __init__(self, track_id, cls, box):
        self.id = track_id
        self.cls = cls
        self.box = box
        self.miss = 0
        self.history = deque(maxlen=60)   # (time, h_ratio)

    def growth_rate(self, now):
        """최근 TRACK_WINDOW_SEC 동안 h_ratio의 선형회귀 기울기 / 평균 높이 → r [1/s]"""
        pts = [(t, h) for t, h in self.history if now - t <= config.TRACK_WINDOW_SEC]
        if len(pts) < config.TRACK_MIN_SAMPLES:
            return None
        t = np.array([p[0] for p in pts])
        h = np.array([p[1] for p in pts])
        if t[-1] - t[0] < config.TRACK_WINDOW_SEC * 0.5:
            return None
        slope = np.polyfit(t - t[0], h, 1)[0]
        return float(slope / max(h.mean(), 1e-3))


class IoUTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, objects, now):
        """
        objects(scene["objects"])에 track_id, growth, ttc, motion 필드를 채운다.
        motion: "approach" | "recede" | "steady" | "unknown"
        """
        boxes = [(o["bbox"]["x1"], o["bbox"]["y1"], o["bbox"]["x2"], o["bbox"]["y2"]) for o in objects]

        # 1) 모든 (트랙, 객체) 쌍의 IoU 계산 → 큰 것부터 탐욕적 매칭
        pairs = []
        for tid, tr in self.tracks.items():
            for i, o in enumerate(objects):
                if o["class"] != tr.cls:
                    continue
                v = iou(tr.box, boxes[i])
                if v >= config.TRACK_IOU:
                    pairs.append((v, tid, i))
        pairs.sort(reverse=True)

        used_t, used_o = set(), set()
        for _, tid, i in pairs:
            if tid in used_t or i in used_o:
                continue
            used_t.add(tid)
            used_o.add(i)
            self._assign(self.tracks[tid], objects[i], boxes[i], now)

        # 2) 매칭 안 된 객체 → 새 트랙
        for i, o in enumerate(objects):
            if i in used_o:
                continue
            tr = Track(self.next_id, o["class"], boxes[i])
            self.next_id += 1
            self.tracks[tr.id] = tr
            used_t.add(tr.id)
            self._assign(tr, o, boxes[i], now)

        # 3) 매칭 안 된 트랙 → miss 증가, 오래되면 삭제
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].miss += 1
                if self.tracks[tid].miss > config.TRACK_MAX_MISS:
                    del self.tracks[tid]

    @staticmethod
    def _assign(tr, obj, box, now):
        tr.box = box
        tr.miss = 0
        tr.history.append((now, obj["h_ratio"]))
        r = tr.growth_rate(now)
        obj["track_id"] = tr.id
        obj["growth"] = None if r is None else round(r, 3)
        if r is None:
            obj["motion"], obj["ttc"] = "unknown", None
        elif r > config.APPROACH_RATE:
            obj["motion"], obj["ttc"] = "approach", round(1.0 / r, 1)
        elif r < config.RECEDE_RATE:
            obj["motion"], obj["ttc"] = "recede", None
        else:
            obj["motion"], obj["ttc"] = "steady", None
