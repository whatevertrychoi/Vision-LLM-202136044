"""
GilBeot (길벗) - 시각·청각장애인 보행 보조 Vision-LLM 시스템
================================================================

[설계 핵심] 경고(Fast Path)와 대화(Slow Path)를 분리한다.

  Fast Path (메인 스레드, 매 프레임, 멈추면 안 됨)
    Camera → 가림 점검 → YOLO → IoU 추적(다가오는 속도) → 규칙 판단 → 캐시된 음성 경고
    · LLM을 쓰지 않는다: Gemma는 수 초 걸리고 틀릴 수 있어 안전 판단에 부적합

  Slow Path (별도 스레드, 사용자가 s/d 키를 누를 때만)
    마이크 → Whisper STT → 의도 분기 ─ "그만"     → 물건 찾기 종료
                                     ├ "○○ 찾아줘" → 물건 찾기 모드
                                     └ 그 외 질문  → Gemma(장면 정보 + 질문) → Piper TTS

  두 경로는 Speaker의 '우선순위 음성 큐'에서 만난다.
  급한 경고가 들어오면 Gemma 답변의 합성·재생을 끊고 경고를 먼저 말한다.

[스레드 구성]
  메인 스레드     : 카메라 루프 (이 파일의 App.run)
  스피커 스레드   : 음성 큐에서 꺼내 합성·재생 (audio.Speaker)
  질문 스레드     : 녹음 → STT → Gemma (App._qa_worker), 누를 때마다 생성
  Gemma 로딩 스레드: 시작 시 1회 모델 로드 (assistant.Assistant)

키 조작
  q : 종료
  s : 음성 명령/질문  ("컵 찾아줘", "앞에 뭐 있어?", "그만")
  d : 질문 없이 현재 장면 설명 (Gemma)
  x : 물건 찾기 취소
  m : 음소거 토글

실행 예
  시연    : python main.py --simple             (전방 사람 → "전방에 사람이 있으니 주의하세요")
  화면    : python main.py --scale 1.5          (카메라 창 배율, 기본 2.0 = 200%)
  Jetson  : python main.py                      (USB 카메라 v4l2src, 전체 기능)
  Jetson  : python main.py --source v4l2:1      (/dev/video1)
  Jetson  : python main.py --no-llm             (경고만, 메모리 절약)
  노트북  : python main.py --source 0 --no-llm
  백업    : python main.py --source demo/street.mp4
"""

import argparse
import threading
import time

import cv2

import config                                   # 모든 임계값·문구·경로
from assistant import Assistant                 # Gemma (Slow Path 두뇌)
from audio import Listener, Speaker             # 마이크(STT) / 스피커(TTS 우선순위 큐)
from camera import Camera, default_source       # GStreamer 카메라 입력
from detector import Detector, check_camera, draw, frame_events, save_scene_json
from finder import Finder, is_find_request, is_stop_request, parse_find_target
from guide import Guide                         # Debounce + Cooldown 규칙 판단
from tracker import IoUTracker                  # IoU 추적 + 크기 증가율(TTC)


# ══════════════════════════════════════════════════════════════
# 실행 옵션
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="GilBeot walking assistant")
    p.add_argument("--source", default=default_source(),
                   help="v4l2 | v4l2:N | csi | csi:N | 웹캠 번호 | 영상 파일 경로")
    p.add_argument("--model", default=None, help="YOLO 모델 경로 (기본: config 자동 탐색)")
    p.add_argument("--no-llm", action="store_true", help="Gemma 비활성화 (메모리 절약/경고만 시연)")
    p.add_argument("--vlm", action="store_true", help="Gemma에 카메라 이미지도 함께 전달 (mmproj 필요)")
    p.add_argument("--simple", action="store_true",
                   help="시연 모드: 전방 사람은 거리와 무관하게 '전방에 사람이 있으니 주의하세요'")
    p.add_argument("--no-approach", action="store_true", help="접근 감지 경고 끄기")
    p.add_argument("--no-obstacle", action="store_true", help="의자·책상 장애물 경고 끄기")
    p.add_argument("--find", default=None, help="시작하자마자 물건 찾기 (예: --find 컵) — 테스트용")
    p.add_argument("--mute", action="store_true", help="음성 출력 끄기")
    p.add_argument("--headless", action="store_true", help="화면 출력 없이 실행")
    p.add_argument("--scale", type=float, default=2.0,
                   help="카메라 화면 표시 배율 (기본 2.0 = 200%%, 1.0 = 원본 640x480)")
    p.add_argument("--save-json-sec", type=float, default=1.0,
                   help="장면 JSON 저장 주기(초), 0이면 저장 안 함")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# 애플리케이션 (부품 조립 + 메인 루프)
# ══════════════════════════════════════════════════════════════
class App:
    def __init__(self, args):
        self.args = args

        # ── 부품 생성: 파일 하나 = 역할 하나 ──
        self.camera = Camera(args.source)          # 눈: 프레임 공급
        self.detector = Detector(args.model)       # 시각: YOLO 박스 → 방향·거리
        self.tracker = IoUTracker()                # 움직임: 같은 사람 잇기, 다가오는 속도
        self.guide = Guide()                       # 판단: 지금 말할지, 무엇을 말할지
        self.finder = Finder()                     # 길 안내: 물건 찾기 상태
        self.speaker = Speaker(enabled=not args.mute)   # 입: 우선순위 음성 큐
        self.listener = Listener()                 # 귀: 녹음 + Whisper
        self.assistant = Assistant(enabled=not args.no_llm, use_image=args.vlm)  # 대화 두뇌

        # ── 스레드 간 공유 상태 ──
        # 메인 스레드가 매 프레임 갱신하고, 질문 스레드가 Gemma에 넘길 때 읽는다.
        # 동시에 읽고 쓰면 반쯤 바뀐 값을 볼 수 있으므로 락으로 보호한다.
        self.latest_scene = {"objects": [], "image_width": 0, "image_height": 0}
        self.latest_frame = None
        self.state_lock = threading.Lock()

        self.qa_busy = False      # 질문 처리 중이면 s/d 를 또 눌러도 무시 (중복 방지)
        self.status = ""          # 화면 상단 상태 표시 (LISTENING / THINKING)

    # ──────────────────────────────────────────────────────────
    # 물건 찾기 시작
    # ──────────────────────────────────────────────────────────
    def start_find(self, text):
        """
        "컵 찾아줘" → 찾을 대상을 YOLO 클래스 이름으로 바꿔 Finder 시작

        1단계: 한국어 키워드 사전 (빠르고 확실)  "컵" → "cup"
        2단계: 사전에 없으면 Gemma에게 클래스 이름 하나만 답하게 함 (05: 출력 형식 제약)
        3단계: 둘 다 실패하면 "알아듣지 못했습니다"
        """
        cls, ko = parse_find_target(text)
        if cls is None:
            cls = self.assistant.extract_find_target(text, set(config.CLASS_KO))
            ko = config.CLASS_KO.get(cls) if cls else None
        if cls is None:
            self.speaker.say_key("find_unknown")
            return
        self.finder.start(cls, ko)
        self.speaker.say_text(f"{ko} 찾기를 시작합니다.")

    # ──────────────────────────────────────────────────────────
    # Slow Path: 음성 명령 / 질문 / 장면 설명
    # ──────────────────────────────────────────────────────────
    def start_qa(self, with_voice: bool):
        """
        s/d 키가 눌리면 질문 처리를 '새 스레드'에서 시작한다.
        녹음 4초 + Whisper + Gemma 는 수 초가 걸리는데,
        메인 스레드에서 하면 그동안 카메라·경고가 멈춰 버린다.
        """
        if self.qa_busy:
            return
        self.qa_busy = True
        threading.Thread(target=self._qa_worker, args=(with_voice,), daemon=True).start()

    def _qa_worker(self, with_voice):
        try:
            question = ""
            if with_voice:
                # ① "말씀하세요" 안내 → 끝날 때까지 대기 (안내 음성이 녹음에 섞이지 않게)
                self.speaker.say_key("listening")
                self.speaker.wait_idle(timeout=3)

                # ② 4초 녹음 → Whisper 한국어 STT
                self.status = "LISTENING..."
                question = self.listener.listen()
                print(f"[STT] {question!r}")
                if not question:
                    return                          # 아무 말도 없으면 조용히 종료

                # ③ 의도 분기 (규칙 기반, LLM 불필요)
                if self.finder.active and is_stop_request(question):   # "그만", "종료"
                    self.finder.stop()
                    self.speaker.say_key("find_cancel")
                    return
                if is_find_request(question):                          # "찾아", "어디"
                    self.start_find(question)
                    return

            # ④ 일반 질문 → Gemma
            self.status = "THINKING..."
            self.speaker.say_key("thinking")
            with self.state_lock:                   # 메인 스레드가 갱신 중일 수 있으므로 락
                scene = dict(self.latest_scene)     # 지금 순간의 장면을 복사해서 사용
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            answer = self.assistant.ask(question, scene, frame)

            # ⑤ 답변은 우선순위 3 → 경고(0~2)가 오면 끊긴다
            self.speaker.say_text(answer)
        except Exception as e:
            print(f"[QA] error: {e}")               # 질문 처리 실패해도 경고 루프는 계속
        finally:
            self.status = ""
            self.qa_busy = False

    # ──────────────────────────────────────────────────────────
    # 메인 루프 (Fast Path)
    # ──────────────────────────────────────────────────────────
    def run(self):
        # ── 준비 단계 ──
        self.speaker.prerender()        # 고정 경고 문구 19개를 wav로 미리 생성 (재생 지연 제거)
        self.speaker.start()            # 스피커 전담 스레드 시작

        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("첫 프레임을 읽지 못했습니다.")
        print("[App] YOLO warm-up...")
        self.detector.warmup(frame)     # TensorRT 첫 추론은 느리므로 미리 한 번
        print("[App] YOLO ready")
        # Gemma 로드는 YOLO(TensorRT) 준비가 끝난 '뒤에' 시작한다.
        # Orin Nano는 CPU·GPU가 메모리 8GB를 공유하므로, 두 모델이 동시에 GPU 메모리를 잡으면
        # 시스템이 멈출 수 있다. 순서를 나누면 경고 기능은 바로 동작하고 Gemma는 뒤에서 준비된다.
        self.assistant.load_async()
        self.speaker.say_key("start")   # "보행 안내를 시작합니다."
        if self.args.find:
            self.start_find(self.args.find)
        print("[App] 시작  q:종료  s:음성명령  d:장면설명  x:찾기취소  m:음소거")

        fps, last_t = 0.0, time.time()
        last_json_t = 0.0
        try:
            while True:
                # ════════ ① 프레임 받기 ════════
                ok, frame = self.camera.read()
                if not ok:
                    print("[App] 프레임 읽기 실패")
                    break
                now = time.time()       # 추적·쿨다운 모두 '실제 초' 기준 → FPS가 흔들려도 일정

                # ════════ ② 카메라 가림 점검 (fail-safe) ════════
                # 밝기 평균이 너무 낮거나, 밝기 편차·선명도(Laplacian)가 모두 낮으면 렌즈가 덮인 것.
                # 가려진 동안에는 YOLO 결과를 믿을 수 없으므로 탐지를 건너뛴다.
                blocked, cam_stat = check_camera(frame)
                if blocked:
                    h, w = frame.shape[:2]
                    scene = {"timestamp": now, "image_width": w, "image_height": h, "objects": []}
                else:
                    # ════════ ③ YOLO 탐지 ════════
                    # 박스마다 direction(왼쪽/정면/오른쪽), distance(멀리/보통/가까이) 를 붙인 scene dict 생성
                    scene = self.detector.detect(frame)

                    # ════════ ④ IoU 추적 + 다가오는 속도 ════════
                    # 이전 프레임 박스와 IoU ≥ 0.3 이면 같은 객체 → 같은 track_id
                    # 최근 0.8초 키 변화를 선형회귀 → r = (dh/dt)/h → TTC ≈ 1/r
                    # 결과는 각 박스 dict 에 motion("approach"/"recede"/"steady"), ttc 로 기록됨
                    self.tracker.update(scene["objects"], now)
                    if self.args.no_approach:
                        for o in scene["objects"]:
                            if o.get("motion") == "approach":
                                o["motion"] = "steady"
                scene["camera"] = {"blocked": blocked, **cam_stat}

                # ════════ ⑤ 규칙 판단 → 음성 경고 ════════
                # frame_events : 이번 프레임의 '원시 사건' (예: {"person_front", "obstacle_chair"})
                # guide.update : 6프레임 중 4프레임 이상 나온 사건만 확정(Debounce),
                #                새로 생겼거나 5초 지났으면 말함(Cooldown),
                #                여러 개면 가장 급한 하나만 반환
                to_say, active = self.guide.update(frame_events(scene, blocked), now)
                for key in to_say:
                    # 직접 말하지 않고 '큐에 넣기만' 하고 바로 다음으로 → 메인 루프는 멈추지 않음
                    self.speaker.say_key(key)

                # ════════ ⑥ 물건 찾기 안내 ════════
                # 찾기 모드일 때만 동작. 우선순위 3 이라 보행 경고(0~2)가 항상 먼저 나온다.
                if not blocked:
                    find_key = self.finder.update(scene, now)
                    if find_key:
                        self.speaker.say_key(find_key)

                # ════════ ⑦ 공유 상태 갱신 (질문 스레드가 Gemma에 넘길 최신 장면) ════════
                with self.state_lock:
                    self.latest_scene = scene
                    self.latest_frame = frame

                # ════════ ⑧ 장면 JSON 저장 (06: 임시 파일 → os.replace 원자적 교체) ════════
                if self.args.save_json_sec > 0 and now - last_json_t >= self.args.save_json_sec:
                    save_scene_json(scene)
                    last_json_t = now

                # FPS: 지수 이동 평균 (매 프레임 값이 튀지 않게 이전 값 90% + 이번 값 10%)
                dt = now - last_t
                last_t = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

                # ════════ ⑨ 화면 출력 + 키 입력 ════════
                # 박스 색(초록/주황/빨강), 경고 테두리, 영문 자막 → 청각장애인·관람자용
                if self.args.headless:
                    continue
                status = self.status or ("MUTED" if self.speaker.muted else "")
                view = draw(frame, scene, active, fps, status, self.finder)
                if self.args.scale != 1.0:
                    # 표시용으로만 확대 (YOLO·판단은 원본 해상도 그대로) — 640x480 → 1280x960
                    view = cv2.resize(view, None, fx=self.args.scale, fy=self.args.scale,
                                      interpolation=cv2.INTER_LINEAR)
                cv2.imshow("GilBeot", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    self.start_qa(with_voice=True)     # Slow Path: 음성 명령
                elif key == ord("d"):
                    self.start_qa(with_voice=False)    # Slow Path: 장면 설명
                elif key == ord("x") and self.finder.active:
                    self.finder.stop()
                    self.speaker.say_key("find_cancel")
                elif key == ord("m"):
                    print(f"[App] mute = {self.speaker.toggle_mute()}")
        except KeyboardInterrupt:
            pass
        finally:
            # 종료 시 스피커 스레드·카메라·창 정리 (Jetson에서 카메라 장치가 잠기지 않도록)
            self.speaker.stop()
            self.camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    if args.simple:
        config.USE_NEAR_LEVEL = False       # "멈추세요" 단계 끔 → 전방 사람은 모두 "주의하세요"
    if args.no_obstacle:
        config.OBSTACLE_EVENTS = {}         # 의자·책상 경고 끔
    App(args).run()
