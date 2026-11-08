"""
Gemma 기반 대화형 설명 (Slow Path)

05: system prompt 설계, Sliding Window 메모리, llama.cpp 로컬 추론
06: 탐지 JSON → 자연어 → Gemma, (옵션) ROI 이미지 멀티모달 입력

- 모델은 프로그램 시작 시 백그라운드 스레드에서 1회 로드 후 상주한다.
  (질문마다 로드하면 수십 초 → 시연 불가)
- llama_cpp 또는 모델 파일이 없으면 자동으로 비활성화 (노트북 테스트용).
"""

import base64
import threading
import time
from collections import deque

import cv2

import config
from detector import scene_to_text


class Assistant:
    def __init__(self, enabled=True, use_image=False):
        self.enabled = enabled
        self.use_image = use_image
        self.llm = None
        self.ready = False
        self.error = None
        self.history = deque(maxlen=config.LLM_HISTORY_TURNS * 2)   # user/assistant 쌍
        self._lock = threading.Lock()

    # ── 로딩 ───────────────────────────────────────────────
    def load_async(self):
        if not self.enabled:
            print("[Assistant] 비활성화 (--no-llm)")
            return
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            self.error = "llama_cpp 미설치"
            print(f"[Assistant] {self.error} → Gemma 비활성화")
            return
        if not config.GEMMA_MODEL_PATH.exists():
            self.error = f"모델 없음: {config.GEMMA_MODEL_PATH}"
            print(f"[Assistant] {self.error} → Gemma 비활성화")
            return

        t0 = time.time()
        kwargs = dict(
            model_path=str(config.GEMMA_MODEL_PATH),
            n_gpu_layers=-1,
            n_ctx=config.LLM_CONTEXT,
            n_batch=32,
            n_ubatch=32,
            verbose=False,
        )
        if self.use_image:
            if config.GEMMA_MMPROJ_PATH.exists():
                from llama_cpp.llama_chat_format import Gemma4ChatHandler
                kwargs["chat_handler"] = Gemma4ChatHandler(clip_model_path=str(config.GEMMA_MMPROJ_PATH))
            else:
                print("[Assistant] mmproj 없음 → 텍스트 전용으로 동작")
                self.use_image = False
        try:
            self.llm = Llama(**kwargs)
        except Exception as e:
            self.error = f"로드 실패: {e}"
            print(f"[Assistant] {self.error}")
            return
        self.ready = True
        mode = "텍스트+이미지" if self.use_image else "텍스트"
        print(f"[Assistant] Gemma 로드 완료 ({mode}, {time.time() - t0:.1f}s)")

    # ── 질의 ───────────────────────────────────────────────
    def ask(self, question, scene, frame=None) -> str:
        if not self.ready:
            return "언어 모델이 아직 준비되지 않았습니다." if self.enabled and not self.error \
                else "현재 대화 기능을 사용할 수 없습니다."

        context = scene_to_text(scene)
        question = question or "지금 주변 상황을 알려줘."
        user_text = f"Context:\n{context}\n\nQuestion:\n{question}"

        if self.use_image and frame is not None:
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": encode_image(frame)}},
            ]
        else:
            user_content = user_text

        messages = [{"role": "system", "content": config.LLM_SYSTEM_PROMPT}]
        messages += list(self.history)
        messages.append({"role": "user", "content": user_content})

        print(f"[Assistant] Q: {question}\n[Assistant] Context:\n{context}")
        with self._lock:   # llama.cpp 인스턴스는 동시 호출 불가
            t0 = time.time()
            resp = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
        answer = resp["choices"][0]["message"]["content"].strip()
        print(f"[Assistant] A ({time.time() - t0:.1f}s): {answer}")

        # Sliding Window 메모리: 이미지는 빼고 텍스트만 저장 (컨텍스트 절약)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        return answer


    # ── 물건 찾기: 키워드 매핑 실패 시 Gemma로 대상 클래스 추출 ──
    def extract_find_target(self, text, valid_classes):
        """사용자 문장에서 찾으려는 물건을 COCO 클래스 이름 하나로 추출. 실패하면 None."""
        if not self.ready:
            return None
        prompt = (
            "Instruction:\n사용자 문장에서 찾으려는 물건을 아래 목록 중 하나의 영어 이름으로만 답하시오.\n"
            f"목록: {', '.join(sorted(valid_classes))}\n\n"
            "Constraint:\n목록에 없으면 none 이라고만 답하시오. 다른 말은 하지 마시오.\n\n"
            "Output Format:\n영어 클래스 이름 하나."
        )
        with self._lock:
            resp = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": text}],
                max_tokens=8,
                temperature=0.0,
            )
        answer = resp["choices"][0]["message"]["content"].strip().lower().strip(" .\"'")
        print(f"[Assistant] find target from Gemma: {answer!r}")
        return answer if answer in valid_classes else None


def encode_image(frame, max_side=512):
    """프레임을 축소 후 base64 data URL로 (mmproj 입력 토큰/시간 절약)"""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("이미지 인코딩 실패")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
