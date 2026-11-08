"""
음성 입출력 (06: Whisper.cpp STT, Piper TTS)

Speaker (TTS)
  - 고정 경고 문구는 시작 시 Piper로 wav를 미리 만들어 두고 aplay만 실행 → 거의 0초 지연
    (06 방식처럼 매번 Piper를 호출하면 모델 로드 때문에 1~2초 이상 걸림)
  - 우선순위 큐: 급한 경고가 오면 재생 중인 덜 급한 음성(예: Gemma 답변)을 끊고 먼저 말한다
  - 오래 기다린 경고는 버린다 (상황이 이미 바뀌었을 수 있음)

Listener (STT)
  - arecord(pasuspender로 PulseAudio 점유 해제) → whisper-cli -l ko

Jetson이 아닌 환경(노트북)에서는 콘솔 출력/키보드 입력으로 자동 폴백.
"""

import hashlib
import heapq
import itertools
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import config


# ══════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════
class Speaker:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.muted = False
        self.has_aplay = shutil.which("aplay") is not None
        self.has_piper = config.PIPER_PYTHON.exists() and config.PIPER_MODEL.exists()
        self.backend = "piper" if (self.has_aplay and self.has_piper) else "console"
        if self.backend == "console":
            print("[Speaker] Piper/aplay 없음 → 콘솔 출력 모드 (노트북 테스트용)")
        else:
            print(f"[Speaker] Piper TTS: {config.PIPER_MODEL.name}")

        self._heap = []
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._proc = None                   # 지금 돌고 있는 외부 프로세스 (piper 또는 aplay)
        self._preempt = threading.Event()   # "지금 하던 말 그만" 신호
        self._playing_priority = None
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)

    # ── 캐시 ───────────────────────────────────────────────
    @staticmethod
    def _cache_path(key, text):
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        return config.TTS_CACHE_DIR / f"{key}_{digest}.wav"

    def prerender(self):
        """config.PHRASES 를 모두 wav로 생성 (문구가 바뀌면 해시가 바뀌어 자동 재생성)"""
        if self.backend != "piper":
            return
        config.TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for key, p in config.PHRASES.items():
            path = self._cache_path(key, p["text"])
            if path.exists():
                continue
            t0 = time.time()
            self._synthesize(p["text"], path)
            print(f"[Speaker] cached '{key}' ({time.time() - t0:.1f}s)")

    def _synthesize(self, text, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(config.PIPER_PYTHON), "-m", "piper",
             "-m", str(config.PIPER_MODEL),
             "-f", str(out_path),
             "--", text],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # ── 공개 API ───────────────────────────────────────────
    def start(self):
        self._thread.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._kill_current()

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self._kill_current()
        return self.muted

    def say_key(self, key):
        """고정 문구(키) 재생. 우선순위 2 이하(보행 경고)는 늦으면 버린다."""
        p = config.PHRASES[key]
        self._push(p["priority"], {"key": key, "text": p["text"], "alert": p["priority"] <= 2})

    def say_text(self, text, priority=config.LLM_PRIORITY):
        """임의 문장 재생 — Gemma 답변용 (실시간 합성)"""
        text = clean_for_tts(text)
        if text:
            self._push(priority, {"key": None, "text": text, "alert": False})

    def wait_idle(self, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            with self._cv:
                if not self._heap and not self._busy:
                    return True
            time.sleep(0.05)
        return False

    # ── 내부 ───────────────────────────────────────────────
    def _push(self, priority, item):
        if not self.enabled or self.muted:
            return
        item["t"] = time.time()
        with self._cv:
            # 같은 경고가 이미 대기 중이면 중복 추가하지 않음
            if item["key"] and any(i[2]["key"] == item["key"] for i in self._heap):
                return
            heapq.heappush(self._heap, (priority, next(self._seq), item))
            # 더 급한 것이 들어오면 현재 음성 중단 (선점)
            if self._playing_priority is not None and priority < self._playing_priority:
                self._kill_current()
            self._cv.notify()

    def _kill_current(self):
        """하던 말 중단: 신호를 세우고, 돌고 있는 piper/aplay 프로세스를 종료"""
        self._preempt.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()

    def _worker(self):
        while True:
            with self._cv:
                while not self._heap and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                priority, _, item = heapq.heappop(self._heap)
                self._busy = True
                self._playing_priority = priority
                self._preempt.clear()
            try:
                if item["alert"] and time.time() - item["t"] > config.ALERT_STALE_SEC:
                    continue  # 너무 늦은 경고는 버림
                self._play(item)
            except Exception as e:  # 오디오 오류로 전체가 죽지 않도록
                print(f"[Speaker] error: {e}")
            finally:
                with self._cv:
                    self._busy = False
                    self._playing_priority = None
                    self._proc = None

    def _run(self, cmd):
        """
        외부 프로세스를 실행하고 끝날 때까지 기다린다. 선점되면 False.
        프로세스를 self._proc 에 등록하는 것과 선점 확인을 같은 락 안에서 해서,
        '확인 직후 실행 직전'에 경고가 끼어드는 경우도 놓치지 않는다.
        """
        with self._cv:
            if self._preempt.is_set():
                return False
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._proc.wait()
        return not self._preempt.is_set()

    def _play(self, item):
        text = item["text"]
        print(f"[TTS] {text}")
        if self.backend == "console":
            # 말하는 시간 흉내. 0.05초마다 선점 신호를 확인 (선점 동작 확인용)
            end = time.time() + min(4.0, 0.08 * len(text))
            while time.time() < end:
                if self._preempt.is_set():
                    print("[TTS] ...(중단됨: 더 급한 경고)")
                    return
                time.sleep(0.05)
            return

        # 1) 음성 파일 준비: 고정 문구는 캐시 사용, Gemma 답변은 지금 합성
        if item["key"]:
            wav = self._cache_path(item["key"], text)
        else:
            wav = config.RUNTIME_DIR / "llm_answer.wav"
        if not item["key"] or not wav.exists():
            wav.parent.mkdir(parents=True, exist_ok=True)
            ok = self._run([str(config.PIPER_PYTHON), "-m", "piper",
                            "-m", str(config.PIPER_MODEL), "-f", str(wav), "--", text])
            if not ok:
                print("[TTS] ...(합성 중 중단됨: 더 급한 경고)")
                return

        # 2) 재생
        cmd = ["aplay", "-q"]
        if config.SPEAKER_DEVICE:
            cmd += ["-D", config.SPEAKER_DEVICE]
        cmd.append(str(wav))
        if not self._run(cmd):
            print("[TTS] ...(재생 중 중단됨: 더 급한 경고)")


def clean_for_tts(text: str) -> str:
    """Gemma 출력에서 읽기 어려운 기호 제거"""
    # 한글, 영문/숫자, 기본 문장부호만 남김 (마크다운 기호·이모지 제거)
    text = re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s.,?!%~-]", " ", text)
    text = text.replace("~", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
# STT
# ══════════════════════════════════════════════════════════════
class Listener:
    def __init__(self):
        self.has_tools = (
            shutil.which("arecord") is not None
            and config.WHISPER_CLI is not None
            and Path(config.WHISPER_MODEL).exists()
        )
        if self.has_tools:
            print(f"[Listener] whisper-cli: {config.WHISPER_CLI}, mic: {config.MIC_DEVICE}")
        else:
            print("[Listener] arecord/whisper 없음 → 콘솔 입력 모드 (노트북 테스트용)")

    def listen(self) -> str:
        if not self.has_tools:
            try:
                return input("[STT 대체] 질문을 입력하세요 > ").strip()
            except EOFError:
                return ""

        config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        rec = ["arecord", "-D", config.MIC_DEVICE, "-f", "S16_LE", "-r", "16000",
               "-c", "1", "-d", str(config.RECORD_SECONDS), str(config.MIC_WAV_PATH)]
        if shutil.which("pasuspender"):
            rec = ["pasuspender", "--"] + rec
        subprocess.run(rec, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        result = subprocess.run(
            [str(config.WHISPER_CLI), "-m", str(config.WHISPER_MODEL),
             "-f", str(config.MIC_WAV_PATH), "-l", "ko", "--no-timestamps"],
            text=True, capture_output=True, check=True,
        )
        text = result.stdout
        text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)   # [BLANK_AUDIO], (음악) 등 제거
        return re.sub(r"\s+", " ", text).strip()
