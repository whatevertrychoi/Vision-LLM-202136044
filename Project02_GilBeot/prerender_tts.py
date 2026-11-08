"""
고정 안내 문구를 Piper로 미리 wav 생성 (Jetson에서 최초 1회 실행 권장)
main.py 도 시작 시 없는 것만 자동 생성하지만, 미리 해두면 시연 시작이 빨라진다.

  python prerender_tts.py           # 없는 것만 생성
  python prerender_tts.py --play    # 생성 후 하나씩 재생해서 스피커 확인
"""

import argparse
import subprocess

import config
from audio import Speaker

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    sp = Speaker()
    if sp.backend != "piper":
        raise SystemExit(f"Piper 환경을 찾을 수 없습니다: {config.PIPER_PYTHON}, {config.PIPER_MODEL}")
    sp.prerender()
    print(f"완료: {config.TTS_CACHE_DIR}")

    if args.play:
        for key, p in config.PHRASES.items():
            wav = Speaker._cache_path(key, p["text"])
            print(f"▶ {key}: {p['text']}")
            cmd = ["aplay", "-q"] + (["-D", config.SPEAKER_DEVICE] if config.SPEAKER_DEVICE else []) + [str(wav)]
            subprocess.run(cmd, check=False)
