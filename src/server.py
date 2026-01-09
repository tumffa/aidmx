import sys
import json
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging

from src.utils.config import load_config, save_config

from src.services.queue_manager import QueueManager
from src.services.data_manager import DataManager
from src.services.qlc_manager import QLCManager


class ThreadSafeLog:
    def __init__(self, max_lines=5000):
        self.lines = deque(maxlen=max_lines)
        self.buffer = ""
        self.lock = threading.Lock()
        self.seq = 0

    def write(self, s: str):
        if not s:
            return
        with self.lock:
            self.buffer += str(s)
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                self.seq += 1
                self.lines.append((self.seq, line))

    def flush(self):
        pass

    def get_since(self, cursor: int, limit: int = 500):
        with self.lock:
            data = [(i, t) for (i, t) in self.lines if i > cursor]
            # Trim to limit
            if len(data) > limit:
                data = data[-limit:]
            next_cursor = data[-1][0] if data else cursor
            return data, next_cursor

def create_app():
    app = Flask(
        __name__,
        static_folder=str((Path(__file__).resolve().parents[1] / "web" / "dist")),
        static_url_path="/"
    )
    CORS(app)

    # Silence Flask/Werkzeug request logs in console panel
    logging.getLogger("werkzeug").disabled = True
    logging.getLogger("flask.app").disabled = True

    # Shared services
    cfg_path = Path("config.json")
    config = load_config(cfg_path)
    setup_path = Path(config["setup_path"]).resolve()
    setupfile_name = setup_path.stem
    data_manager = DataManager(config)
    qlc = QLCManager(setupfile_name, setup_path)
    qm = QueueManager(setupfile_name, data_manager, qlc)

    # Logging
    tslog = ThreadSafeLog(max_lines=10000)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tslog
    sys.stderr = tslog

    # Task threads
    analyze_thread = {"t": None}
    play_thread = {"t": None}

    @app.route("/api/status", methods=["GET"])
    def status():
        a_running = analyze_thread["t"].is_alive() if analyze_thread["t"] else False
        p_running = play_thread["t"].is_alive() if play_thread["t"] else False
        return jsonify({"analyze_running": a_running, "play_running": p_running})

    @app.route("/api/logs", methods=["GET"])
    def logs():
        try:
            cursor = int(request.args.get("cursor", 0))
        except Exception:
            cursor = 0
        lines, next_cursor = tslog.get_since(cursor)
        return jsonify({
            "lines": [{"id": i, "text": t} for (i, t) in lines],
            "next_cursor": next_cursor,
        })

    @app.route("/api/analyze", methods=["POST"])
    def start_analyze():
        body = request.get_json(force=True)
        audio_name = body.get("audio_name")
        strobes = bool(body.get("strobes", False))
        simple = bool(body.get("simple", False))
        qlc_delay = float(body.get("qlc_delay", 0.0))
        qlc_lag = float(body.get("qlc_lag", 1.0))

        if not audio_name:
            return jsonify({"error": "audio_name required"}), 400

        # Avoid multiple concurrent analyze runs
        if analyze_thread["t"] and analyze_thread["t"].is_alive():
            return jsonify({"status": "already_running"}), 409

        qm.cancel_event.clear()

        def run():
            try:
                qm.analyze_file(audio_name, strobes=strobes, simple=simple, qlc_delay=qlc_delay, qlc_lag=qlc_lag)
            except Exception as e:
                print(f"Error: {e}")

        t = threading.Thread(target=run, daemon=True)
        analyze_thread["t"] = t
        t.start()
        return jsonify({"status": "started"})

    @app.route("/api/cancel/analyze", methods=["POST"])
    def cancel_analyze():
        qm.request_cancel_analysis()
        return jsonify({"status": "cancel_requested"})

    @app.route("/api/play", methods=["POST"])
    def start_play():
        body = request.get_json(force=True)
        audio_name = body.get("audio_name")
        delay = float(body.get("delay", 0.0))
        universe = int(body.get("universe", 1))
        start_at_sec = float(body.get("start_at_sec", 0.0))

        if not audio_name:
            return jsonify({"error": "audio_name required"}), 400

        if play_thread["t"] and play_thread["t"].is_alive():
            return jsonify({"status": "already_running"}), 409

        qm.cancel_event.clear()

        def run():
            try:
                print(f"Starting playback for '{audio_name}' (delay={delay}, universe={universe}, start={start_at_sec})")
                qm.play_ola_show(audio_name, delay, universe, start_at_sec=start_at_sec)
                print("Playback finished.")
            except Exception as e:
                print(f"Error: {e}")

        t = threading.Thread(target=run, daemon=True)
        play_thread["t"] = t
        t.start()
        return jsonify({"status": "started"})

    @app.route("/api/cancel/play", methods=["POST"])
    def cancel_play():
        qm.request_cancel_playback()
        return jsonify({"status": "cancel_requested"})

    @app.route("/api/universe", methods=["GET"])
    def get_universe():
        cfg = load_config(cfg_path)
        uni = cfg.get("universe", {})
        return jsonify({"universe": uni})

    @app.route("/api/universe", methods=["PUT"])
    def put_universe():
        payload = request.get_json(force=True)
        new_uni = payload.get("universe")
        if not isinstance(new_uni, dict):
            return jsonify({"error": "universe must be an object"}), 400
        cfg = load_config(cfg_path)
        cfg["universe"] = new_uni
        try:
            save_config(cfg_path, cfg)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        # Also update in-memory
        data_manager.universe = new_uni
        return jsonify({"status": "ok"})

    @app.route("/api/ping", methods=["GET"])
    def ping():
        return jsonify({"status": "ok"})

    @app.route("/api/shutdown", methods=["POST"])
    def shutdown():
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        return jsonify({"status": "bye"})

    # Static file routes (serve React app)
    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    @app.route("/<path:path>")
    def static_proxy(path):
        dist = Path(app.static_folder)
        file_path = dist / path
        if file_path.exists():
            return send_from_directory(dist, path)
        # SPA fallback
        return app.send_static_file("index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    # Disable debug and reloader to avoid extra console noise
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)