# -*- coding: utf-8 -*-
# Web 服务器模块 (web_server.py)
# 功能：提供本地 Web 界面，实时显示视频流和识别结果
# 职责：
#   1. 提供 Web 页面访问
#   2. 推送实时视频流（MJPEG）
#   3. 通过 WebSocket 推送识别结果数据
#   4. 接收其他硬件设备数据（血压、心率等）
#   5. 使用 SQLite 存储历史训练数据
import cv2
import json
import logging
import sqlite3
import threading
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart_hospital_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


def init_db(db_path='rehab.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                  score REAL, 
                  knee_angle REAL, 
                  hip_angle REAL,
                  heart_rate INTEGER, 
                  systolic_bp INTEGER, 
                  diastolic_bp INTEGER,
                  spo2 INTEGER,
                  action_type TEXT, 
                  feedback TEXT)''')
    conn.commit()
    conn.close()
    logger.info(f"数据库初始化完成: {db_path}")


class WebServer:
    def __init__(self, config):
        self.config = config
        self.port = config.get('web_port', 5000)
        self.host = config.get('web_host', '0.0.0.0')
        self.db_path = config.get('db_path', 'rehab.db')

        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.latest_result = None
        self.result_lock = threading.Lock()

        self.vitals_data = {}
        self.vitals_lock = threading.Lock()

        init_db(self.db_path)

        self._setup_routes()

    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup_routes(self):
        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @app.route('/api/latest_result')
        def latest_result():
            with self.result_lock:
                return jsonify(self.latest_result or {})

        @app.route('/api/vitals', methods=['POST'])
        def post_vitals():
            data = request.json
            if not data:
                return jsonify({'error': '缺少数据'}), 400

            with self.vitals_lock:
                self.vitals_data.update(data)

            socketio.emit('vitals_update', self.vitals_data)

            return jsonify({'status': 'success'})

        @app.route('/api/history')
        def get_history():
            period = request.args.get('period', 'today')

            now = datetime.now()
            if period == 'today':
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == 'week':
                start_time = now - timedelta(days=7)
            elif period == 'month':
                start_time = now - timedelta(days=30)
            else:
                start_time = now - timedelta(days=1)

            conn = self._get_db()
            c = conn.cursor()
            c.execute('''SELECT timestamp, score, knee_angle, hip_angle, heart_rate, 
                                systolic_bp, diastolic_bp, spo2, action_type, feedback 
                         FROM training_data 
                         WHERE timestamp >= ? 
                         ORDER BY timestamp ASC''', (start_time.isoformat(),))
            records = [dict(row) for row in c.fetchall()]

            c.execute('''SELECT COUNT(*) as total_sessions, 
                                AVG(score) as avg_score, 
                                MAX(score) as max_score, 
                                AVG(heart_rate) as avg_heart_rate 
                         FROM training_data 
                         WHERE timestamp >= ?''', (start_time.isoformat(),))
            stats = dict(c.fetchone())

            conn.close()

            return jsonify({'history': records, 'stats': stats})

    def _generate_frames(self):
        while True:
            with self.frame_lock:
                if self.current_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', self.current_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            import time
            time.sleep(0.03)

    def update_frame(self, frame):
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None

    def update_result(self, result):
        with self.result_lock:
            self.latest_result = result

        self._save_to_db(result)

        socketio.emit('result_update', result)

    def _save_to_db(self, result):
        try:
            result_data = result.get('result', {})
            joint_angles = result_data.get('joint_angles', {})

            conn = self._get_db()
            c = conn.cursor()
            c.execute('''INSERT INTO training_data 
                         (score, knee_angle, hip_angle, heart_rate, systolic_bp, diastolic_bp, spo2, action_type, feedback) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (result_data.get('quality_score', 0),
                       joint_angles.get('left_knee', 0),
                       joint_angles.get('left_hip', 0),
                       self.vitals_data.get('heart_rate'),
                       self.vitals_data.get('systolic'),
                       self.vitals_data.get('diastolic'),
                       self.vitals_data.get('spo2'),
                       result_data.get('action_type', ''),
                       result_data.get('feedback', '')))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存数据库失败: {e}")

    def run(self):
        logger.info(f"Web 服务器启动在 http://{self.host}:{self.port}")
        socketio.run(app, host=self.host, port=self.port, debug=False)

    def run_async(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread
