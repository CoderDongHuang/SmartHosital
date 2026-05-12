# -*- coding: utf-8 -*-
# Web 服务器模块 (web_server.py)
# 功能：提供本地 Web 界面，实时显示视频流和识别结果
# 职责：
#   1. 提供 Web 页面访问
#   2. 推送实时视频流（MJPEG）
#   3. 通过 WebSocket 推送识别结果数据
#   4. 接收其他硬件设备数据（血压、心率等）
import cv2
import json
import logging
import threading
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from datetime import datetime

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart_hospital_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


class WebServer:
    def __init__(self, config):
        self.config = config
        self.port = config.get('web_port', 5000)
        self.host = config.get('web_host', '0.0.0.0')

        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.latest_result = None
        self.result_lock = threading.Lock()

        self.device_data = {}
        self.device_data_lock = threading.Lock()

        self._setup_routes()

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

        @app.route('/api/device_data', methods=['GET'])
        def get_device_data():
            with self.device_data_lock:
                return jsonify(self.device_data)

        @app.route('/api/device_data', methods=['POST'])
        def post_device_data():
            data = request.json
            if not data or 'device_id' not in data:
                return jsonify({'error': '缺少 device_id'}), 400

            device_id = data['device_id']
            with self.device_data_lock:
                self.device_data[device_id] = {
                    'data': data,
                    'updated_at': datetime.now().isoformat()
                }

            socketio.emit('device_data_update', {
                'device_id': device_id,
                'data': data
            })

            return jsonify({'status': 'success'})

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

        socketio.emit('result_update', result)

    def run(self):
        logger.info(f"Web 服务器启动在 http://{self.host}:{self.port}")
        socketio.run(app, host=self.host, port=self.port, debug=False)

    def run_async(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread
