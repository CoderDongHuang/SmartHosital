# 数据模拟模块 (data_simulator.py)
# 功能：系统的"离线演示器"，比赛保命符！当网络断连时自动切换本地模拟数据
# 职责：
#   1. 从本地文件夹读取预存的图片/视频进行推理
#   2. 模拟上报结果，标记为模拟数据
#   3. 支持循环播放和单次播放两种模式
#   4. 一键切换"线上真实数据模式"和"线下模拟数据模式"
import os
import cv2
import numpy as np
import base64
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class DataSimulator:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('sim_data_dir', './sim_data')
        self.interval = config.get('sim_interval', 2)
        self.loop = config.get('sim_loop', True)
        self.current_index = 0
        self.data_list = []
        self.is_running = False

    def initialize(self):
        self.data_list = self._load_sim_data()
        if not self.data_list:
            logger.warning("未找到模拟数据")
            return False

        logger.info(f"已加载 {len(self.data_list)} 条模拟数据")
        return True

    def _load_sim_data(self):
        data_list = []

        if not os.path.exists(self.data_dir):
            logger.warning(f"模拟数据目录不存在: {self.data_dir}")
            return data_list

        for filename in sorted(os.listdir(self.data_dir)):
            filepath = os.path.join(self.data_dir, filename)

            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                data_list.append({
                    'type': 'image',
                    'path': filepath,
                    'filename': filename
                })
            elif ext in ['.mp4', '.avi', '.mov']:
                data_list.append({
                    'type': 'video',
                    'path': filepath,
                    'filename': filename
                })

        return data_list

    def get_next_data(self):
        if not self.data_list:
            return None

        if self.current_index >= len(self.data_list):
            if self.loop:
                self.current_index = 0
                logger.info("模拟数据循环重新开始")
            else:
                logger.info("模拟数据已用完")
                return None

        data_item = self.data_list[self.current_index]
        self.current_index += 1

        try:
            if data_item['type'] == 'image':
                return self._process_image(data_item)
            elif data_item['type'] == 'video':
                return self._process_video_frame(data_item)
        except Exception as e:
            logger.error(f"处理模拟数据失败: {e}")
            return None

    def _process_image(self, data_item):
        image = cv2.imread(data_item['path'])
        if image is None:
            logger.error(f"读取图像失败: {data_item['path']}")
            return None

        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'image': image_base64,
            'timestamp': datetime.now().isoformat(),
            'device_id': 'simulator_001',
            'data_id': f"sim_{data_item['filename']}_{int(time.time())}",
            'metadata': {
                'source': 'simulation',
                'filename': data_item['filename'],
                'type': 'image'
            }
        }

    def _process_video_frame(self, data_item):
        cap = cv2.VideoCapture(data_item['path'])
        if not cap.isOpened():
            logger.error(f"打开视频失败: {data_item['path']}")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = int(time.time()) % frame_count

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error("读取视频帧失败")
            return None

        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'image': image_base64,
            'timestamp': datetime.now().isoformat(),
            'device_id': 'simulator_001',
            'data_id': f"sim_{data_item['filename']}_frame{frame_idx}_{int(time.time())}",
            'metadata': {
                'source': 'simulation',
                'filename': data_item['filename'],
                'type': 'video_frame',
                'frame_index': frame_idx
            }
        }

    def reset(self):
        self.current_index = 0
        logger.info("模拟数据已重置")

    def get_status(self):
        return {
            'total_items': len(self.data_list),
            'current_index': self.current_index,
            'remaining': len(self.data_list) - self.current_index,
            'loop_enabled': self.loop,
            'data_dir': self.data_dir
        }
