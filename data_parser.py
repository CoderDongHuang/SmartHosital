# 数据解析与校验模块 (data_parser.py)
# 功能：系统的"质量检查员"，负责解析收到的 JSON 数据并校验数据完整性
# 职责：
#   1. 强校验：检查字段是否完整、图片格式是否正确、时间戳是否合理
#   2. 数据转换：将图片的 base64 编码或 URL 转换成 OpenCV 能处理的格式
#   3. 错误处理：数据格式错误或缺失关键字段时直接丢弃并记录日志
#   4. 支持多种图像来源：base64、URL、字节流、numpy 数组
import base64
import numpy as np
import cv2
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataParser:
    REQUIRED_FIELDS = ['image', 'timestamp', 'device_id']

    def __init__(self, config=None):
        self.config = config or {}
        self.max_image_size = self.config.get('max_image_size', 1920 * 1080 * 3)
        self.allowed_formats = self.config.get('allowed_formats', ['jpg', 'jpeg', 'png'])

    def parse_and_validate(self, raw_data):
        if not isinstance(raw_data, dict):
            logger.error("数据格式无效：期望字典类型")
            return None

        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in raw_data]
        if missing_fields:
            logger.error(f"缺少必填字段: {missing_fields}")
            return None

        if not self._validate_timestamp(raw_data.get('timestamp')):
            logger.error("时间戳无效")
            return None

        image_data = self._process_image(raw_data.get('image'))
        if image_data is None:
            return None

        parsed_data = {
            'image': image_data,
            'timestamp': raw_data['timestamp'],
            'device_id': raw_data['device_id'],
            'data_id': raw_data.get('data_id', f"{raw_data['device_id']}_{raw_data['timestamp']}"),
            'metadata': raw_data.get('metadata', {})
        }

        logger.debug(f"数据解析成功: {parsed_data['data_id']}")
        return parsed_data

    def _validate_timestamp(self, timestamp):
        try:
            if isinstance(timestamp, (int, float)):
                ts = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                return False

            now = datetime.now()
            time_diff = abs((now - ts).total_seconds())
            if time_diff > 3600:
                logger.warning(f"时间戳过旧或来自未来: {timestamp}")
                return False

            return True
        except Exception as e:
            logger.error(f"时间戳验证失败: {e}")
            return False

    def _process_image(self, image_data):
        if image_data is None:
            logger.error("未提供图像数据")
            return None

        try:
            if isinstance(image_data, str):
                if image_data.startswith('http'):
                    return self._load_image_from_url(image_data)
                elif image_data.startswith('data:image'):
                    base64_data = image_data.split(',')[1]
                    return self._decode_base64_image(base64_data)
                else:
                    return self._decode_base64_image(image_data)
            elif isinstance(image_data, bytes):
                return self._decode_bytes_image(image_data)
            elif isinstance(image_data, np.ndarray):
                if image_data.size * image_data.itemsize > self.max_image_size:
                    logger.error("图像过大")
                    return None
                return image_data
            else:
                logger.error(f"不支持的图像类型: {type(image_data)}")
                return None
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return None

    def _decode_base64_image(self, base64_str):
        try:
            image_bytes = base64.b64decode(base64_str)
            if len(image_bytes) > self.max_image_size:
                logger.error("Base64图像过大")
                return None
            return self._decode_bytes_image(image_bytes)
        except Exception as e:
            logger.error(f"Base64解码失败: {e}")
            return None

    def _decode_bytes_image(self, image_bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("图像解码失败")
                return None
            return image
        except Exception as e:
            logger.error(f"字节解码失败: {e}")
            return None

    def _load_image_from_url(self, url):
        try:
            import requests
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return self._decode_bytes_image(response.content)
            else:
                logger.error(f"从URL加载图像失败: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"URL加载失败: {e}")
            return None
