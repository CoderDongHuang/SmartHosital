# 可视化模块 (visualizer.py)
# 功能：在原图上绘制骨架和文字提示，用于本地显示和推流
# 职责：
#   1. 手写极简绘图代码（比 MediaPipe 官方函数快且省内存）
#   2. 只画线段（关节连线），不画填充圆点
#   3. 在画面角落显示动作评分和反馈
#   4. 跳帧绘制：每 3 帧绘制一次，降低 CPU 占用
# 硬件适配：树莓派 5 2GB，CPU 绘图，单帧 < 5ms
import cv2
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
import os

logger = logging.getLogger(__name__)


# 主要关节连接线（只画关键肢体，不画面部细节）
MAIN_CONNECTIONS = [
    # 左臂
    (11, 13),  # 左肩 -> 左肘
    (13, 15),  # 左肘 -> 左腕
    # 右臂
    (12, 14),  # 右肩 -> 右肘
    (14, 16),  # 右肘 -> 右腕
    # 躯干
    (11, 12),  # 左肩 -> 右肩
    (11, 23),  # 左肩 -> 左髋
    (12, 24),  # 右肩 -> 右髋
    (23, 24),  # 左髋 -> 右髋
    # 左腿
    (23, 25),  # 左髋 -> 左膝
    (25, 27),  # 左膝 -> 左踝
    # 右腿
    (24, 26),  # 右髋 -> 右膝
    (26, 28),  # 右膝 -> 右踝
]

# 关键点中文名称
LANDMARK_NAMES = [
    '鼻子', '左眼内', '左眼', '左眼外',
    '右眼内', '右眼', '右眼外',
    '左耳', '右耳', '左嘴角', '右嘴角',
    '左肩', '右肩', '左肘', '右肘',
    '左腕', '右腕', '左小指', '右小指',
    '左食指', '右食指', '左拇指', '右拇指',
    '左髋', '右髋', '左膝', '右膝',
    '左踝', '右踝', '左脚跟', '右脚跟',
    '左脚尖', '右脚尖'
]


def get_chinese_font(font_size=24):
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    return ImageFont.load_default()


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.frame_skip = config.get('draw_frame_skip', 3)
        self.frame_counter = 0
        self.line_color = tuple(config.get('line_color', [0, 255, 0]))
        self.line_thickness = config.get('line_thickness', 2)
        self.text_color = tuple(config.get('text_color', [0, 255, 255]))
        self.text_size = config.get('text_size', 24)

    def draw(self, image, processed_result):
        if image is None or processed_result is None:
            return image

        self.frame_counter += 1

        if self.frame_counter % self.frame_skip != 0:
            return image

        try:
            result_data = processed_result.get('result', {})
            persons = result_data.get('persons', [])

            if not persons:
                cv2.putText(image, "未检测到人体", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return image

            for person in persons:
                keypoints = person.get('keypoints', [])
                kp_dict = {kp['index']: kp for kp in keypoints if kp['confidence'] > 0.5}

                h, w = image.shape[:2]

                for start_idx, end_idx in MAIN_CONNECTIONS:
                    if start_idx in kp_dict and end_idx in kp_dict:
                        start = kp_dict[start_idx]
                        end = kp_dict[end_idx]

                        pt1 = (int(start['x'] * w), int(start['y'] * h))
                        pt2 = (int(end['x'] * w), int(end['y'] * h))

                        cv2.line(image, pt1, pt2, self.line_color, self.line_thickness)

            self._draw_info(image, processed_result)

            return image
        except Exception as e:
            logger.error(f"绘制失败: {e}")
            return image

    def _draw_info(self, image, processed_result):
        result_data = processed_result.get('result', {})
        action_type = result_data.get('action_type', '未知')
        quality_score = result_data.get('quality_score', 0)
        feedback = result_data.get('feedback', '')

        persons_count = result_data.get('persons_count', 0)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = get_chinese_font(self.text_size)

        y_offset = 30
        line_height = self.text_size + 5

        draw.text((10, y_offset), f"检测到 {persons_count} 人", fill=self.text_color, font=font)
        y_offset += line_height

        draw.text((10, y_offset), f"动作: {action_type}", fill=self.text_color, font=font)
        y_offset += line_height

        draw.text((10, y_offset), f"评分: {quality_score:.1f}", fill=self.text_color, font=font)
        y_offset += line_height

        if feedback:
            draw.text((10, y_offset), feedback, fill=(255, 255, 0), font=font)

        image[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
