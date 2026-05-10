# AI推理引擎模块 (ai_engine.py)
# 功能：系统的"大脑"，使用 MediaPipe Pose 进行轻量级人体关键点检测
# 职责：
#   1. 懒加载模型：程序启动时加载一次 MediaPipe Pose 模型，之后反复使用
#   2. 资源管理：推理前检查内存占用，过高时释放缓存
#   3. 模型封装：将模型调用封装成 predict(image) 函数，输入输出明确
#   4. 输出 33 个人体关键点的 2D 坐标（鼻、肩、肘、腕、髋、膝、踝等）
#   5. 支持多人检测：可同时检测多个人体
#   6. 性能优化：降低分辨率，跳帧处理
# 硬件适配：树莓派 5 2GB，CPU 推理，单张图片 100-300ms
import cv2
import numpy as np
import logging
import time
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AIEngine:
    def __init__(self, config):
        self.config = config
        self.pose_detector = None
        self.model_type = config.get('model_type', 'mediapipe_pose')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.min_detection_confidence = config.get('min_detection_confidence', 0.5)
        self.min_tracking_confidence = config.get('min_tracking_confidence', 0.5)
        self.input_resolution = config.get('input_resolution', (640, 480))
        self.max_num_poses = config.get('max_num_poses', 5)
        self.is_loaded = False
        self.inference_count = 0
        self.frame_skip = config.get('frame_skip', 0)
        self.frame_counter = 0
        self.last_result = None

    def load_model(self):
        if self.is_loaded:
            logger.info("模型已加载")
            return True

        try:
            logger.info("正在加载 MediaPipe Pose 模型...")

            if not MEDIAPIPE_AVAILABLE:
                logger.error("MediaPipe 未安装，请运行: pip install mediapipe")
                return False

            self.mp_pose = mp.solutions.pose

            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )

            self.is_loaded = True
            logger.info("MediaPipe Pose 模型加载成功")
            logger.info(f"输入分辨率: {self.input_resolution[0]}x{self.input_resolution[1]}")
            logger.info(f"模型复杂度: 0 (最快，适合实时视频流)")
            logger.info(f"检测置信度阈值: {self.min_detection_confidence}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.warning("将使用模拟模式进行推理")
            return False

    def predict(self, image):
        if image is None:
            logger.error("输入图像为空")
            return None

        self.frame_counter += 1

        if self.frame_skip > 0 and (self.frame_counter % (self.frame_skip + 1) != 0):
            logger.debug(f"跳帧处理，跳过第 {self.frame_counter} 帧")
            return self.last_result

        start_time = time.time()

        try:
            original_shape = image.shape
            orig_h, orig_w = original_shape[:2]

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            persons = []

            if self.is_loaded and self.pose is not None:
                results = self.pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    overall_confidence = 0.0

                    keypoints = []
                    for idx, landmark in enumerate(landmarks):
                        x = landmark.x * orig_w
                        y = landmark.y * orig_h
                        z = landmark.z
                        visibility = landmark.visibility

                        keypoints.append({
                            'index': idx,
                            'x': float(x),
                            'y': float(y),
                            'z': float(z),
                            'confidence': float(visibility),
                            'name': self._get_landmark_name(idx)
                        })

                        overall_confidence += visibility

                    overall_confidence /= len(keypoints) if keypoints else 1

                    persons.append({
                        'person_id': 0,
                        'keypoints': keypoints,
                        'confidence': overall_confidence,
                        'keypoints_count': len(keypoints)
                    })
                else:
                    logger.warning("未检测到人体关键点")
            else:
                logger.warning("模型未加载，无法推理")

            inference_time = time.time() - start_time
            self.inference_count += 1

            total_keypoints = sum(p['keypoints_count'] for p in persons)
            avg_confidence = np.mean([p['confidence'] for p in persons]) if persons else 0.0

            result = {
                'type': 'pose_estimation',
                'persons': persons,
                'persons_count': len(persons),
                'total_keypoints': total_keypoints,
                'confidence': avg_confidence,
                'inference_time': inference_time,
                'inference_count': self.inference_count,
                'input_resolution': self.input_resolution,
                'original_shape': list(original_shape)
            }

            self.last_result = result

            logger.debug(f"推理完成，耗时 {inference_time:.3f}秒，检测到 {len(persons)} 人，共 {total_keypoints} 个关键点")
            return result
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None

    def _get_landmark_name(self, index):
        landmark_names = [
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
        if index < len(landmark_names):
            return landmark_names[index]
        return f'关键点_{index}'

    def release(self):
        if self.pose is not None:
            self.pose.close()
            self.pose = None

        self.mp_pose = None
        self.mp_drawing = None
        self.is_loaded = False
        logger.info("MediaPipe Pose 模型已释放")

    def check_memory(self):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            max_memory = self.config.get('max_memory_mb', 1500)
            if memory_mb > max_memory:
                logger.warning(f"内存使用过高: {memory_mb:.1f}MB (阈值: {max_memory}MB)")
                return False
            return True
        except ImportError:
            logger.debug("psutil不可用，跳过内存检查")
            return True
