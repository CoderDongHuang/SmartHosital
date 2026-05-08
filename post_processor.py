# 结果后处理模块 (post_processor.py)
# 功能：系统的"裁判"，把AI引擎的原始输出进行加工，使其变得有用
# 职责：
#   1. 计算关节点角度，判断动作是否标准（如膝、髋、踝角度）
#   2. 根据置信度过滤无效识别结果
#   3. 生成可读的评分或建议反馈
#   4. 核心算法亮点：从"识别出一个人"到"评价他的动作是否标准"
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PostProcessor:
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.angle_thresholds = config.get('angle_thresholds', {
            'squat': {'knee': (90, 140), 'hip': (80, 120)},
            'pushup': {'elbow': (60, 120), 'shoulder': (80, 160)},
            'jump': {'knee': (150, 180), 'ankle': (80, 110)}
        })

    def process(self, ai_result, parsed_data):
        if ai_result is None or parsed_data is None:
            logger.error("后处理输入无效")
            return None

        try:
            if ai_result.get('type') == 'pose_estimation':
                return self._process_pose(ai_result, parsed_data)
            elif ai_result.get('type') == 'action_recognition':
                return self._process_action(ai_result, parsed_data)
            else:
                return self._process_generic(ai_result, parsed_data)
        except Exception as e:
            logger.error(f"后处理失败: {e}")
            return None

    def _process_pose(self, ai_result, parsed_data):
        persons = ai_result.get('persons', [])
        confidence = ai_result.get('confidence', 0)

        if not persons:
            logger.warning("未检测到人体")
            return self._create_result(parsed_data, {
                'status': 'no_person',
                'confidence': 0,
                'message': '未检测到人体'
            })

        if confidence < self.confidence_threshold:
            logger.warning(f"置信度过低: {confidence}")
            return self._create_result(parsed_data, {
                'status': 'low_confidence',
                'confidence': confidence,
                'message': '识别置信度不足'
            })

        person_results = []
        for person in persons:
            person_id = person.get('person_id', 0)
            keypoints = person.get('keypoints', [])
            person_confidence = person.get('confidence', 0)

            angles = self._calculate_joint_angles(keypoints)
            action_type = self._classify_action(angles)
            quality_score = self._evaluate_action_quality(angles, action_type)

            person_results.append({
                'person_id': person_id,
                'action_type': action_type,
                'quality_score': quality_score,
                'joint_angles': angles,
                'keypoints_count': len(keypoints),
                'confidence': person_confidence,
                'feedback': self._generate_feedback(action_type, quality_score, angles)
            })

        result_data = {
            'status': 'success',
            'persons_count': len(persons),
            'persons': person_results,
            'total_keypoints': ai_result.get('total_keypoints', 0),
            'confidence': confidence,
            'feedback': self._generate_multi_person_feedback(person_results)
        }

        return self._create_result(parsed_data, result_data)

    def _process_action(self, ai_result, parsed_data):
        action_class = ai_result.get('action_class', -1)
        confidence = ai_result.get('confidence', 0)
        action_name = ai_result.get('action_name', '未知')

        if confidence < self.confidence_threshold:
            return self._create_result(parsed_data, {
                'status': 'low_confidence',
                'confidence': confidence,
                'message': '识别置信度不足'
            })

        action_names = ['深蹲', '俯卧撑', '跳跃', '站立', '行走']
        action_type = action_name if action_name else (action_names[action_class] if action_class < len(action_names) else '未知')

        quality_score = confidence * 100

        result_data = {
            'status': 'success',
            'action_type': action_type,
            'quality_score': quality_score,
            'confidence': confidence,
            'feedback': f"检测到动作: {action_type}, 置信度: {confidence:.2%}"
        }

        return self._create_result(parsed_data, result_data)

    def _process_generic(self, ai_result, parsed_data):
        result_data = {
            'status': 'success',
            'action_type': '未知',
            'quality_score': ai_result.get('confidence', 0) * 100,
            'confidence': ai_result.get('confidence', 0),
            'feedback': '通用识别结果'
        }

        return self._create_result(parsed_data, result_data)

    def _calculate_joint_angles(self, keypoints):
        if len(keypoints) < 17:
            return {}

        def get_point(idx):
            if idx < len(keypoints):
                return np.array([keypoints[idx]['x'], keypoints[idx]['y']])
            return None

        angles = {}

        hip_l = get_point(11)
        hip_r = get_point(12)
        knee_l = get_point(13)
        knee_r = get_point(14)
        ankle_l = get_point(15)
        ankle_r = get_point(16)
        shoulder_l = get_point(5)
        shoulder_r = get_point(6)
        elbow_l = get_point(7)
        elbow_r = get_point(8)

        if knee_l is not None and hip_l is not None and ankle_l is not None:
            angles['knee_left'] = self._calculate_angle(hip_l, knee_l, ankle_l)
        if knee_r is not None and hip_r is not None and ankle_r is not None:
            angles['knee_right'] = self._calculate_angle(hip_r, knee_r, ankle_r)

        if hip_l is not None and shoulder_l is not None and knee_l is not None:
            angles['hip_left'] = self._calculate_angle(shoulder_l, hip_l, knee_l)
        if hip_r is not None and shoulder_r is not None and knee_r is not None:
            angles['hip_right'] = self._calculate_angle(shoulder_r, hip_r, knee_r)

        if elbow_l is not None and shoulder_l is not None and hip_l is not None:
            angles['elbow_left'] = self._calculate_angle(shoulder_l, elbow_l, hip_l)
        if elbow_r is not None and shoulder_r is not None and hip_r is not None:
            angles['elbow_right'] = self._calculate_angle(shoulder_r, elbow_r, hip_r)

        if ankle_l is not None and knee_l is not None and hip_l is not None:
            angles['ankle_left'] = self._calculate_angle(knee_l, ankle_l, hip_l)
        if ankle_r is not None and knee_r is not None and hip_r is not None:
            angles['ankle_right'] = self._calculate_angle(knee_r, ankle_r, hip_r)

        return angles

    def _calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return float(angle)

    def _classify_action(self, angles):
        if not angles:
            return '未知'

        knee_angles = [angles.get('knee_left', 180), angles.get('knee_right', 180)]
        avg_knee = np.mean(knee_angles)

        hip_angles = [angles.get('hip_left', 180), angles.get('hip_right', 180)]
        avg_hip = np.mean(hip_angles)

        if 90 <= avg_knee <= 140 and 80 <= avg_hip <= 120:
            return '深蹲'
        elif avg_knee > 150:
            return '站立'
        elif avg_knee < 90:
            return '深蹲（深度）'
        else:
            return '其他'

    def _evaluate_action_quality(self, angles, action_type):
        if action_type not in self.angle_thresholds:
            return 50.0

        thresholds = self.angle_thresholds[action_type]
        scores = []

        for joint, (min_angle, max_angle) in thresholds.items():
            left_key = f"{joint}_left"
            right_key = f"{joint}_right"

            left_angle = angles.get(left_key)
            right_angle = angles.get(right_key)

            if left_angle is not None:
                if min_angle <= left_angle <= max_angle:
                    scores.append(100)
                else:
                    deviation = min(abs(left_angle - min_angle), abs(left_angle - max_angle))
                    score = max(0, 100 - deviation)
                    scores.append(score)

            if right_angle is not None:
                if min_angle <= right_angle <= max_angle:
                    scores.append(100)
                else:
                    deviation = min(abs(right_angle - min_angle), abs(right_angle - max_angle))
                    score = max(0, 100 - deviation)
                    scores.append(score)

        if scores:
            return float(np.mean(scores))
        return 50.0

    def _generate_feedback(self, action_type, quality_score, angles):
        if quality_score >= 90:
            return f"{action_type}动作标准，继续保持！"
        elif quality_score >= 70:
            return f"{action_type}动作基本标准，注意调整关节角度"
        elif quality_score >= 50:
            return f"{action_type}动作需要改进，请参考标准动作"
        else:
            return f"{action_type}动作不规范，建议重新练习"

    def _generate_multi_person_feedback(self, person_results):
        if not person_results:
            return "未检测到人体"
        
        if len(person_results) == 1:
            return person_results[0]['feedback']
        
        feedback_parts = []
        for person in person_results:
            person_id = person['person_id']
            feedback_parts.append(f"人员{person_id}: {person['feedback']}")
        
        return " | ".join(feedback_parts)

    def _create_result(self, parsed_data, result_data):
        return {
            'data_id': parsed_data.get('data_id'),
            'device_id': parsed_data.get('device_id'),
            'timestamp': parsed_data.get('timestamp'),
            'processed_at': datetime.now().isoformat(),
            'result': result_data
        }
