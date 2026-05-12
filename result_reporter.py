# 结果上报模块 (result_reporter.py)
# 功能：系统的"广播员"，将后处理的结果封装成约定格式并发送回物联网平台
# 职责：
#   1. 结果中必须包含能关联原始数据的唯一ID（如图片ID、时间戳）
#   2. 加入推理耗时、置信度等调试信息，方便后期优化
#   3. 统计上报成功率等运行指标
#   4. 区分真实数据和模拟数据的上报标记
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultReporter:
    def __init__(self, config, network_client):
        self.config = config
        self.network_client = network_client
        self.result_topic = config.get('result_topic', 'smart_hospital/result')
        self.report_count = 0
        self.success_count = 0
        self.fail_count = 0

    def report(self, processed_result, is_mock=False):
        if processed_result is None:
            logger.error("无法上报空结果")
            return False

        try:
            report_data = self._format_result(processed_result, is_mock)

            success = self.network_client.publish(self.result_topic, report_data)

            self.report_count += 1
            if success:
                self.success_count += 1
                logger.info(f"结果上报成功: {processed_result.get('data_id')}")
            else:
                self.fail_count += 1
                logger.error(f"结果上报失败: {processed_result.get('data_id')}")

            return success
        except Exception as e:
            self.fail_count += 1
            logger.error(f"上报异常: {e}")
            return False

    def _format_result(self, processed_result, is_mock=False):
        result_data = processed_result.get('result', {})

        platform = self.config.get('platform', 'custom')

        if platform == 'onenet':
            return self._format_onenet_result(processed_result, result_data, is_mock)
        else:
            return self._format_custom_result(processed_result, result_data, is_mock)

    def _format_onenet_result(self, processed_result, result_data, is_mock=False):
        import time
        timestamp_ms = int(time.time() * 1000)

        formatted = {
            'id': str(processed_result.get('data_id', ''))[:13],
            'version': '1.0',
            'params': {
                'action_type': {
                    'value': result_data.get('action_type', '未知'),
                    'time': timestamp_ms
                },
                'quality_score': {
                    'value': result_data.get('quality_score', 0),
                    'time': timestamp_ms
                },
                'confidence': {
                    'value': result_data.get('confidence', 0),
                    'time': timestamp_ms
                },
                'feedback': {
                    'value': result_data.get('feedback', ''),
                    'time': timestamp_ms
                },
                'status': {
                    'value': result_data.get('status', 'unknown'),
                    'time': timestamp_ms
                }
            }
        }

        if 'joint_angles' in result_data:
            angles = result_data['joint_angles']
            for angle_name, angle_value in angles.items():
                formatted['params'][f'angle_{angle_name}'] = {
                    'value': angle_value,
                    'time': timestamp_ms
                }

        if 'keypoints_count' in result_data:
            formatted['params']['keypoints_count'] = {
                'value': result_data['keypoints_count'],
                'time': timestamp_ms
            }

        return formatted

    def _format_custom_result(self, processed_result, result_data, is_mock=False):
        formatted = {
            'data_id': processed_result.get('data_id'),
            'device_id': processed_result.get('device_id'),
            'original_timestamp': processed_result.get('timestamp'),
            'processed_at': processed_result.get('processed_at'),
            'status': result_data.get('status'),
            'action_type': result_data.get('action_type'),
            'quality_score': result_data.get('quality_score'),
            'confidence': result_data.get('confidence'),
            'feedback': result_data.get('feedback'),
            'is_mock': is_mock,
            'metadata': {
                'report_count': self.report_count,
                'success_rate': self.success_count / max(1, self.report_count),
                'inference_time': processed_result.get('inference_time', 0)
            }
        }

        if 'joint_angles' in result_data:
            formatted['joint_angles'] = result_data['joint_angles']

        if 'keypoints_count' in result_data:
            formatted['keypoints_count'] = result_data['keypoints_count']

        return formatted

    def get_stats(self):
        return {
            'total_reports': self.report_count,
            'successful': self.success_count,
            'failed': self.fail_count,
            'success_rate': self.success_count / max(1, self.report_count)
        }

    def reset_stats(self):
        self.report_count = 0
        self.success_count = 0
        self.fail_count = 0
