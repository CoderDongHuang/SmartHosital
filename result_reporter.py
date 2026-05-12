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
