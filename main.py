# 主控调度模块 (main.py)
# 功能：系统的总开关和看门狗，负责初始化所有模块、启动主循环、监控程序状态、捕获全局异常
# 职责：
#   1. 加载配置文件并初始化所有子模块
#   2. 维护无限主循环，协调在线模式和离线模拟模式的切换
#   3. 监控错误次数，超过阈值时自动尝试恢复（网络重连、模型重载）
#   4. 处理信号中断（Ctrl+C），优雅关闭系统
#   5. 统计并输出最终运行数据
# 数据流：接收数据 -> 解析校验 -> AI推理 -> 后处理评分 -> 结果上报
import os
import sys
import time
import json
import logging
import signal
from datetime import datetime

from network_client import NetworkClient
from data_parser import DataParser
from ai_engine import AIEngine
from post_processor import PostProcessor
from result_reporter import ResultReporter
from data_simulator import DataSimulator


class SmartHospitalSystem:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        self.is_running = False
        self.modules = {}
        self.stats = {
            'start_time': None,
            'data_received': 0,
            'data_processed': 0,
            'results_reported': 0,
            'errors': 0
        }

        self._setup_logging()
        self._setup_signal_handlers()

    def _load_config(self, config_path):
        default_config = {
            'system': {
                'mode': 'online',
                'loop_interval': 0.1,
                'max_errors': 10,
                'error_reset_time': 60
            },
            'network': {
                'network_mode': 'mqtt',
                'mqtt_broker': 'localhost',
                'mqtt_port': 1883,
                'mqtt_topic': 'smart_hospital/data',
                'result_topic': 'smart_hospital/result',
                'heartbeat_interval': 30,
                'reconnect_interval': 5
            },
            'ai': {
                'model_type': 'pose_estimation',
                'model_path': None,
                'confidence_threshold': 0.5,
                'max_memory_mb': 4096
            },
            'post_processing': {
                'confidence_threshold': 0.5,
                'angle_thresholds': {
                    'squat': {'knee': (90, 140), 'hip': (80, 120)},
                    'pushup': {'elbow': (60, 120), 'shoulder': (80, 160)},
                    'jump': {'knee': (150, 180), 'ankle': (80, 110)}
                }
            },
            'simulator': {
                'sim_data_dir': './sim_data',
                'sim_interval': 2,
                'sim_loop': True
            }
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                self._merge_config(default_config, user_config)
                logging.info(f"配置文件已从 {config_path} 加载")
            except Exception as e:
                logging.warning(f"配置文件加载失败: {e}，使用默认配置")
        else:
            logging.info(f"配置文件不存在: {config_path}，使用默认配置")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logging.info(f"默认配置已创建: {config_path}")

        return default_config

    def _merge_config(self, default, user):
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value

    def _setup_logging(self):
        log_level = self.config.get('system', {}).get('log_level', 'INFO')
        log_file = self.config.get('system', {}).get('log_file', 'smart_hospital.log')

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger('SmartHospital')

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.logger.info(f"收到信号 {signum}，正在关闭系统...")
        self.is_running = False

    def initialize(self):
        self.logger.info("=" * 50)
        self.logger.info("智慧医院AI系统初始化中...")
        self.logger.info("=" * 50)

        try:
            self.logger.info("初始化网络客户端...")
            self.modules['network'] = NetworkClient(self.config['network'])
            if not self.modules['network'].connect():
                self.logger.warning("网络连接失败，将重试")

            self.logger.info("初始化数据解析器...")
            self.modules['parser'] = DataParser(self.config.get('parser', {}))

            self.logger.info("初始化AI引擎...")
            self.modules['ai'] = AIEngine(self.config['ai'])
            if not self.modules['ai'].load_model():
                self.logger.warning("AI模型加载失败，将使用模拟模式")

            self.logger.info("初始化后处理器...")
            self.modules['post_processor'] = PostProcessor(self.config['post_processing'])

            self.logger.info("初始化结果上报器...")
            self.modules['reporter'] = ResultReporter(
                self.config['network'],
                self.modules['network']
            )

            self.logger.info("初始化数据模拟器...")
            self.modules['simulator'] = DataSimulator(self.config['simulator'])
            if not self.modules['simulator'].initialize():
                self.logger.warning("模拟器初始化失败")

            self.modules['network'].start_heartbeat()

            self.stats['start_time'] = datetime.now()
            self.is_running = True

            self.logger.info("=" * 50)
            self.logger.info("所有模块初始化成功！")
            self.logger.info(f"系统模式: {self.config['system']['mode']}")
            self.logger.info("=" * 50)

            return True
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            return False

    def run(self):
        if not self.is_running:
            self.logger.error("系统未初始化，无法运行")
            return

        self.logger.info("系统已启动，进入主循环...")
        error_count = 0
        max_errors = self.config['system'].get('max_errors', 10)
        loop_interval = self.config['system'].get('loop_interval', 0.1)

        try:
            while self.is_running:
                try:
                    self._process_cycle()
                    error_count = 0
                except Exception as e:
                    error_count += 1
                    self.stats['errors'] += 1
                    self.logger.error(f"循环错误 ({error_count}/{max_errors}): {e}")

                    if error_count >= max_errors:
                        self.logger.error("错误次数过多，尝试恢复...")
                        self._attempt_recovery()
                        error_count = 0

                time.sleep(loop_interval)
        except KeyboardInterrupt:
            self.logger.info("收到键盘中断信号")
        except Exception as e:
            self.logger.error(f"致命错误: {e}")
        finally:
            self.shutdown()

    def _process_cycle(self):
        mode = self.config['system']['mode']

        if mode == 'online':
            self._process_online_mode()
        elif mode == 'offline':
            self._process_offline_mode()
        else:
            self._process_online_mode()
            if self.stats['data_received'] == 0:
                self.logger.info("未收到数据，切换到离线模式")
                self.config['system']['mode'] = 'offline'

    def _process_online_mode(self):
        raw_data = self.modules['network'].get_data(block=False)
        if raw_data is None:
            return

        self.stats['data_received'] += 1
        self.logger.info(f"收到数据: {self.stats['data_received']}")

        parsed_data = self.modules['parser'].parse_and_validate(raw_data)
        if parsed_data is None:
            self.logger.warning("数据验证失败，跳过")
            return

        if not self.modules['ai'].check_memory():
            self.logger.warning("内存使用过高，释放缓存")
            self.modules['ai'].release()
            self.modules['ai'].load_model()

        ai_result = self.modules['ai'].predict(parsed_data['image'])
        if ai_result is None:
            self.logger.error("AI推理失败")
            return

        processed_result = self.modules['post_processor'].process(ai_result, parsed_data)
        if processed_result is None:
            self.logger.error("后处理失败")
            return

        self.stats['data_processed'] += 1

        success = self.modules['reporter'].report(processed_result, is_mock=False)
        if success:
            self.stats['results_reported'] += 1
            result = processed_result.get('result', {})
            self.logger.info(
                f"结果: {result.get('action_type', '未知')} | "
                f"评分: {result.get('quality_score', 0):.1f} | "
                f"置信度: {result.get('confidence', 0):.2%}"
            )

    def _process_offline_mode(self):
        sim_data = self.modules['simulator'].get_next_data()
        if sim_data is None:
            self.logger.info("无可用模拟数据")
            return

        self.stats['data_received'] += 1
        self.logger.info(f"模拟数据: {self.stats['data_received']}")

        parsed_data = self.modules['parser'].parse_and_validate(sim_data)
        if parsed_data is None:
            self.logger.warning("模拟数据验证失败")
            return

        ai_result = self.modules['ai'].predict(parsed_data['image'])
        if ai_result is None:
            self.logger.error("模拟数据AI推理失败")
            return

        processed_result = self.modules['post_processor'].process(ai_result, parsed_data)
        if processed_result is None:
            self.logger.error("模拟数据后处理失败")
            return

        self.stats['data_processed'] += 1

        success = self.modules['reporter'].report(processed_result, is_mock=True)
        if success:
            self.stats['results_reported'] += 1
            result = processed_result.get('result', {})
            self.logger.info(
                f"[模拟] 结果: {result.get('action_type', '未知')} | "
                f"评分: {result.get('quality_score', 0):.1f} | "
                f"置信度: {result.get('confidence', 0):.2%}"
            )

        time.sleep(self.config['simulator'].get('sim_interval', 2))

    def _attempt_recovery(self):
        self.logger.info("尝试系统恢复...")

        try:
            self.modules['network'].stop()
            time.sleep(2)

            if self.modules['network'].connect():
                self.logger.info("网络已重新连接")
            else:
                self.logger.error("网络重新连接失败")

            self.modules['ai'].release()
            if self.modules['ai'].load_model():
                self.logger.info("AI模型已重新加载")
            else:
                self.logger.error("AI模型重新加载失败")

            self.logger.info("恢复完成")
        except Exception as e:
            self.logger.error(f"恢复失败: {e}")

    def shutdown(self):
        self.logger.info("正在关闭系统...")
        self.is_running = False

        try:
            if 'network' in self.modules:
                self.modules['network'].stop()

            if 'ai' in self.modules:
                self.modules['ai'].release()

            self._print_final_stats()
        except Exception as e:
            self.logger.error(f"关闭时出错: {e}")

        self.logger.info("系统关闭完成")

    def _print_final_stats(self):
        self.logger.info("=" * 50)
        self.logger.info("最终统计:")
        self.logger.info(f"  运行时间: {datetime.now() - self.stats['start_time']}")
        self.logger.info(f"  接收数据: {self.stats['data_received']}")
        self.logger.info(f"  处理数据: {self.stats['data_processed']}")
        self.logger.info(f"  上报结果: {self.stats['results_reported']}")
        self.logger.info(f"  错误次数: {self.stats['errors']}")

        if 'reporter' in self.modules:
            reporter_stats = self.modules['reporter'].get_stats()
            self.logger.info(f"  上报成功率: {reporter_stats['success_rate']:.2%}")

        self.logger.info("=" * 50)


def main():
    config_path = 'config.json'

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    system = SmartHospitalSystem(config_path)

    if system.initialize():
        system.run()
    else:
        logging.error("系统初始化失败，退出")
        sys.exit(1)


if __name__ == '__main__':
    main()
