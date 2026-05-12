# 网络通信模块 (network_client.py)
# 功能：与物联网平台通信的"电话线"，负责连接平台、订阅数据、发布结果
# 职责：
#   1. 支持 MQTT 和 HTTP 两种通信模式
#   2. 连接复用：初始化时建立连接，不在循环里反复连接断开
#   3. 心跳与重连：实现心跳包和断线自动重连机制
#   4. 异步处理：收到数据后放入队列，不阻塞接收线程
#   5. 发布识别结果到指定主题
import paho.mqtt.client as mqtt
import requests
import json
import time
import threading
import logging
from queue import Queue

logger = logging.getLogger(__name__)


class NetworkClient:
    def __init__(self, config):
        self.config = config
        self.mode = config.get('network_mode', 'mqtt')
        self.data_queue = Queue()
        self.is_connected = False
        self.client = None
        self.running = False
        self.heartbeat_interval = config.get('heartbeat_interval', 30)
        self.reconnect_interval = config.get('reconnect_interval', 5)

    def connect(self):
        if self.mode == 'mqtt':
            return self._connect_mqtt()
        elif self.mode == 'http':
            return self._connect_http()
        else:
            logger.error(f"不支持的网络模式: {self.mode}")
            return False

    def _connect_mqtt(self):
        try:
            broker = self.config.get('mqtt_broker', 'localhost')
            port = self.config.get('mqtt_port', 1883)
            topic = self.config.get('mqtt_topic', 'smart_hospital/data')

            self.client = mqtt.Client()
            self.client.on_connect = self._on_mqtt_connect
            self.client.on_message = self._on_mqtt_message
            self.client.on_disconnect = self._on_mqtt_disconnect

            self.client.connect(broker, port, 60)
            self.client.subscribe(topic)
            self.client.loop_start()

            self.is_connected = True
            logger.info(f"MQTT已连接到 {broker}:{port}，订阅主题 {topic}")
            return True
        except Exception as e:
            logger.error(f"MQTT连接失败: {e}")
            return False

    def _connect_http(self):
        try:
            self.is_connected = True
            logger.info("HTTP模式已初始化")
            return True
        except Exception as e:
            logger.error(f"HTTP初始化失败: {e}")
            return False

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("MQTT连接成功")
            self.is_connected = True
        else:
            logger.error(f"MQTT连接失败，错误码 {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.data_queue.put(payload)
            logger.debug(f"从主题 {msg.topic} 接收数据")
        except Exception as e:
            logger.error(f"处理MQTT消息时出错: {e}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        self.is_connected = False
        logger.warning(f"MQTT断开连接，错误码 {rc}")

    def publish(self, topic, data):
        if self.mode == 'mqtt' and self.client:
            try:
                payload = json.dumps(data)
                self.client.publish(topic, payload)
                logger.debug(f"数据已发布到 {topic}")
                return True
            except Exception as e:
                logger.error(f"MQTT发布失败: {e}")
                return False
        elif self.mode == 'http':
            return self._publish_http(data)
        return False

    def _publish_http(self, data):
        try:
            url = self.config.get('http_url', 'http://localhost:8080/api/data')
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"HTTP发布失败: {e}")
            return False

    def start_heartbeat(self):
        self.running = True
        thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        thread.start()

    def _heartbeat_loop(self):
        while self.running:
            time.sleep(self.heartbeat_interval)
            if not self.is_connected:
                logger.info("尝试重新连接...")
                self.connect()

    def stop(self):
        self.running = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.is_connected = False
        logger.info("网络客户端已停止")

    def get_data(self, block=True, timeout=1):
        try:
            return self.data_queue.get(block=block, timeout=timeout)
        except:
            return None
