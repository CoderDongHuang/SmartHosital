# 语音提示模块 (voice_tts.py)
# 功能：根据动作评分生成语音提示，通过串口控制语音模块
# 职责：
#   1. 根据动作类型和评分生成语音指令
#   2. 支持 SYN6288 文本转语音模块（串口控制）
#   3. 支持 ISD1820 录放音模块（GPIO 触发）
#   4. 防重复播放：同一提示至少间隔 3 秒
# 硬件适配：树莓派 5 GPIO/串口，外接语音模块
import logging
import time
import serial
import os

logger = logging.getLogger(__name__)


class VoiceTTS:
    def __init__(self, config):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.device_type = config.get('device_type', 'syn6288')
        self.last_play_time = 0
        self.min_interval = config.get('min_interval', 3.0)
        self.serial_port = None

        if self.enabled:
            self._init_device()

    def _init_device(self):
        if self.device_type == 'syn6288':
            self._init_syn6288()
        elif self.device_type == 'isd1820':
            self._init_isd1820()
        else:
            logger.warning(f"不支持的语音设备类型: {self.device_type}")
            self.enabled = False

    def _init_syn6288(self):
        try:
            port = self.config.get('serial_port', '/dev/ttyUSB0')
            baudrate = self.config.get('baudrate', 9600)
            self.serial_port = serial.Serial(port, baudrate, timeout=1)
            logger.info(f"SYN6288 语音模块已初始化: {port}")
        except Exception as e:
            logger.error(f"SYN6288 初始化失败: {e}")
            self.enabled = False

    def _init_isd1820(self):
        try:
            import RPi.GPIO as GPIO
            self.gpio_pin = self.config.get('gpio_pin', 17)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            GPIO.output(self.gpio_pin, GPIO.LOW)
            logger.info(f"ISD1820 语音模块已初始化: GPIO{self.gpio_pin}")
        except Exception as e:
            logger.error(f"ISD1820 初始化失败: {e}")
            self.enabled = False

    def speak(self, processed_result):
        if not self.enabled:
            return

        result_data = processed_result.get('result', {})
        feedback = result_data.get('feedback', '')
        action_type = result_data.get('action_type', '')
        quality_score = result_data.get('quality_score', 0)

        if not feedback:
            return

        now = time.time()
        if now - self.last_play_time < self.min_interval:
            return

        self._play_voice(feedback)
        self.last_play_time = now

    def _play_voice(self, text):
        if self.device_type == 'syn6288':
            self._syn6288_speak(text)
        elif self.device_type == 'isd1820':
            self._isd1820_play()

    def _syn6288_speak(self, text):
        if not self.serial_port:
            return

        try:
            header = bytes([0xFD, 0x00])
            length = len(text) + 3
            length_bytes = bytes([length >> 8, length & 0xFF])
            control = bytes([0x01, 0x01])
            data = text.encode('gb2312')
            checksum = 0
            for b in header[1:] + length_bytes + control + data:
                checksum ^= b
            checksum_bytes = bytes([checksum])

            frame = header + length_bytes + control + data + checksum_bytes
            self.serial_port.write(frame)
            logger.debug(f"语音播放: {text}")
        except Exception as e:
            logger.error(f"SYN6288 播放失败: {e}")

    def _isd1820_play(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.output(self.gpio_pin, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(self.gpio_pin, GPIO.LOW)
            logger.debug(f"ISD1820 触发播放")
        except Exception as e:
            logger.error(f"ISD1820 播放失败: {e}")

    def cleanup(self):
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None

        if self.device_type == 'isd1820':
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup(self.gpio_pin)
            except:
                pass

        logger.info("语音模块已停止")
