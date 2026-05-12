# OneNET Token 生成工具
# 功能：生成 OneNET 平台 MQTT 连接所需的 Token 鉴权信息
# 使用方式：
#   python onenet_token.py <product_id> <device_name> <device_secret> <expire_time>
# 示例：
#   python onenet_token.py 123456 camera_001 my_secret_key 1893456000
import hashlib
import hmac
import base64
import time
import sys


def generate_onenet_token(product_id, device_name, device_secret, expire_time=None):
    if expire_time is None:
        expire_time = int(time.time()) + 365 * 24 * 3600

    version = '2018-10-31'
    res = f'products/{product_id}'

    sign_str = f'version={version}&res={res}&et={expire_time}&method=md5&key={device_secret}'

    sign = hmac.new(
        device_secret.encode('utf-8'),
        sign_str.encode('utf-8'),
        hashlib.md5
    ).hexdigest()

    token = base64.b64encode(
        f'version={version}&res={res}&et={expire_time}&method=md5&sign={sign}'.encode('utf-8')
    ).decode('utf-8')

    return token


def main():
    if len(sys.argv) < 4:
        print("使用方法: python onenet_token.py <product_id> <device_name> <device_secret> [expire_time]")
        print("\n参数说明:")
        print("  product_id   - OneNET 产品 ID（数字）")
        print("  device_name  - 设备名称")
        print("  device_secret - 设备密钥（在 OneNET 平台查看）")
        print("  expire_time  - 过期时间（Unix 时间戳，可选，默认 1 年后）")
        print("\n示例:")
        print("  python onenet_token.py 123456 camera_001 my_secret_key")
        sys.exit(1)

    product_id = sys.argv[1]
    device_name = sys.argv[2]
    device_secret = sys.argv[3]
    expire_time = int(sys.argv[4]) if len(sys.argv) > 4 else None

    token = generate_onenet_token(product_id, device_name, device_secret, expire_time)

    print("=" * 60)
    print("OneNET MQTT 连接参数")
    print("=" * 60)
    print(f"Broker Address: mqtt.heclouds.com")
    print(f"Broker Port:    1883")
    print(f"Client ID:      {device_name}")
    print(f"User Name:      {product_id}")
    print(f"Password:       {token}")
    print("=" * 60)
    print("\n将以上参数填入 config.json 的 network 配置中")


if __name__ == '__main__':
    main()
