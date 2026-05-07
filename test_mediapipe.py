# -*- coding: utf-8 -*-
# 测试脚本 (test_mediapipe.py)
# 功能：使用单张图片测试 MediaPipe Pose 推理引擎
# 用法：python test_mediapipe.py [图片路径]
# 如果不提供图片路径，会自动生成一张测试图片
import cv2
import numpy as np
import os
import sys
import time

# 导入 AI 推理引擎
from ai_engine import AIEngine


def create_test_image():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (200, 100), (440, 400), (0, 255, 0), 2)
    cv2.circle(img, (320, 150), 30, (0, 255, 255), -1)
    cv2.line(img, (320, 180), (320, 280), (255, 0, 0), 3)
    cv2.line(img, (320, 280), (260, 380), (0, 0, 255), 3)
    cv2.line(img, (320, 280), (380, 380), (0, 0, 255), 3)
    cv2.line(img, (320, 200), (240, 260), (255, 255, 0), 3)
    cv2.line(img, (320, 200), (400, 260), (255, 255, 0), 3)
    cv2.putText(img, "TEST IMAGE", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


def test_single_image(image_path):
    print("=" * 60)
    print("MediaPipe Pose 推理测试")
    print("=" * 60)

    config = {
        'model_type': 'mediapipe_pose',
        'confidence_threshold': 0.5,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'input_resolution': (256, 256),
        'max_memory_mb': 1500
    }

    engine = AIEngine(config)

    print("\n[1/4] 加载 MediaPipe Pose 模型...")
    if not engine.load_model():
        print("模型加载失败！")
        return False
    print("模型加载成功！")

    print(f"\n[2/4] 读取测试图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"图片读取失败: {image_path}")
        return False
    print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")

    print("\n[3/4] 执行推理...")
    start_time = time.time()
    result = engine.predict(image)
    inference_time = time.time() - start_time

    if result is None:
        print("推理失败！")
        return False

    print(f"推理耗时: {inference_time:.3f}秒")
    print(f"检测到关键点数量: {result['keypoints_count']}")
    print(f"整体置信度: {result['confidence']:.2%}")

    print("\n[4/4] 关键点详情:")
    print("-" * 60)
    print(f"{'索引':<6} {'名称':<20} {'X':<10} {'Y':<10} {'置信度':<10}")
    print("-" * 60)

    for kp in result['keypoints']:
        if kp['confidence'] > 0.3:
            print(f"{kp['index']:<6} {kp['name']:<20} {kp['x']:<10.1f} {kp['y']:<10.1f} {kp['confidence']:<10.2%}")

    print("-" * 60)

    important_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee']
    print("\n关键关节点坐标:")
    for kp in result['keypoints']:
        if kp['name'] in important_joints and kp['confidence'] > 0.3:
            print(f"  {kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f})")

    engine.release()
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    return True


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("未指定图片路径，自动生成测试图片...")
        test_img = create_test_image()
        image_path = os.path.join(os.path.dirname(__file__), 'test_image.jpg')
        cv2.imwrite(image_path, test_img)
        print(f"测试图片已保存: {image_path}")

    if os.path.exists(image_path):
        test_single_image(image_path)
    else:
        print(f"图片不存在: {image_path}")
