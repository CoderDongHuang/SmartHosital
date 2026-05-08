# 测试脚本 (test_mediapipe.py)
# 功能：使用单张图片测试 MediaPipe Pose 推理引擎，并可视化显示检测结果
# 用法：python test_mediapipe.py [图片路径]
# 如果不提供图片路径，会自动生成一张测试图片
import cv2
import numpy as np
import os
import sys
import time
from PIL import Image, ImageDraw, ImageFont

# 导入 AI 推理引擎
from ai_engine import AIEngine


def get_chinese_font(font_size=20):
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    return ImageFont.load_default()


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (27, 31), (28, 32), (29, 31), (30, 32)
]


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


def draw_pose_results(image, result):
    display_img = image.copy()

    keypoints = result.get('keypoints', [])
    if not keypoints:
        cv2.putText(display_img, "未检测到人体", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return display_img

    kp_dict = {kp['index']: kp for kp in keypoints if kp['confidence'] > 0.3}

    for i, j in POSE_CONNECTIONS:
        if i in kp_dict and j in kp_dict:
            pt1 = (int(kp_dict[i]['x']), int(kp_dict[i]['y']))
            pt2 = (int(kp_dict[j]['x']), int(kp_dict[j]['y']))
            cv2.line(display_img, pt1, pt2, (0, 255, 0), 2)

    for kp in keypoints:
        if kp['confidence'] > 0.3:
            x, y = int(kp['x']), int(kp['y'])
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

    pil_img = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_small = get_chinese_font(14)
    font_info = get_chinese_font(18)

    for kp in keypoints:
        if kp['confidence'] > 0.3:
            x, y = int(kp['x']), int(kp['y'])
            draw.text((x + 5, y - 5), kp['name'], fill=(255, 255, 0), font=font_small)

    info_text = [
        f"推理耗时: {result['inference_time']:.3f}s",
        f"关键点: {result['keypoints_count']}",
        f"置信度: {result['confidence']:.2%}"
    ]
    for idx, text in enumerate(info_text):
        draw.text((10, 30 + idx * 30), text, fill=(0, 255, 255), font=font_info)

    display_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return display_img


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

    important_joints = ['左肩', '右肩', '左髋', '右髋', '左膝', '右膝']
    print("\n关键关节点坐标:")
    for kp in result['keypoints']:
        if kp['name'] in important_joints and kp['confidence'] > 0.3:
            print(f"  {kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f})")

    print("\n[5/5] 生成可视化结果...")
    result_img = draw_pose_results(image, result)

    results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(results_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(results_dir, f'{base_name}_result.jpg')
    cv2.imwrite(output_path, result_img)
    print(f"结果图片已保存: {output_path}")

    cv2.imshow("MediaPipe Pose 检测结果", result_img)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
