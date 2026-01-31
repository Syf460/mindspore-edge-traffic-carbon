# ==============================================================
# 🚗 Mory · 车流过线计数稳定版（最终答辩版本）
# YOLOv8 + ByteTrack + 过线计数 + 视频输出 + JSON 统计
# ==============================================================

import cv2
import time
import json
from ultralytics import YOLO

# ---------------- 基础配置 ----------------
VIDEO_PATH = "/home/HwHiAiUser/Downloads/test.mp4"
MODEL_PATH = "/home/HwHiAiUser/Downloads/yolov8n.pt"

OUTPUT_VIDEO = "/home/HwHiAiUser/Downloads/result_crossline.mp4"
OUTPUT_JSON  = "/home/HwHiAiUser/Downloads/result_stats.json"

CONF_THRESHOLD = 0.25          # 置信度阈值
COUNT_LINE_RATIO = 0.85        # 过线位置（画面高度比例）

# YOLO 类别：2=car, 7=truck
TARGET_CLASSES = [2, 7]

# ---------------- 初始化 ----------------
print("\n🚗 初始化 YOLOv8 模型（CPU）...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开视频文件")

fps    = cap.get(cv2.CAP_PROP_FPS) or 25
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

line_y = int(height * COUNT_LINE_RATIO)

# 统计变量
tracked_ids = set()
car_count = 0
truck_count = 0
frame_id = 0
start_time = time.time()

print("🚦 开始检测与过线计数...")

# ---------------- 主循环 ----------------
for result in model.track(
    source=VIDEO_PATH,
    device="cpu",
    conf=CONF_THRESHOLD,
    classes=TARGET_CLASSES,
    tracker="bytetrack.yaml",
    stream=True,
    verbose=False
):
    frame_id += 1
    frame = result.orig_img.copy()

    # 画计数线
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        ids  = boxes.id.int().cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()

        for tid, cls_id, box in zip(ids, clss, xyxy):
            x1, y1, x2, y2 = map(int, box)
            cy = int((y1 + y2) / 2)

            if cls_id == 2:
                label = "car"
                color = (0, 255, 0)
            elif cls_id == 7:
                label = "truck"
                color = (255, 255, 0)
            else:
                continue

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}#{tid}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # -------- 过线计数逻辑 --------
            if cy > line_y and tid not in tracked_ids:
                tracked_ids.add(tid)
                if cls_id == 2:
                    car_count += 1
                elif cls_id == 7:
                    truck_count += 1

    # 显示统计信息
    total = car_count + truck_count
    fps_now = frame_id / max(1e-6, (time.time() - start_time))

    cv2.putText(
        frame,
        f"Car: {car_count} | Truck: {truck_count} | Total: {total}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"FPS: {fps_now:.1f}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    writer.write(frame)
    cv2.imshow("Mory Traffic Crossline", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- 收尾 ----------------
cap.release()
writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time

print("\n✅ 检测完成")
print(f"⏱️ 用时：{elapsed:.1f} 秒")
print(f"🚗 汽车：{car_count} 辆")
print(f"🚚 货车：{truck_count} 辆")

# 保存 JSON
stats = {
    "car": car_count,
    "truck": truck_count,
    "total": car_count + truck_count,
    "video": VIDEO_PATH,
    "output_video": OUTPUT_VIDEO,
    "elapsed_seconds": round(elapsed, 2)
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"🎞️ 已保存视频：{OUTPUT_VIDEO}")
print(f"📄 已保存统计结果：{OUTPUT_JSON}")
print("\n🎯 系统运行完成（稳定版）")
