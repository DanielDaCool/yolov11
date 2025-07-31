
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math
from ultralytics import YOLO
from decimal import Decimal, ROUND_HALF_UP
import sys
import os
import matplotlib.pyplot as plt
from collections import deque

# Ensure tmp folder exists
os.makedirs("tmp", exist_ok=True)

live_coords_buffer = deque(maxlen=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.ion()
plt.show()


def init_camera(laser_power=10, use_bag_file=False, bag_file_path=None, record_to_bag=False):
    pipeline = rs.pipeline()
    config = rs.config()

    if use_bag_file and bag_file_path:
        config.enable_device_from_file(bag_file_path, repeat_playback=False)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)

        if record_to_bag:
            timestamp = int(time.time())
            bag_output_path = f"recordings/session_{timestamp}.bag"
            os.makedirs("recordings", exist_ok=True)
            config.enable_record_to_file(bag_output_path)
            print(f"[INFO] Recording to: {bag_output_path}")

    profile = pipeline.start(config)
    align = rs.align(rs.stream.infrared)

    if not use_bag_file:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()

        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 4)

        if depth_sensor.supports(rs.option.laser_power):
            range = depth_sensor.get_option_range(rs.option.laser_power)
            power = max(range.min, min(laser_power, range.max))
            depth_sensor.set_option(rs.option.laser_power, power)
            print(f"Set laser power to {power} (range: {range.min}–{range.max})")

        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

    return pipeline, profile, align



def convert_pixel_to_ros_coords(cx, cy, depth_frame, intr, pitch_deg=45, offset=(0, 3370, 3770)):
    depth = depth_frame.get_distance(cx, cy)
    if depth == 0:
        return None, None

    depth_mm = depth * 1000
    point_3d = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_mm)
    x_opt, y_opt, z_opt = point_3d

    pitch_rad = math.radians(pitch_deg)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch_rad), math.sin(pitch_rad)],
        [0, -math.sin(pitch_rad),  math.cos(pitch_rad)]
    ])
    rotated = Rx @ np.array([x_opt, y_opt, z_opt])
    x_rot, y_rot, z_rot = rotated

    x_ros = z_rot + offset[0]
    y_ros = -x_rot + offset[1]
    z_ros = -y_rot + offset[2]

    return (x_rot, y_rot, z_rot), (x_ros, y_ros, z_ros)


def update_live_3d_plot():
    if len(live_coords_buffer) < 2:
        return

    data = np.array(live_coords_buffer)

    ax1.clear()
    ax1.plot(data[:, 1], data[:, 0], marker='o', color='blue')  # Y vs X
    ax1.scatter(0, 0, color='red', s=100, label='Origin')
    ax1.set_xlim(left=max(data[:, 1]) + 100, right=0)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')
    ax1.set_title("Live Y-X Trajectory (ROS2 view: origin bottom-right)")
    ax1.legend()

    ax2.clear()
    ax2.plot(data[:, 0], data[:, 2], marker='o', color='green')  # X vs Z
    ax2.scatter(0, 0, color='red', s=100, label='Origin')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title("Live X-Z Trajectory")
    ax2.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def process_frame(model, ir_bgr, depth_frame, intr, confidence=0.5):
    results = model(ir_bgr, verbose=False)[0]
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence:
            continue

        label = model.names[int(box.cls[0])]
        if label.lower() != "basketball":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rot_coords, ros_coords = convert_pixel_to_ros_coords(cx, cy, depth_frame, intr, pitch_deg=45)

        if ros_coords and ros_coords[2] <= 10000:
            rot_str = tuple(str(Decimal(c).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) for c in rot_coords)
            ros_str = tuple(str(Decimal(c).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) for c in ros_coords)

            ros_text = f"ROS: ({ros_str[0]}, {ros_str[1]}, {ros_str[2]}) mm"
            rot_text = f"ROT: ({rot_str[0]}, {rot_str[1]}, {rot_str[2]}) mm"

            cv2.rectangle(ir_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(ir_bgr, ros_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(ir_bgr, rot_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            return ros_coords, True, ir_bgr

    return None, False, ir_bgr


def show_detections(ir_bgr, coords_3d, frame_count, fps):
    cv2.putText(ir_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if coords_3d:
        cv2.imwrite(f"tmp/{frame_count}.jpg", ir_bgr)
    cv2.imshow("IR + Basketball Detection + 3D Coords", ir_bgr)


def detection_loop(model_path="yolov8n.pt", confidence=0.5, laser_power=10,
                   use_bag_file=False, bag_file_path=None,record_to_bag=False):
    model = YOLO(model_path)
    model.verbose = False

    pipeline, profile, align = init_camera(laser_power, use_bag_file, bag_file_path,record_to_bag)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    frame_count = 0
    start_time = time.time()
    detections_list = []
    last_coords_3d = None
    last_detected_frame = 0
    no_detection_counter = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            ir_frame = aligned_frames.get_infrared_frame()
            if not depth_frame or not ir_frame:
                continue

            ir_image = np.asanyarray(ir_frame.get_data())
            ir_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

            coords_3d, detected, ir_bgr = process_frame(model, ir_bgr, depth_frame, intr, confidence)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed

            if detected:
                last_coords_3d = coords_3d
                last_detected_frame = frame_count
                no_detection_counter = 0
                detections_list.append((frame_count, *tuple(round(c, 2) for c in coords_3d)))
                live_coords_buffer.append(coords_3d)
                update_live_3d_plot()
            else:
                no_detection_counter += 1
                if no_detection_counter >= 30 and len(detections_list) >= 2:
                    print("\n\n--- 30 frames with no basketball detected ---")
                    for coord in detections_list:
                        print(f"Frame {coord[0]}: X={coord[1]} Y={coord[2]} Z={coord[3]}")
                    detections_list.clear()
                    no_detection_counter = 0

            sys.stdout.write("\033[F\033[K")
            print(f"FPS: {fps:.2f} (frame {frame_count})")
            sys.stdout.write("\033[K")
            if last_coords_3d:
                print(f"[{last_detected_frame}] Basketball at X={int(last_coords_3d[0])} Y={int(last_coords_3d[1])} Z={int(last_coords_3d[2])} mm", end="\r")
            else:
                print("No basketball detected", end="\r")

            show_detections(ir_bgr, coords_3d, frame_count, fps)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:  
        pipeline.stop()
        cv2.destroyAllWindows()
        plt.ioff()

if __name__ == "__main__":
    # Option 1: Live streaming
    # detection_loop(model_path="weights.pt", laser_power=10,record_to_bag=True)

  # Option 2: Bag file playback
    detection_loop(model_path="weights (1).pt", use_bag_file=True, bag_file_path="recording.bag")