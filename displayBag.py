import pyrealsense2 as rs
import numpy as np
import cv2

# Path to recorded .bag file
bag_path = "recording.bag"

# Set up pipeline and config
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_path, repeat_playback=False)

# Enable streams (must match what was recorded)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)

# Start playback
profile = pipeline.start(config)

print("Playing back recorded RealSense data...")

try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()

        if not depth_frame or not color_frame or not ir_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
            cv2.COLORMAP_JET
        )
        ir_image = cv2.cvtColor(np.asanyarray(ir_frame.get_data()), cv2.COLOR_GRAY2BGR)

        # Stack for display
        top = np.hstack((color_image, depth_colormap))
        bottom = np.hstack((ir_image, np.zeros_like(ir_image)))  # pad to match width

        full_view = np.vstack((top, bottom))
        cv2.imshow('Playback: Color | Depth | Infrared', full_view)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
except Exception as e:
    print("Playback finished or error:", e)
final
