import pyrealsense2 as rs
import time

# Configure streams
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)

# Save to bag file
config.enable_record_to_file("recording.bag")

# Start pipeline
profile = pipeline.start(config)

# Set laser power
depth_sensor = profile.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.laser_power):
    depth_sensor.set_option(rs.option.laser_power, 10)

print("Recording started... Press Ctrl+C to stop.")
try:
    while True:
        pipeline.wait_for_frames()
        time.sleep(0.01)  # Reduce CPU usage
except KeyboardInterrupt:
    print("\nRecording stopped.")
finally:
    pipeline.stop()
