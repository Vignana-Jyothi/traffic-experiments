import pyopencl as cl

try:
    platforms = cl.get_platforms()
    print("OpenCL platforms found:")
    for idx, platform in enumerate(platforms):
        print(f"Platform {idx}: {platform.name}")
        devices = platform.get_devices()
        for dev_idx, device in enumerate(devices):
            print(f"  Device {dev_idx}: {device.name}")
except cl.LogicError as e:
    print(f"Error detecting platforms: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
