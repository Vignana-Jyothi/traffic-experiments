import pyopencl as cl

platforms = cl.get_platforms()
for platform in platforms:
    print(f"Platform: {platform.name}")
    devices = platform.get_devices()
    for device in devices:
        print(f"  Device: {device.name} ({device.type})")

