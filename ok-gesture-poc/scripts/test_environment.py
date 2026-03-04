# Phase 1 Test - Verify Environment
# Run this script to check if all dependencies are installed correctly

import sys


def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"[OK] {package_name}")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name} - {e}")
        return False


print("Checking Python Environment...")
print(f"Python version: {sys.version}")
print()

# Core packages
core_deps = [
    ("cv2", "opencv-python"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
]

print("Core Dependencies:")
all_ok = True
for module, package in core_deps:
    if not check_import(module, package):
        all_ok = False

print()

# ML/DL packages
ml_deps = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
]

print("ML/DL Dependencies:")
for module, package in ml_deps:
    if not check_import(module, package):
        all_ok = False

print()

# Tracking packages
tracking_deps = [
    ("scipy", "scipy"),
]

print("Tracking Dependencies:")
for module, package in tracking_deps:
    if not check_import(module, package):
        all_ok = False

print()

# Check camera
print("Checking Camera Access:")
try:
    import cv2

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[OK] Camera accessible: {width}x{height}")
        cap.release()
    else:
        print("[FAIL] Camera not accessible")
        all_ok = False
except Exception as e:
    print(f"[FAIL] Camera check failed: {e}")
    all_ok = False

print()

if all_ok:
    print("[OK] All critical dependencies installed!")
    print("\nYou can proceed to Phase 2: Data Collection")
else:
    print("[FAIL] Some dependencies missing. Install with:")
    print("  pip install -r requirements-minimal.txt")
    sys.exit(1)
