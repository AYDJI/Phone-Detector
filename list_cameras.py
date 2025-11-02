import cv2
import platform
import subprocess

def list_camera_indexes(max_tested=10):
    """Return a list of camera indexes that can be opened."""
    indexes = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                indexes.append(i)
            cap.release()
    return indexes

def get_camera_names():
    """Try to get human-readable camera names using system tools."""
    system = platform.system()
    names = {}

    try:
        if system == "Windows":
            # Use pygrabber if available
            try:
                from pygrabber.dshow_graph import FilterGraph
                graph = FilterGraph()
                devices = graph.get_input_devices()
                for i, name in enumerate(devices):
                    names[i] = name
            except ImportError:
                print("(Tip: install pygrabber for camera names: pip install pygrabber)")

        elif system == "Linux":
            # Use v4l2-ctl if available
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True
            )
            lines = result.stdout.strip().splitlines()
            dev_name = None
            for line in lines:
                if not line.startswith("\t"):
                    dev_name = line.strip()
                else:
                    dev_path = line.strip()
                    # Extract index number from /dev/videoX
                    if "/dev/video" in dev_path:
                        try:
                            idx = int(dev_path.split("video")[1])
                            names[idx] = dev_name
                        except ValueError:
                            pass

        elif system == "Darwin":  # macOS
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True, text=True
            )
            lines = result.stdout.splitlines()
            camera_names = [l.strip() for l in lines if l.strip().endswith("Camera:")]
            for i, name in enumerate(camera_names):
                names[i] = name.replace(":", "")
    except Exception as e:
        print("Could not read camera names:", e)

    return names


if __name__ == "__main__":
    available = list_camera_indexes()
    names = get_camera_names()

    if not available:
        print("No cameras found.")
    else:
        print("Detected cameras:")
        for idx in available:
            cam_name = names.get(idx, "(unknown name)")
            print(f"  Index {idx}: {cam_name}")
