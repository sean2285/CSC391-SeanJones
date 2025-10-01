import cv2
import numpy as np
import time


def convolve_gray(image, kernel):
    """Manual convolution for grayscale images."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float32),
                    ((pad_h, pad_h), (pad_w, pad_w)),
                    mode="edge")
    h, w = image.shape
    result = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            roi = padded[y:y+kh, x:x+kw]
            result[y, x] = np.sum(roi * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


FILTERS = {
    "0": ("None", np.array([[0,0,0],[0,1,0],[0,0,0]], np.float32)),
    "1": ("Box", (1/9) * np.ones((3,3), np.float32)),
    "2": ("Gaussian", (1/16) * np.array([[1,2,1],[2,4,2],[1,2,1]], np.float32)),
    "3": ("Sobel X", np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32)),
    "4": ("Sobel Y", np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)),
    "5": ("Sharpen", np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)),
    "6": ("Emboss", np.array([[-2,-1,0],[-1,1,1],[0,1,2]], np.float32)),
}


def open_cam(index=0, w=320, h=240):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open camera {index}")
    return cap


def run():
    cam_index = 0
    cap = open_cam(cam_index)
    filter_key = "0"
    filter_name, kernel = FILTERS[filter_key]

    fps_timer = time.time()
    frames = 0
    fps = 0.0

    print("Press keys 0–6 to change filters | c to switch camera | q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ Camera read failed, retrying...")
            cap.release()
            cap = open_cam(cam_index)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if filter_key == "0":
            processed = gray
        else:
            processed = convolve_gray(gray, kernel)
            if filter_key == "6":  # Emboss
                processed = np.clip(processed.astype(np.int16) + 128, 0, 255).astype(np.uint8)

        stacked = np.vstack((gray, processed))  # show one above the other

        # FPS counter
        frames += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = frames / (now - fps_timer)
            fps_timer, frames = now, 0

        # Overlay FPS
        cv2.putText(stacked, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)

        cv2.imshow("Grayscale (top) vs Filtered (bottom)", stacked)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if ord("0") <= key <= ord("6"):
            filter_key = chr(key)
            filter_name, kernel = FILTERS[filter_key]
            print(f"➡️ Switched to filter: {filter_name}")
        if key == ord("c"):
            cam_index = (cam_index + 1) % 3  # try 0,1,2
            cap.release()
            cap = open_cam(cam_index)
            print(f"🔄 Switched to camera {cam_index}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
