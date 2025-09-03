import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np
import traceback

# Load model once
MODEL_PATH = r"D:\AI-Camera-DressCode-main\AI Camera Dress Code\runs\detect\dresscode_baseline9\weights\best.pt"
model = YOLO(MODEL_PATH)

# Classes
violation_classes = ['Crop_Top', 'Miniskirt', 'Shorts', 'Sleeveless', 'low_neckline', 'ripped_pants']
class_names = model.names  # id -> name mapping


class DressCodeCamApp:
    def __init__(self, root, camera_index=0, conf_threshold=0.45):
        self.root = root
        self.root.title("📷 Dress Code Detector (Webcam)")
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        # enforce sensible minimum to avoid tiny dimensions
        self.root.minsize(600, 400)

        # layout
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.image_label = Label(root, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.status_label = Label(root, text="Starting webcam...", font=("Arial", 14))
        self.status_label.grid(row=1, column=0, pady=6, sticky="ew")

        Button(root, text="Exit", command=self.close_app).grid(row=2, column=0, pady=6)

        # camera and inference params
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.status_label.config(text=f"❌ Webcam {camera_index} not accessible.", fg="red")
            self.running = False
            return
        self.conf_threshold = conf_threshold

        # tracked display size (updated via configure event)
        self.win_w = 1280
        self.win_h = 720
        self._resize_threshold = 8
        self.root.bind("<Configure>", self.on_resize)

        # Start loop
        self.running = True
        self.root.after(100, self.update_frame)  # slight delay to let window initialize

    def on_resize(self, event):
        try:
            new_w = max(1, int(event.width))
            new_h = max(1, int(event.height) - 120)  # leave space for status/buttons
        except Exception:
            return
        if abs(new_w - self.win_w) > self._resize_threshold or abs(new_h - self.win_h) > self._resize_threshold:
            self.win_w, self.win_h = new_w, new_h

    def safe_results_to_arrays(self, results):
        """
        Convert an Ultralytics Results object to (boxes Nx4 numpy, class_ids N numpy).
        Robust to empty results and different versions of the API.
        """
        try:
            if results is None:
                return np.zeros((0, 4)), np.array([], dtype=int)

            boxes_obj = getattr(results, "boxes", None)
            if boxes_obj is None:
                # older API or empty
                return np.zeros((0, 4)), np.array([], dtype=int)

            # boxes_obj.xyxy should be a tensor-like Nx4
            xyxy = None
            if hasattr(boxes_obj, "xyxy"):
                xyxy = boxes_obj.xyxy
            elif hasattr(boxes_obj, "xyxyn"):  # fallback if different naming
                xyxy = boxes_obj.xyxyn
            else:
                # try attribute 'xyxy' via dict-like access
                xyxy = getattr(boxes_obj, "xyxy", None)

            if xyxy is None:
                return np.zeros((0, 4)), np.array([], dtype=int)

            # convert to numpy
            try:
                boxes_np = np.array(xyxy.cpu()) if hasattr(xyxy, "cpu") else np.array(xyxy)
            except Exception:
                boxes_np = np.array(xyxy)

            # classes
            cls_attr = getattr(boxes_obj, "cls", None)
            if cls_attr is None:
                # some versions keep classes in results.boxes.data[:, 5]
                try:
                    data = getattr(boxes_obj, "data", None)
                    if data is not None:
                        arr = np.array(data.cpu()) if hasattr(data, "cpu") else np.array(data)
                        if arr.shape[1] >= 6:
                            cls_np = arr[:, 5].astype(int)
                        else:
                            cls_np = np.array([], dtype=int)
                    else:
                        cls_np = np.array([], dtype=int)
                except Exception:
                    cls_np = np.array([], dtype=int)
            else:
                try:
                    cls_np = np.array(cls_attr.cpu()).astype(int) if hasattr(cls_attr, "cpu") else np.array(cls_attr).astype(int)
                except Exception:
                    cls_np = np.array(cls_attr).astype(int)

            # ensure shapes valid
            if boxes_np.ndim != 2 or boxes_np.shape[1] < 4:
                return np.zeros((0, 4)), np.array([], dtype=int)

            return boxes_np[:, :4], cls_np
        except Exception:
            traceback.print_exc()
            return np.zeros((0, 4)), np.array([], dtype=int)

    def update_frame(self):
        if not self.running:
            return

        try:
            if str(self.root.state()) == "iconic":
                self.root.after(250, self.update_frame)
                return
        except Exception:
            pass

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.status_label.config(text="Failed to access camera.", fg="red")
            self.root.after(500, self.update_frame)
            return

        # run inference (Ultralytics can accept a numpy array) -- robust call with fallback
        boxes_np, cls_np = np.zeros((0, 4)), np.array([], dtype=int)
        try:
            # preferred: use predict (supports more kwargs)
            results = model.predict(source=frame, conf=self.conf_threshold, save=False)
            # predict may return a list-like; pick first
            results = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results
            boxes_np, cls_np = self.safe_results_to_arrays(results)
        except Exception:
            # fallback: try direct call
            try:
                results = model(frame)  # sometimes model(...) works
                results = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results
                boxes_np, cls_np = self.safe_results_to_arrays(results)
            except Exception:
                # print traceback to console to help debugging
                traceback.print_exc()
                boxes_np, cls_np = np.zeros((0, 4)), np.array([], dtype=int)

        violations = []
        # Draw boxes on the frame
        for box, cls_id in zip(boxes_np, cls_np):
            try:
                x1, y1, x2, y2 = map(int, map(round, box[:4]))
            except Exception:
                continue
            cls_idx = int(cls_id)
            if isinstance(class_names, dict):
                label = class_names.get(cls_idx, str(cls_idx))
            else:
                try:
                    label = class_names[cls_idx]
                except Exception:
                    label = str(cls_idx)
            color = (0, 0, 255) if label in violation_classes else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if label in violation_classes:
                violations.append(label)

        # determine desired display size from tracked values (fall back to frame size)
        frame_h, frame_w = frame.shape[:2]
        win_w = self.win_w if self.win_w > 1 else frame_w
        win_h = self.win_h if self.win_h > 1 else frame_h

        # preserve aspect ratio
        scale = min(win_w / frame_w, win_h / frame_h)
        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        try:
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        except Exception:
            pass

        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        if violations:
            uniq = ", ".join(sorted(set(violations)))
            self.status_label.config(text="❌ Violation Detected: " + uniq, fg="red")
        else:
            self.status_label.config(text="✅ Dress Code OK", fg="green")

        self.root.after(30, self.update_frame)

    def close_app(self):
        self.running = False
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = DressCodeCamApp(root)
    root.mainloop()
