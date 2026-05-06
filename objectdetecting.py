from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opened")
    exit()

# Line position
line_y = 250

# Tracking
tracked_objects = {}
object_id = 0
max_disappear = 10

# Counters
count_up = {}
count_down = {}

running = False

def get_center(x1, y1, x2, y2):
    return int((x1+x2)/2), int((y1+y2)/2)

def update_frame():
    global tracked_objects, object_id, running

    if not running:
        root.after(100, update_frame)
        return

    ret, frame = cap.read()
    if not ret:
        root.after(100, update_frame)
        return

    frame = cv2.resize(frame, (640, 480))

    results = model(frame)[0]
    new_tracked = {}

    if results.boxes is not None:
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]

            center = get_center(x1, y1, x2, y2)
            matched = False

            for obj_id, obj in tracked_objects.items():
                prev_center = obj["center"]
                dist = np.linalg.norm(np.array(center) - np.array(prev_center))

                if dist < 50:
                    matched = True

                    new_tracked[obj_id] = {
                        "center": center,
                        "label": label,
                        "counted": obj["counted"],
                        "miss": 0
                    }

                    if not obj["counted"]:
                        if prev_center[1] < line_y and center[1] >= line_y:
                            count_down[label] = count_down.get(label, 0) + 1
                            new_tracked[obj_id]["counted"] = True

                        elif prev_center[1] > line_y and center[1] <= line_y:
                            count_up[label] = count_up.get(label, 0) + 1
                            new_tracked[obj_id]["counted"] = True

                    break

            if not matched:
                new_tracked[object_id] = {
                    "center": center,
                    "label": label,
                    "counted": False,
                    "miss": 0
                }
                object_id += 1

            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.circle(frame, center, 4, (0,0,255), -1)

    # Handle lost objects
    for obj_id, obj in tracked_objects.items():
        if obj_id not in new_tracked:
            obj["miss"] += 1
            if obj["miss"] < max_disappear:
                new_tracked[obj_id] = obj

    tracked_objects = new_tracked

    # Draw counting line
    cv2.line(frame, (0,line_y), (640,line_y), (255,0,0), 2)

    # Convert to Tkinter image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Display counts
    text = "UP COUNT\n---------\n"
    for k, v in count_up.items():
        text += f"{k}: {v}\n"

    text += "\nDOWN COUNT\n-----------\n"
    for k, v in count_down.items():
        text += f"{k}: {v}\n"

    count_label.config(text=text)

    root.after(30, update_frame)

def start():
    global running
    running = True

def stop():
    global running
    running = False

def reset():
    global count_up, count_down, tracked_objects, object_id
    count_up = {}
    count_down = {}
    tracked_objects = {}
    object_id = 0

# GUI setup
root = tk.Tk()
root.title("AI Object Counting Dashboard")

video_label = tk.Label(root)
video_label.pack()

count_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
count_label.pack()

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Start", command=start).pack(side="left", padx=10)
tk.Button(btn_frame, text="Stop", command=stop).pack(side="left", padx=10)
tk.Button(btn_frame, text="Reset", command=reset).pack(side="left", padx=10)

# Start loop
root.after(100, update_frame)

root.mainloop()

cap.release()
cv2.destroyAllWindows()