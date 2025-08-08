import cv2
import numpy as np
import math
import glob
import threading
import time
import playsound
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from email import encoders
from email.mime.base import MIMEBase

# ----------------------------
# Config (fill these)
# ----------------------------
SENDER_EMAIL = "auffarooqui75@gmail.com"        # e.g. "auffarooqui74@gmail.com"
APP_PASSWORD = "mrwtywqhyvoujxuj"            # Gmail app password (do NOT hardcode in production)
RECIPIENT_EMAIL = "cyverraadwh@gmail.com"    # destination: e.g. "cyverraadwh@gmail.com"
EMAIL_SUBJECT = "🔥 Fire Alert - Live Photos"
EMAIL_BODY = "Fire detected. Attached latest photos (captured at detection)."
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
# ----------------------------

frame_count = 0
last_sound_time = 0
last_zoom_time = 0
zoom_hold_start_time = 0
zoom_active = False
zoom_hold = False
zoom_lock = threading.Lock()
fire_detected_in_main_window = False
last_photo_time = 0  # last photo capture time
send_lock = threading.Lock()  # prevents parallel email sends

# Folders
save_dir = "fire_alert_photos"
sent_dir = "sent_photos"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(sent_dir, exist_ok=True)

# ----------------------------
# Email sending helper
# ----------------------------
def send_email_with_attachments(sender, password, recipient, subject, body, file_paths):
    """
    Sends an email using Gmail SMTP with the given attachments.
    Returns True on success, else False.
    """
    try:
        msg = EmailMessage()
        msg["From"] = formataddr(("Alert System", sender))
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)

        for fp in file_paths:
            try:
                with open(fp, "rb") as f:
                    data = f.read()
                maintype = "application"
                # Attach as octet-stream – images will still be viewable
                part = MIMEBase("application", "octet-stream")
                part.set_payload(data)
                encoders.encode_base64(part)
                filename = os.path.basename(fp)
                part.add_header("Content-Disposition", f"attachment; filename={filename}")
                msg.add_attachment(data, maintype="image", subtype="jpeg", filename=filename)
            except Exception as e:
                print(f"⚠️ Could not attach {fp}: {e}")

        # SMTP send
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent with attachments:", file_paths)
        return True
    except Exception as e:
        print("❌ Failed to send email:", e)
        return False

def move_files_to_sent(file_paths):
    moved = []
    for fp in file_paths:
        try:
            base = os.path.basename(fp)
            dest = os.path.join(sent_dir, base)
            os.replace(fp, dest)  # atomic move/rename
            moved.append(dest)
        except Exception as e:
            print(f"⚠️ Could not move {fp} to sent: {e}")
    return moved

def send_alert_thread(file_paths):
    """
    Runs in background: sends email with file_paths and then moves them to sent folder.
    Uses send_lock to avoid races.
    """
    if not file_paths:
        return
    # Acquire send lock to avoid multiple concurrent sends
    if not send_lock.acquire(blocking=False):
        print("🔁 Send already in progress, skipping this send.")
        return

    def _worker():
        try:
            # Only send files that still exist and are recent
            valid_files = []
            now_ts = time.time()
            for f in file_paths:
                if os.path.exists(f):
                    # check recency: file created/modified in last 30 seconds
                    mtime = os.path.getmtime(f)
                    if now_ts - mtime <= 60:  # 60s grace window
                        valid_files.append(f)
            if not valid_files:
                print("⚠️ No valid recent files to send.")
                return

            success = send_email_with_attachments(SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL,
                                                 EMAIL_SUBJECT, EMAIL_BODY, valid_files)
            if success:
                moved = move_files_to_sent(valid_files)
                print(f"➡️ Moved sent files to '{sent_dir}': {moved}")
        finally:
            send_lock.release()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

# ----------------------------
# Existing detection code (kept same structure)
# ----------------------------
class ThreadedCamera:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.ret, self.frame = self.capture.read()
        self.running = True
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.capture.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.capture.release()

def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 1
    return (4 * math.pi * area) / (perimeter * perimeter)

def is_flame_shape(contour):
    area = cv2.contourArea(contour)
    if area < 150:
        return False
    perimeter = cv2.arcLength(contour, True)
    circularity = calculate_circularity(area, perimeter)
    if circularity > 0.78:
        return False
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0 or float(area) / hull_area > 0.95:
        return False
    return True

def load_orb_descriptors(file_list):
    descriptors = []
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            print(f"⚠️ Could not load: {file}")
            continue
        img = cv2.resize(img, (320, 320))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = orb.detectAndCompute(gray, None)
        if des is not None:
            descriptors.append(des)
    return descriptors

def get_orb_match_score(des_frame, sample_descriptors):
    if des_frame is None:
        return 0
    match_count = 0
    for des_sample in sample_descriptors:
        if des_sample is None or des_sample.shape[1] != des_frame.shape[1]:
            continue
        matches = bf.match(des_sample, des_frame)
        good = [m for m in matches if m.distance < 70]
        if len(good) >= 10:
            match_count += 1
    return match_count

def save_photos(frame):
    """
    Save 3 full-frame photos right away. Return list of saved file paths (new files).
    """
    saved = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(1, 4):
        filename = os.path.join(save_dir, f"fire_{timestamp}_{i}.jpg")
        # Save a higher-quality version by writing directly (OpenCV default uses JPEG high-ish quality)
        cv2.imwrite(filename, frame)
        saved.append(filename)
        time.sleep(0.2)  # small gap so files have slightly different timestamps
    print(f"📸 3 photos saved in '{save_dir}': {saved}")
    return saved

def zoom_worker(frame):
    global zoom_active, zoom_hold, zoom_hold_start_time, last_zoom_time
    window_name = "🔥 Zoom Detection"
    zoom_levels = [2, 4, 8]
    fire_detected = False

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2  # center of the frame

    with zoom_lock:
        for scale in zoom_levels:
            # Get center ROI for zoom
            zoom_w = w // scale
            zoom_h = h // scale
            x1 = max(cx - zoom_w // 2, 0)
            y1 = max(cy - zoom_h // 2, 0)
            x2 = min(cx + zoom_w // 2, w)
            y2 = min(cy + zoom_h // 2, h)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            zoomed = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale))
            cv2.imshow(window_name, zoomed)
            cv2.waitKey(1)

            gray_zoomed = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
            _, des_zoomed = orb.detectAndCompute(gray_zoomed, None)
            if des_zoomed is None:
                continue

            fire_score = get_orb_match_score(des_zoomed, fire_descriptors)
            flicker = np.std(cv2.cvtColor(zoomed, cv2.COLOR_BGR2HSV)[:, :, 2]) > 15

            if flicker and fire_score >= 1:
                fire_detected = True
                print(f"🔥 Fire confirmed at {scale}x zoom")
                zoom_active = True
                zoom_hold = True
                zoom_hold_start_time = time.time()
                # Hold display for a short while
                while zoom_hold:
                    cv2.imshow(window_name, zoomed)
                    cv2.waitKey(1)
                    if time.time() - zoom_hold_start_time > 5:
                        zoom_hold = False
                break  # exit from zoom_levels loop if fire detected

            time.sleep(1)

        # If no fire detected even at 8x
        if not fire_detected:
            print("❌ No fire found in any zoom level.")
            zoom_active = True  # prevent re-triggering immediately
            try:
                cv2.imshow(window_name, zoomed)
                cv2.waitKey(1)
            except:
                pass
            time.sleep(10)  # Wait before next attempt
            try:
                cv2.destroyWindow(window_name)
            except:
                pass
            zoom_active = False
        else:
            zoom_active = False
            try:
                cv2.destroyWindow(window_name)
            except:
                pass

# ORB and BF
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load samples
fire_files = sorted(glob.glob("fire*.jp*g"))
no_fire_files = sorted(glob.glob("no_fire*.jp*g"))

print("📥 Loading fire and no-fire samples...")
fire_descriptors = load_orb_descriptors(fire_files)
no_fire_descriptors = load_orb_descriptors(no_fire_files)
print(f"✅ Loaded {len(fire_descriptors)} fire samples and {len(no_fire_descriptors)} no-fire samples.")

# If you have dataset matching requirement, you can load dataset_descriptors separately:
dataset_descriptors = fire_descriptors  # reuse

# Camera init (change IP/source as needed)
camera = ThreadedCamera("http://10.125.159.217:8080/video")
print("🔥 Fire Detection Started...")

prev_gray_for_motion = None

while True:
    ret, frame = camera.read()
    frame_count += 1
    if not ret or frame is None or frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (320, 320))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(hsv, (0, 80, 80), (40, 255, 255))
    blur = cv2.GaussianBlur(mask, (7, 7), 2)
    contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious_found = False
    target_box = None

    for cnt in contours:
        if not is_flame_shape(cnt):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        suspicious_found = True
        target_box = (x, y, w, h)
        break

    if suspicious_found:
        x, y, w, h = target_box
        # ensure bounding box inside frame
        x = max(0, x); y = max(0, y)
        w = max(1, min(w, frame.shape[1]-x))
        h = max(1, min(h, frame.shape[0]-y))

        roi_gray = cv2.resize(gray[y:y+h, x:x+w], (320, 320))
        _, des_roi = orb.detectAndCompute(roi_gray, None)
        if des_roi is None:
            continue

        roi_v = v[y:y+h, x:x+w]
        flicker = np.std(roi_v) > 15 and np.mean(roi_v) > 140

        motion_detected = False
        if prev_gray_for_motion is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray_for_motion, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # safe indexing
            mean_motion = np.mean(mag[y:y+h, x:x+w]) if mag.shape[0] > (y+h-1) and mag.shape[1] > (x+w-1) else np.mean(mag)
            if mean_motion > 1.5:
                motion_detected = True
        prev_gray_for_motion = gray.copy()

        fire_score = get_orb_match_score(des_roi, fire_descriptors)

        # ---------------------------
        # Main alert condition (unchanged)
        # ---------------------------
        if flicker and motion_detected and fire_score >= 1:
            current_time = time.time()
            # sound cooldown
            if current_time - last_sound_time >= 5:
                try:
                    threading.Thread(target=playsound.playsound, args=("alert.mp3",), daemon=True).start()
                except Exception as e:
                    print("⚠️ playsound error:", e)
                last_sound_time = current_time

                # 📸 Save photos and send email only once per detection window
                # Use last_photo_time to avoid duplicates
                if current_time - last_photo_time >= 5:
                    saved_files = save_photos(frame)  # returns list of file paths
                    last_photo_time = current_time
                    # start background send (only these saved files)
                    send_alert_thread(saved_files)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "🔥 FIRE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if zoom_active:
                zoom_hold_start_time = time.time()  # Reset hold time

        # trigger zoom worker occasionally
        if time.time() - last_zoom_time >= 10 and not zoom_active:
            threading.Thread(target=zoom_worker, args=(frame.copy(),), daemon=True).start()
            last_zoom_time = time.time()

    else:
        zoom_active = False

    try:
        cv2.imshow("🔥 Fire Detector", frame)
    except:
        pass
    if cv2.waitKey(10) == 27:
        break

camera.stop()
