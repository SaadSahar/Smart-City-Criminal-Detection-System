# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime, timedelta
import urllib.request
import pandas as pd
import threading
import pygame

# ===================== الإعدادات =====================
# مجلد الأشخاص: مجلد لكل شخص بداخله عدة صور
BASE_DIR = r"D:\\مشاريع تخرج\\تطبيقية\\تسجيل حضور\\data_base_img"

# اختر مصدر الكاميرا الواحدة:
SOURCE_IS_IP = True  # True: كاميرا IP, False: كاميرا ويب
IP_URL =  "http://192.168.136.32:8080/shot.jpg"  # مثال: .../shot.jpg
WEBCAM_INDEX = 0

# ملف الصوت (mp3/wav) لتشغيله عند التعرّف
SOUND_FILE = r"D:\python\project_cv\face_recognation\ding.mp3"

# مَهلة التهدئة لتفادي التكرار (بالثواني) لكل شخص
COOLDOWN_SECONDS = 30

# مسار ملف Excel المُصدّر عند الخروج
EXCEL_EXPORT_PATH = r"D:\f.xlsx"

# احفظ لقطات الوجوه المجهولة؟
SAVE_UNKNOWN = True
UNKNOWN_DIR = r"D:\unknown_faces"
# =====================================================

if SAVE_UNKNOWN:
    os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ========= صوت pygame + Threading =========
pygame.mixer.init()  # تهيئة الميكسر مرّة واحدة
def play_sound_bg(path: str):
    """ تشغيل الصوت بالخلفية دون تعطيل الحلقة الرئيسية. """
    if not os.path.isfile(path):
        print(f"[صوت] الملف غير موجود: {path}")
        return
    def _run():
        try:
            # ملاحظة: Sound يعطي latency أقل من music
            snd = pygame.mixer.Sound(path)
            snd.play()
            # ننتظر حتى ينتهي الصوت بدون حجب الخيط الرئيسي
            while pygame.mixer.get_busy():
                pygame.time.wait(50)
        except Exception as e:
            print(f"[صوت] خطأ تشغيل الصوت: {e}")
    threading.Thread(target=_run, daemon=True).start()

# ========= كلاس التعرف =========
class SimpleFacerec:
    def __init__(self, frame_resizing=0.25, tolerance=0.45):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = frame_resizing
        self.tolerance = tolerance

    def _is_image_file(self, path):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return os.path.splitext(path)[1].lower() in exts

    def load_encoding_images(self, base_dir):
        if not os.path.isdir(base_dir):
            raise ValueError(f"المسار غير صحيح: {base_dir}")

        persons = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not persons:
            raise ValueError("لا توجد مجلدات أشخاص داخل المسار المحدد.")

        total_imgs = 0
        total_encs = 0
        for person in persons:
            person_dir = os.path.join(base_dir, person)
            img_paths = [p for p in glob.glob(os.path.join(person_dir, "*")) if self._is_image_file(p)]
            if not img_paths:
                print(f"تحذير: لا توجد صور في مجلد {person}")
                continue

            print(f"[{person}] {len(img_paths)} صورة.")
            total_imgs += len(img_paths)

            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"تخطي ملف غير صالح: {img_path}")
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                if not encs:
                    print(f"لا يوجد وجه في: {os.path.basename(img_path)}")
                    continue

                self.known_face_encodings.append(encs[0])
                self.known_face_names.append(person)
                total_encs += 1

        if total_encs == 0:
            raise ValueError("لم يتم توليد أي ترميزات. تحقق من الصور.")

        print(f"تم التحميل: {total_encs} ترميز من {total_imgs} صورة لعدد {len(set(self.known_face_names))} أشخاص.")

    def detect_known_faces(self, frame):
        small = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        names = []
        for enc in face_encodings:
            if not self.known_face_encodings:
                names.append("Unknown")
                continue
            dists = face_recognition.face_distance(self.known_face_encodings, enc)
            idx = int(np.argmin(dists))
            name = self.known_face_names[idx] if dists[idx] <= self.tolerance else "Unknown"
            names.append(name)

        face_locations = np.array(face_locations)
        if len(face_locations) > 0:
            face_locations = (face_locations / self.frame_resizing).astype(int)

        return face_locations, names

# ========= تهيئة التعرف + قاعدة البيانات =========
sfr = SimpleFacerec(frame_resizing=0.25, tolerance=0.45)
sfr.load_encoding_images(BASE_DIR)

conn = sqlite3.connect('face_recognition_log.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS recognition_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp DATETIME,
    camera_id INTEGER
)
""")
conn.commit()

last_logged_at = {}  # name -> datetime آخر تسجيل

def should_log(name: str, now: datetime) -> bool:
    if name == "Unknown":
        return False
    prev = last_logged_at.get(name)
    return (prev is None) or ((now - prev) >= timedelta(seconds=COOLDOWN_SECONDS))

def draw_and_handle(frame, face_locations, face_names, camera_id=1):
    now = datetime.now()
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 250, 0) if name != "Unknown" else (0, 0, 200)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.putText(frame, f"{name} (ID:{camera_id})", (left, max(25, top - 10)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        # حفظ المجهول (اختياري)
        if SAVE_UNKNOWN and name == "Unknown":
            ts = now.strftime("%Y%m%d_%H%M%S_%f")
            crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
            if crop.size > 0:
                cv2.imwrite(os.path.join(UNKNOWN_DIR, f"unknown_{camera_id}_{ts}.jpg"), crop)

        # تسجيل + صوت
        if should_log(name, now):
            ts = now.strftime('%Y-%m-%d %H:%M:%S')
            try:
                cursor.execute("INSERT INTO recognition_log (name, timestamp, camera_id) VALUES (?, ?, ?)",
                               (name, ts, camera_id))
                conn.commit()
                last_logged_at[name] = now
                print(f"[LOG] {name} @ {ts} (cam {camera_id})")
                play_sound_bg(SOUND_FILE)
            except Exception as e:
                print(f"[DB] خطأ أثناء التسجيل: {e}")

    return frame

# ========= فتح المصدر الواحد =========
if SOURCE_IS_IP:
    cap = None  # سنقرأ بإستخدام urllib أدناه
else:
    cap = cv2.VideoCapture(WEBCAM_INDEX)

# ========= الحلقة الرئيسية =========
while True:
    # جلب إطار واحد فقط من المصدر
    if SOURCE_IS_IP:
        try:
            im_arr = np.array(bytearray(urllib.request.urlopen(IP_URL, timeout=3).read()), dtype=np.uint8)
            frame = cv2.imdecode(im_arr, -1)
            if frame is None:
                raise ValueError("تعذّر فك ترميز الإطار من كاميرا IP.")
        except Exception as e:
            # لو فشل، أعرض شاشة سوداء مع رسالة
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"IP cam error: {e}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        ret, frame = cap.read()
        if not ret:
            print("خطأ في قراءة كاميرا الويب.")
            break

    # تجهيز الإطار
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    # التعرف
    face_locations, face_names = sfr.detect_known_faces(frame)
    frame = draw_and_handle(frame, face_locations, face_names, camera_id=1)

    # عرض
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC: تصدير ثم خروج
        try:
            df = pd.read_sql_query("SELECT * FROM recognition_log", conn)
            df.to_excel(EXCEL_EXPORT_PATH, index=False)
            print(f"تم تصدير السجل إلى: {EXCEL_EXPORT_PATH}")
        except Exception as e:
            print(f"تعذّر التصدير إلى Excel: {e}")
        break

    elif key == ord('w'):  # مسح السجل فقط
        try:
            cursor.execute("DELETE FROM recognition_log")
            conn.commit()
            last_logged_at.clear()
            print("تم مسح السجل.")
        except Exception as e:
            print(f"تعذّر مسح السجل: {e}")

# تنظيف
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
conn.close()
pygame.mixer.quit()
