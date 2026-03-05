import cv2
import easyocr
import re
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, deque

model = YOLO("license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=False)

plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}

def correct_plate_format(ocr_text):

    mapping_num_to_alpha = {"0":"O","1":"I","5":"S","8":"B"}
    mapping_alpha_to_num = {"O":"0","I":"1","Z":"2","S":"5","B":"8"}

    ocr_text = ocr_text.upper().replace(" ","")

    if len(ocr_text) != 7:
        return ""

    corrected=[]

    for i,ch in enumerate(ocr_text):

        if i < 2 or i >= 4:

            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])

            elif ch.isalpha():
                corrected.append(ch)

            else:
                return ""

        else:

            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])

            elif ch.isdigit():
                corrected.append(ch)

            else:
                return ""

    return "".join(corrected)


def recognize_plate(plate_crop):

    if plate_crop is None or plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )

    plate_resized = cv2.resize(
        thresh,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC
    )

    try:

        result = reader.readtext(
            plate_resized,
            detail=0,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        if len(result) > 0:

            candidate = correct_plate_format(result[0])

            if candidate and plate_pattern.match(candidate):
                return candidate

    except:
        pass

    return ""


def get_box_id(x1,y1,x2,y2):
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"


def get_stable_plate(box_id,new_text):

    if new_text:

        plate_history[box_id].append(new_text)

        most_common = max(
            set(plate_history[box_id]),
            key=plate_history[box_id].count
        )

        plate_final[box_id] = most_common

    return plate_final.get(box_id,"")


def process_video(video_path, progress_callback=None):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detected_data=[]

    frame_count=0

    while cap.isOpened():

        ret,frame = cap.read()

        if not ret:
            break

        results = model(frame,verbose=False)

        for r in results:

            for box in r.boxes:

                conf = float(box.conf.cpu().numpy()[0])

                if conf < 0.3:
                    continue

                x1,y1,x2,y2 = map(int,box.xyxy.cpu().numpy()[0])

                plate_crop = frame[y1:y2,x1:x2]

                text = recognize_plate(plate_crop)

                box_id = get_box_id(x1,y1,x2,y2)

                stable = get_stable_plate(box_id,text)

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000

                if stable:

                    detected_data.append({
                        "plate":stable,
                        "timestamp":round(timestamp,2),
                        "image":plate_crop
                    })

        frame_count+=1

        if progress_callback:
            progress_callback(frame_count/total_frames)

    cap.release()

    return detected_data

