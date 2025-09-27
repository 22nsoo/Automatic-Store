#!/usr/bin/env python3
import time
import requests
import cv2
import numpy as np
from datetime import datetime, timezone
from ultralytics import YOLO

# =========================
# Firebase (REST) 설정
# =========================
FIREBASE_DB_URL = "https://inventory-tracking-1-default-rtdb.firebaseio.com"
FIREBASE_AUTH_QS = ""  # 인증을 쓰면 ?auth=토큰

def _fb_put(path, data):
    url = f"{FIREBASE_DB_URL}/{path}.json{FIREBASE_AUTH_QS}"
    return requests.put(url, json=data, timeout=5)

def _fb_post(path, data):
    url = f"{FIREBASE_DB_URL}/{path}.json{FIREBASE_AUTH_QS}"
    return requests.post(url, json=data, timeout=5)

def send_total_count(c):
    ts = datetime.now(timezone.utc).isoformat()
    _fb_put("people_counter/current", {"count": int(c), "updated_at": ts})
    _fb_post("people_counter/events",  {"count": int(c), "ts": ts})

def send_roi_count(c):
    ts = datetime.now(timezone.utc).isoformat()
    _fb_put("people_in_zone_counter/current", {"count": int(c), "updated_at": ts})
    _fb_post("people_in_zone_counter/events",  {"count": int(c), "ts": ts})

# ========== 카메라 오픈 유틸 (DSHOW → MSMF 순차 시도) ==========
def open_camera(index=0, w=640, h=480, fps=30):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    return cap if cap.isOpened() else None

def run_tracking_with_roi(cam_index=0, detect_every=10, conf_thresh=0.6):
    # 카메라
    cap = open_camera(cam_index, w=640, h=480, fps=30)
    if cap is None:
        raise RuntimeError("웹캠을 열 수 없습니다. (인덱스/권한/점유 확인)")

    # YOLOv8 Pose (가벼운 n 권장; m는 무거움)
    model = YOLO("yolov8n-pose.pt")

    # ROI: 1920x1080 기준 좌표 (원하는 좌표로 바꾸세요)
    roi_pts_1080p = np.array([[586, 428], [1091, 434], [1253, 1078], [231, 1079]], dtype=np.int32)
    base_w, base_h = 1920, 1080

    prev_total = None
    prev_roi = None
    idx = 0

    cv2.namedWindow("People Counter (q to quit)", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]
            sx, sy = w / base_w, h / base_h
            roi_pts = np.round(roi_pts_1080p * np.array([sx, sy])).astype(np.int32)

            annotated = frame.copy()
            total_count = prev_total if prev_total is not None else 0
            roi_count = prev_roi if prev_roi is not None else 0

            if idx % detect_every == 0:
                r = model.predict(source=frame, classes=[0], verbose=False)[0]
                annotated = r.plot()

                # 전체 사람 수(박스 신뢰도 기준)
                if r.boxes is not None and r.boxes.conf is not None:
                    confs = r.boxes.conf.detach().cpu().numpy()
                    total_count = int((confs >= conf_thresh).sum())
                else:
                    total_count = 0

                # ROI 내부 사람 수(양쪽 발목이 다각형 내부)
                roi_count = 0
                if getattr(r, "keypoints", None) is not None and getattr(r, "boxes", None) is not None:
                    for conf, kps in zip(r.boxes.conf, r.keypoints.xy):
                        if conf is None or float(conf) < conf_thresh:
                            continue
                        inside = True
                        for ank in (15, 16):  # ankles
                            x, y = kps[ank]
                            if x <= 0 or y <= 0 or cv2.pointPolygonTest(roi_pts, (int(x), int(y)), False) < 0:
                                inside = False
                                break
                        if inside:
                            roi_count += 1

                # 변화 시에만 로그 + Firebase 업데이트
                if prev_total is None or total_count != prev_total:
                    ts = datetime.now().strftime("%H:%M:%S")
                    delta = "" if prev_total is None else (f" (+{total_count - prev_total})" if total_count > prev_total else f" ({total_count - prev_total})")
                    print(f"[{ts}] total: {prev_total if prev_total is not None else '-'} -> {total_count}{delta}")
                    send_total_count(total_count)
                    prev_total = total_count

                if prev_roi is None or roi_count != prev_roi:
                    ts = datetime.now().strftime("%H:%M:%S")
                    delta = "" if prev_roi is None else (f" (+{roi_count - prev_roi})" if roi_count > prev_roi else f" ({roi_count - prev_roi})")
                    print(f"[{ts}] ROI  : {prev_roi if prev_roi is not None else '-'} -> {roi_count}{delta}")
                    send_roi_count(roi_count)
                    prev_roi = roi_count

            # 오버레이 후 창 표시
            cv2.polylines(annotated, [roi_pts], True, (255, 0, 0), 2)
            cv2.putText(annotated, f"Total: {prev_total or 0}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.putText(annotated, f"ROI  : {prev_roi or 0}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

            cv2.imshow("People Counter (q to quit)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking_with_roi(cam_index=0, detect_every=10, conf_thresh=0.6)
