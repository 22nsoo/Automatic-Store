#!/usr/bin/env python3
import time
import requests
import cv2
import numpy as np
from datetime import datetime, timezone
from threading import Thread, Event, Lock
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO

# =========================
# Firebase (REST) 설정
# =========================
FIREBASE_DB_URL = "https://inventory-tracking-1-default-rtdb.firebaseio.com"  # ← 네 RTDB URL
FIREBASE_AUTH_QS = ""  # 인증 쓰면 "?auth=ID_TOKEN" 등으로 설정

def _fb_url(path: str) -> str:
    path = path.strip("/")
    return f"{FIREBASE_DB_URL}/{path}.json{FIREBASE_AUTH_QS}"

def fb_get(path: str):
    r = requests.get(_fb_url(path), timeout=5)
    r.raise_for_status()
    return r.json()

def fb_put(path: str, data):
    r = requests.put(_fb_url(path), json=data, timeout=5)
    r.raise_for_status()
    return r.json()

def fb_post(path: str, data):
    r = requests.post(_fb_url(path), json=data, timeout=5)
    r.raise_for_status()
    return r.json()

def fb_delete(path: str):
    r = requests.delete(_fb_url(path), timeout=5)
    r.raise_for_status()
    return r.json()

# =========================
# 전역 상태
# =========================
cart = {}
last_detected = {}
cooldown = 5

zone_active = False          # ROI 내 인원수 >= 1 ?
zone_lock = Lock()

stop_event = Event()         # 전체 종료 신호 (결제 버튼에서 set)
yolo_ready = Event()

# =========================
# Tkinter 메인 장바구니 창
# =========================
cart_root = tk.Tk()
cart_root.title("🛒 장바구니")
cart_root.geometry("300x460")
cart_root.attributes("-topmost", True)

# 시작 시 창 숨김
cart_root.withdraw()

tk.Label(cart_root, text="실시간 장바구니", font=("Arial", 14)).pack(pady=10)
cart_frame = tk.Frame(cart_root); cart_frame.pack()

def update_cart_display():
    for w in cart_frame.winfo_children():
        w.destroy()
    for label, count in cart.items():
        if count > 0:
            tk.Label(cart_frame, text=f"{label} × {count}", font=("Arial", 12)).pack(anchor='w')

def toggle_cart_window(active: bool):
    """ROI 활성/비활성에 따라 창을 보이거나 숨김 (Tk 메인스레드)"""
    if active:
        cart_root.deiconify()
        cart_root.lift()
        cart_root.attributes('-topmost', True)
    else:
        cart_root.withdraw()

def show_popup(label):
    # ROI 비활성 시 팝업 생성 방지(동시성 보호)
    with zone_lock:
        if not zone_active:
            return

    def popup():
        win = tk.Toplevel(cart_root)
        win.title("제품 감지됨")
        win.geometry("280x150")
        win.attributes('-topmost', True)

        tk.Label(win, text=f"'{label}'이(가) 감지되었습니다!", font=("Arial", 12)).pack(pady=10)
        tk.Label(win, text="장바구니에 담으시겠습니까?", font=("Arial", 10)).pack(pady=5)

        def add_to_cart():
            cart[label] = cart.get(label, 0) + 1
            update_cart_display()
            print(f"[➕ 담기] {label} → 수량: {cart[label]}")
            try:
                fb_put(f"cart/{label}", cart[label])
            except Exception as e:
                print("[WARN] cart update failed:", e)
            win.destroy()

        def cancel():
            print(f"[❌ 취소] {label}")
            win.destroy()

        tk.Button(win, text="장바구니 담기", command=add_to_cart, width=12).pack(pady=5)
        tk.Button(win, text="취소", command=cancel, width=12).pack(pady=2)

    cart_root.after(0, popup)

def checkout():
    print(f"[✅ 결제 완료] {cart}")
    messagebox.showinfo("결제", "결제가 완료되었습니다!")
    # 판매/재고 반영 (REST)
    for label, count in cart.items():
        try:
            prev_sales = fb_get(f"sales/{label}") or 0
            fb_put(f"sales/{label}", int(prev_sales) + int(count))
        except Exception as e:
            print(f"[WARN] sales update failed {label}:", e)

        try:
            inv = fb_get(f"inventory/{label}")
            if inv is not None:
                new_inv = max(0, int(inv) - int(count))
                fb_put(f"inventory/{label}", new_inv)
                print(f"[📦 재고 반영] {label}: {inv} → {new_inv}")
            else:
                print(f"[⚠️] '{label}' 재고 정보 없음")
        except Exception as e:
            print(f"[WARN] inventory update failed {label}:", e)

    cart.clear()
    update_cart_display()
    try:
        fb_delete("cart")
    except Exception as e:
        print("[WARN] cart delete failed:", e)

    # 전체 종료
    stop_event.set()
    cart_root.after(300, cart_root.destroy)

def cancel_cart():
    print("[🛑 장바구니 취소] 초기화됨")
    cart.clear()
    update_cart_display()
    try:
        fb_delete("cart")
    except Exception as e:
        print("[WARN] cart delete failed:", e)

tk.Button(cart_root, text="결제하기", command=checkout, width=15, bg="green", fg="white").pack(pady=5)
tk.Button(cart_root, text="장바구니 초기화", command=cancel_cart, width=15, bg="red", fg="white").pack(pady=5)

# =========================
# ROI 인원수 폴링 (REST)
# =========================
def poll_people_in_zone(interval_sec=1.0):
    global zone_active
    path = "people_in_zone_counter/current/count"
    last = None
    while not stop_event.is_set():
        try:
            val = fb_get(path)
            active = (isinstance(val, (int, float)) and val >= 1)
            with zone_lock:
                zone_active = active
            # 상태 변화 시에만 창 토글
            if active != last:
                last = active
                cart_root.after(0, toggle_cart_window, active)
        except Exception as e:
            print("[WARN] people_in_zone_counter polling failed:", e)
        time.sleep(interval_sec)

# =========================
# YOLO 감지 쓰레드 (대기 중 창은 숨김, 감지만 일시정지)
# =========================
def run_yolo():
    # ★ 모델 경로를 너의 파일로 바꿔
    MODEL_PATH = "best_v3.pt"  # 예: runs/detect/train/weights/best.pt
    model = YOLO(MODEL_PATH)
    names = model.model.names

    # 카메라 열기 (DSHOW → MSMF)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("[ERR] 웹캠을 열 수 없습니다.")
        stop_event.set()
        return

    yolo_ready.set()

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            with zone_lock:
                active_now = zone_active

            if not active_now:
                # ROI 비활성: 감지 일시정지 (OpenCV 창은 띄우지 않음)
                # 필요하면 아래 두 줄 주석 해제해서 카메라 미리보기도 숨길 수 있음
                # cv2.imshow("YOLOv8 실시간 탐지", frame)
                # cv2.waitKey(1)
                time.sleep(0.05)
                continue

            # 활성 상태: 감지
            results = model.predict(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    conf = float(box.conf[0])
                    if label in ['homerunball', 'miz'] and conf > 0.85:
                        now = time.time()
                        if label not in last_detected or now - last_detected[label] > cooldown:
                            last_detected[label] = now
                            print(f"[🎯 감지] {label} (conf={conf:.2f}) → 팝업")
                            show_popup(label)

            # 활성 상태에서만 프리뷰 표시
            cv2.imshow("YOLOv8 실시간 탐지", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                stop_event.set()
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# =========================
# 쓰레드 시작 & Tk 메인루프
# =========================
Thread(target=poll_people_in_zone, daemon=True).start()
Thread(target=run_yolo, daemon=True).start()
cart_root.mainloop()
