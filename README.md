# Automatic-Store
=======
# Automatic-Store
## Smart Snack Tracking System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Firebase](https://img.shields.io/badge/Firebase-Realtime%20Database-yellow)
![Tkinter](https://img.shields.io/badge/UI-Tkinter-lightgrey)

A smart retail demo system that combines **real-time people tracking**, **ROI-based activation**, **snack detection**, and a **Firebase-connected checkout flow**.

실시간 **사람 수 카운팅**과 **ROI(관심 구역) 내부 사용자 감지**를 기반으로,
**스낵 객체 인식**, **장바구니 UI**, **결제 및 재고 반영**까지 연결한 스마트 무인 매대 시스템입니다.

사람이 특정 구역(ROI)에 들어오면 상품 감지가 활성화되고,
감지된 상품은 장바구니에 추가되며 결제 시 Firebase에 판매량과 재고가 반영됩니다.

---

## Overview

이 프로젝트는 두 개의 핵심 모듈로 구성됩니다.

### `people_tracking_lee.py`
- YOLOv8 Pose 기반 사람 탐지
- 전체 인원 수 계산
- ROI 내부 인원 수 계산
- Firebase Realtime Database 업데이트

### `object_snacks.py`
- Firebase에서 ROI 내부 인원 수 확인
- ROI 활성 시에만 상품 감지 수행
- 장바구니 UI 제공
- 결제 시 판매량 및 재고 업데이트

---

## Demo Flow

```text
Webcam
  ↓
People Tracking (YOLOv8 Pose)
  ↓
Total Count / ROI Count
  ↓
Firebase RTDB 저장
  ↓
Object Detection 활성화
  ↓
Snack Detection (YOLO)
  ↓
Cart UI
  ↓
Checkout
  ↓
Sales / Inventory Update
```

---

## Project Structure

```text
.
├── people_tracking_lee.py   # 사람 수 / ROI 내부 인원 수 카운팅
└── object_snacks.py         # 상품 감지 / 장바구니 / 결제 / 재고 반영
```

---

## Features

### Real-time People Counting
- OpenCV 웹캠 입력 사용
- `yolov8n-pose.pt` 기반 사람 탐지
- 전체 인원 수 계산
- 양쪽 발목 keypoint를 기준으로 ROI 내부 여부 판단

### ROI-based Activation
- Firebase의 ROI 내부 인원 수를 기준으로 상품 감지 활성화
- 사용자가 ROI 안에 있을 때만 snack detection 수행
- ROI 밖에서는 감지 중지 및 장바구니 UI 숨김

### Snack Detection
- `best_v3.pt` 모델 기반 상품 탐지
- 특정 confidence 이상일 때만 감지 처리
- 동일 상품의 중복 감지를 막기 위한 cooldown 적용

### Cart & Checkout UI
- Tkinter 기반 장바구니 인터페이스
- 감지된 상품을 장바구니에 추가
- 결제 시 판매량 증가 및 재고 감소
- cart 데이터 초기화

---

## Tech Stack

- **Language**: Python
- **Computer Vision**: OpenCV, Ultralytics YOLOv8
- **Pose Estimation**: YOLOv8 Pose
- **Database**: Firebase Realtime Database
- **UI**: Tkinter
- **Utilities**: NumPy, Requests

---

## Requirements

- Python 3.9+
- OpenCV
- NumPy
- Requests
- Ultralytics
- Tkinter

### Installation

```bash
pip install ultralytics opencv-python numpy requests
```

> `tkinter`는 운영체제에 따라 별도 설치가 필요할 수 있습니다.

---

## Model Files

이 프로젝트는 두 종류의 모델을 사용합니다.

### People Tracking Model
- `yolov8n-pose.pt`

### Snack Detection Model
- `best_v3.pt`

실행 전에 모델 파일이 프로젝트 디렉터리에 위치해 있어야 합니다.
필요하다면 코드 내 모델 경로를 환경에 맞게 수정하세요.

---

## Firebase Realtime Database Structure

예시 구조는 아래와 같습니다.

```json
{
  "people_counter": {
    "current": {
      "count": 0,
      "updated_at": "2026-03-08T00:00:00Z"
    },
    "events": {
      "-event_id": {
        "count": 1,
        "ts": "2026-03-08T00:00:00Z"
      }
    }
  },
  "people_in_zone_counter": {
    "current": {
      "count": 0,
      "updated_at": "2026-03-08T00:00:00Z"
    },
    "events": {
      "-event_id": {
        "count": 1,
        "ts": "2026-03-08T00:00:00Z"
      }
    }
  },
  "cart": {
    "miz": 1,
    "homerunball": 2
  },
  "sales": {
    "miz": 10,
    "homerunball": 7
  },
  "inventory": {
    "miz": 20,
    "homerunball": 15
  }
}
```

---

## How It Works

### 1. People Tracking

`people_tracking_lee.py`는 웹캠 프레임에서 사람을 검출하고,
전체 인원 수와 ROI 내부 인원 수를 계산하여 Firebase에 저장합니다.

#### Run
```bash
python people_tracking_lee.py
```

#### Main Process
- 웹캠 오픈
- 사람 탐지
- ROI 내부 인원 수 계산
- Firebase 업데이트
- OpenCV 창 출력 (`q` 입력 시 종료)

### 2. Snack Detection & Cart

`object_snacks.py`는 Firebase에서 ROI 내부 인원 수를 확인하고,
ROI 안에 사용자가 있을 때만 상품 감지를 수행합니다.

#### Run
```bash
python object_snacks.py
```

#### Main Process
- Firebase polling
- ROI 활성 여부 확인
- 상품 감지
- 장바구니 추가 팝업
- 결제 처리
- 판매량 및 재고 업데이트

---

## Key Parameters

### `people_tracking_lee.py`

```python
cam_index = 0
detect_every = 10
conf_thresh = 0.6
```

- `cam_index`: 사용할 카메라 인덱스
- `detect_every`: 몇 프레임마다 탐지를 수행할지 결정
- `conf_thresh`: 사람 탐지 confidence threshold

### `object_snacks.py`

```python
cooldown = 5
```

- `cooldown`: 동일 상품 중복 감지 방지 시간(초)

#### Target Labels
- `homerunball`
- `miz`

#### Detection Threshold
- `conf > 0.85`

---

## ROI Logic

ROI는 특정 polygon 영역으로 정의되며,
사람의 양쪽 발목 keypoint가 모두 해당 polygon 내부에 있을 때
그 사람을 ROI 내부 사용자로 판단합니다.

이 방식은 단순 bounding box 기반 방식보다
실제로 구역 안에 들어온 사용자를 보다 안정적으로 판별하는 데 적합합니다.

---

## UI Behavior

### When no user is inside ROI
- 상품 감지 비활성화
- 장바구니 창 숨김

### When a user is inside ROI
- 상품 감지 활성화
- 장바구니 창 표시
- 감지된 상품에 대해 장바구니 추가 여부 팝업 표시

---

## Example Use Cases

- 스마트 스낵바 / 무인 매대
- 비전 기반 상품 인식 데모
- Firebase 기반 실시간 모니터링 시스템
- 스마트 편의점 / 스마트 리테일 프로토타입

---

## Future Improvements

- 상품 라벨 및 가격 정보 외부 설정 파일화
- 다중 상품 확장
- 사용자 세션 기반 장바구니 분리
- Streamlit 대시보드 연동
- ONNX / TensorRT 기반 추론 최적화
- 구매 이력 분석 및 통계 시각화

---

## Screenshots

프로젝트 실행 화면, ROI 감지 화면, 장바구니 UI 이미지를 여기에 추가하면 좋습니다.

```md
![People Tracking](./assets/people_tracking.png)
![Snack Detection](./assets/snack_detection.png)
![Cart UI](./assets/cart_ui.png)
```

---

## Author

**Insu Lee**  
Smart inventory / people tracking demo using YOLO, OpenCV, Tkinter, and Firebase RTDB.
>>>>>>> 2191162 (readme 수정)
