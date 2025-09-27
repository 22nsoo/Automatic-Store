# your_model_pkg/tinyv3_build.py
import os
import torch
import torch.nn as nn

# 우선순위:
#  A) ultralytics/yolov3 (legacy) - Darknet(cfg)
#  B) pytorchyolo (eriklindernoren/PyTorch-YOLOv3 계열) - Darknet(cfg)
#  C) fallback: TinyV3Stub (Conv1+BN만 있는 안전 스텁)

_DEFAULT_CFG = os.environ.get("TINYV3_CFG", "cfg/yolov3-tiny.cfg")

def _try_ultralytics_legacy(cfg_path: str):
    """
    ultralytics/yolov3 레거시 레포 구조:
      models/common.py: Darknet
      (pip 패키지 'yolov3'가 아니라 깃 레포 소스 경로에 있어야 함)
    """
    try:
        from models.common import Darknet  # ultralytics/yolov3 레거시
        m = Darknet(cfg_path)
        return m
    except Exception:
        return None

def _try_pytorchyolo(cfg_path: str):
    """
    pytorchyolo 레포 구조:
      from models import Darknet
    """
    try:
        from models import Darknet  # eriklindernoren/PyTorch-YOLOv3 계열
        m = Darknet(cfg_path)
        return m
    except Exception:
        return None

class TinyV3Stub(nn.Module):
    """
    최소 스텁: Conv1(+BN)만 제공 (커널 3x3, stride=1, pad=1).
    make_conv1_bins.py는 '첫 Conv2d(+BatchNorm2d)'만 필요하므로 충분.
    """
    def __init__(self, cin=3, cout=16, k=3, s=1, p=1, bias=True, bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn1 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.rest = nn.Identity()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return self.rest(x)

def build_model(cfg_path: str = None):
    """
    외부에서 importlib로 호출되는 엔트리포인트.
    우선 tiny-yolov3 cfg 기반 Darknet을 시도하고, 실패 시 스텁으로 대체.
    """
    cfg = cfg_path or _DEFAULT_CFG

    # A) ultralytics/yolov3 레거시
    m = _try_ultralytics_legacy(cfg)
    if m is not None:
        m.eval()
        return m

    # B) pytorchyolo 계열
    m = _try_pytorchyolo(cfg)
    if m is not None:
        m.eval()
        return m

    # C) fallback: Conv1만 있는 안전 스텁
    #  - Conv1: 3x3, stride=1, pad=1 → 레퍼런스/평탄화 코드와 호환
    stub = TinyV3Stub(cin=3, cout=16, k=3, s=1, p=1, bias=True, bn=True)
    stub.eval()
    return stub
