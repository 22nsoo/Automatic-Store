# save_tinyv3_pth.py
import torch, sys
CFG = r'cfg\yolov3-tiny.cfg'       # tiny cfg 경로
WTS = r'weights\yolov3-tiny.weights' # darknet .weights 경로
OUT = r'weights\tinyv3.pth'          # 출력 .pth

try:
    # 우선순위 1: ultralytics yolov3 레거시
    from models.common import Darknet
except Exception:
    # 우선순위 2: pytorchyolo
    from models import Darknet

m = Darknet(CFG)
# ↓ Darknet 로더들은 보통 load_darknet_weights 지원
m.load_darknet_weights(WTS)
m.eval()

# (A) 그대로 전체 state_dict 저장
torch.save(m.state_dict(), OUT)

# (B) 또는 make_conv1_bins.py가 첫 Conv만 쓰므로,
#     Conv1(+BN)만 뽑아 최소 state_dict로 저장하려면(선택):
# sd = m.state_dict()
# keep = {k:v for k,v in sd.items() if k.startswith('module_list.0.') or k.startswith('module_list.1.')}
# torch.save(keep, OUT)

print('saved:', OUT)
