#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_conv1_bins.py
- Tiny YOLOv3의 Conv1(+BN fold)만 대상으로,
  1) 입력/가중치 INT8 양자화
  2) int8×int8→int32 누적 Conv 레퍼런스 계산(골든)
  3) AXI-Stream용 패킹(bin 파일) 저장
출력:
  - in.bin              : 입력(feature) int8를 4바이트(=4xint8)씩 u32로 패킹
  - w.bin               : 가중치 int8를 4바이트(=4xint8)씩 u32로 패킹
  - golden_acc.bin      : Conv1 누적 결과(int32, BN/bias 반영)
  - scales_conv1.npz    : scale_in(float), scale_w(float[Cout]), meta(JSON 비슷)
사용 예:
  python make_conv1_bins.py --img img.jpg --model_py path.to.module --model_fn build_model --ckpt weights.pth --size 416
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image

import importlib
import torch
import torch.nn as nn


# ---------- 이미지 전처리 (간단 리사이즈; 필요하면 letterbox로 교체) ----------
def load_preprocessed_input(img_path, size=416):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)  # 간단 리사이즈
    x = np.asarray(img).astype(np.float32) / 255.0   # [H,W,3], 0~1
    x = x.transpose(2, 0, 1)                         # [3,H,W] (C,H,W)
    x = np.expand_dims(x, 0)                         # [1,3,H,W]
    return x  # float32


# ---------- 모델 로드 & Conv1 + BN 추출 ----------
def load_model(model_py, model_fn, ckpt=None, device='cpu'):
    """
    model_py: 'package.module' 형식 (importlib로 임포트)
    model_fn: 모듈 내에서 모델을 생성하는 함수명 (예: 'build_model')
    ckpt    : state_dict 또는 torch.save한 경로(선택)
    """
    mod = importlib.import_module(model_py)
    constructor = getattr(mod, model_fn)
    m = constructor()
    m.to(device)
    m.eval()
    if ckpt:
        sd = torch.load(ckpt, map_location=device)
        # state_dict만 저장된 경우와 전체 저장 모두 지원
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m.load_state_dict(sd, strict=False)
    return m


def find_first_conv_bn(model: nn.Module):
    """
    모델에서 '첫 Conv2d'와 바로 뒤에 오는 BatchNorm2d(있으면) 자동 탐색.
    Darknet식 module_list가 아니어도 동작하도록 순회 탐색.
    """
    conv = None
    bn = None
    # 순서 보장을 위해 modules() 대신 named_modules()와 자손 순회
    prev = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and conv is None:
            conv = m
            prev = 'conv'
        elif isinstance(m, nn.BatchNorm2d) and prev == 'conv' and bn is None:
            bn = m
            break
        else:
            prev = None
    return conv, bn


# ---------- BN folding ----------
def bn_fold(conv: nn.Conv2d, bn: nn.BatchNorm2d | None):
    """
    Conv W,b와 BN(γ,β,μ,σ)로 BN fold 수행.
    반환: Wf[Co,Ci,K,K] float32, bf[Co] float32
    """
    W = conv.weight.detach().cpu().numpy().astype(np.float32)  # [Co,Ci,K,K]
    if conv.bias is not None:
        b = conv.bias.detach().cpu().numpy().astype(np.float32)
    else:
        b = np.zeros((W.shape[0],), dtype=np.float32)

    if bn is None:
        return W, b

    gamma = bn.weight.detach().cpu().numpy().astype(np.float32)
    beta  = bn.bias.detach().cpu().numpy().astype(np.float32)
    mu    = bn.running_mean.detach().cpu().numpy().astype(np.float32)
    var   = bn.running_var.detach().cpu().numpy().astype(np.float32)
    eps   = float(bn.eps)

    sigma = np.sqrt(var + eps)  # [Co]
    # Wf[Co,Ci,K,K] = W * (gamma/sigma) (out-channel 별 스케일)
    scale = (gamma / sigma).reshape(-1, 1, 1, 1)     # [Co,1,1,1]
    Wf = W * scale
    # bf[Co] = b - mu*(gamma/sigma) + beta
    bf = b - mu * (gamma / sigma) + beta
    return Wf, bf


# ---------- INT8 양자화(대칭) ----------
def quantize_input_int8_symmetric(x_f):
    """
    x_f: [1,C,H,W] float32
    반환: x_q int8, scale_in float
    """
    max_abs = float(np.max(np.abs(x_f)))
    scale_in = max_abs / 127.0 + 1e-12
    x_q = np.clip(np.round(x_f / scale_in), -128, 127).astype(np.int8)
    return x_q, scale_in


def quantize_weight_int8_perchannel(Wf):
    """
    Wf: [Co,Ci,K,K] float32
    반환: Wq int8 same shape, scale_w float[Co]
    """
    Co = Wf.shape[0]
    Wq = np.zeros_like(Wf, dtype=np.int8)
    scale_w = np.zeros((Co,), dtype=np.float32)
    for co in range(Co):
        w = Wf[co]
        s = float(np.max(np.abs(w)))
        s = s / 127.0 + 1e-12
        scale_w[co] = s
        Wq[co] = np.clip(np.round(w / s), -128, 127).astype(np.int8)
    return Wq, scale_w


# ---------- Conv 정수 누적 레퍼런스 ----------
def conv2d_int8_reference(x_q, Wq, bf, scale_in, scale_w, stride=1, pad=1):
    """
    x_q: [1,Ci,H,W] int8
    Wq : [Co,Ci,Kh,Kw] int8 (여기서는 3x3 가정)
    bf : [Co] float32 (BN fold된 bias)
    scale_in: float
    scale_w : [Co] float32
    반환: y_acc [1,Co,H_out,W_out] int32  (누적 도메인)
    """
    N, Ci, H, W = x_q.shape
    Co, Ci2, Kh, Kw = Wq.shape
    assert Ci == Ci2 and Kh == 3 and Kw == 3
    assert N == 1

    # 출력 크기 (same padding 가정)
    Hout = (H + 2 * pad - Kh) // stride + 1
    Wout = (W + 2 * pad - Kw) // stride + 1

    xpad = np.pad(x_q.astype(np.int32), ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    y_acc = np.zeros((1, Co, Hout, Wout), dtype=np.int32)

    for co in range(Co):
        sw = scale_w[co]
        bias_int = int(np.round(bf[co] / (scale_in * sw)))  # 누적 도메인으로 변환
        for i in range(Hout):
            for j in range(Wout):
                acc = 0
                ii = i * stride
                jj = j * stride
                region = xpad[0, :, ii:ii+3, jj:jj+3].astype(np.int32)    # [Ci,3,3]
                kernel = Wq[co].astype(np.int32)                           # [Ci,3,3]
                acc += int(np.sum(region * kernel))
                acc += bias_int
                y_acc[0, co, i, j] = acc
    return y_acc


# ---------- 패킹(채널-우선: 한 픽셀의 C 채널을 연속 배치) ----------
def flatten_feature_cfast(x_q):
    """
    x_q: [1,C,H,W] int8
    반환: 1D int8 (픽셀 단위로 C가 먼저: (i,j)마다 c=0..C-1 순서)
    """
    _, C, H, W = x_q.shape
    out = np.empty((H*W*C,), dtype=np.int8)
    idx = 0
    for i in range(H):
        for j in range(W):
            out[idx:idx+C] = x_q[0, :, i, j]
            idx += C
    return out


def flatten_weight_standard(Wq):
    """
    Wq: [Co,Ci,3,3] int8
    반환: 1D int8 (co fastest? → 여기서는 co가 바깥)
    순서: for co in Co: for ci in Ci: for ky in 0..2: for kx in 0..2
    (RTL에서도 동일 순서를 가정)
    """
    Co, Ci, Kh, Kw = Wq.shape
    out = np.empty((Co*Ci*Kh*Kw,), dtype=np.int8)
    idx = 0
    for co in range(Co):
        for ci in range(Ci):
            k = Wq[co, ci]
            out[idx:idx+Kh*Kw] = k.reshape(-1)
            idx += Kh*Kw
    assert idx == out.size
    return out


def pack_int8_to_u32(arr_int8: np.ndarray) -> np.ndarray:
    """
    4개의 int8 -> 1개의 uint32 (LSB부터)
    길이가 4의 배수가 아니면 0으로 패딩
    """
    assert arr_int8.dtype == np.int8 and arr_int8.ndim == 1
    pad = (-arr_int8.size) % 4
    if pad:
        arr_int8 = np.concatenate([arr_int8, np.zeros(pad, dtype=np.int8)])
    packed = np.frombuffer(arr_int8.tobytes(), dtype=np.uint32)
    return packed


# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True, help='입력 이미지 경로 (jpg/png)')
    ap.add_argument('--model_py', required=True, help='모델 모듈 경로 (e.g., pkg.mod)')
    ap.add_argument('--model_fn', required=True, help='모델 생성 함수명 (e.g., build_model)')
    ap.add_argument('--ckpt', default=None, help='가중치 경로(.pth/.pt), 선택')
    ap.add_argument('--size', type=int, default=416, help='입력 해상도(정사각)')
    ap.add_argument('--outdir', default='.', help='결과 파일 저장 폴더')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) 입력 로드
    x_f = load_preprocessed_input(args.img, size=args.size)  # [1,3,H,W] float32
    N, Cin, H, W = x_f.shape
    assert N == 1 and Cin == 3, "Conv1 입력은 보통 3채널(RGB) 기준입니다."

    # 2) 모델 로드 & Conv1+BN 추출 & BN fold
    model = load_model(args.model_py, args.model_fn, args.ckpt, device='cpu')
    conv, bn = find_first_conv_bn(model)
    assert isinstance(conv, nn.Conv2d), "모델에서 Conv2d를 찾지 못했습니다."
    Wf, bf = bn_fold(conv, bn)  # [Co,Ci,3,3], [Co]
    Co = Wf.shape[0]

    # 3) INT8 양자화
    x_q, scale_in = quantize_input_int8_symmetric(x_f)
    Wq, scale_w = quantize_weight_int8_perchannel(Wf)

    # 4) Conv 정수 누적 골든
    # stride/pad는 Conv1 설정을 따름
    stride = conv.stride[0] if isinstance(conv.stride, tuple) else int(conv.stride)
    pad    = conv.padding[0] if isinstance(conv.padding, tuple) else int(conv.padding)
    y_acc = conv2d_int8_reference(x_q, Wq, bf, scale_in, scale_w, stride=stride, pad=pad)  # [1,Co,Hout,Wout]

    # 5) 패킹 & 저장
    # 입력: 픽셀별로 C-fast 순서로 평탄화 후 pack4
    x_flat  = flatten_feature_cfast(x_q)         # int8 1D
    x_pack  = pack_int8_to_u32(x_flat)           # u32 1D
    # 가중치: (co,ci,ky,kx) 순으로 평탄화 후 pack4
    w_flat  = flatten_weight_standard(Wq)        # int8 1D
    w_pack  = pack_int8_to_u32(w_flat)           # u32 1D

    in_bin   = os.path.join(args.outdir, 'in.bin')
    w_bin    = os.path.join(args.outdir, 'w.bin')
    golden   = os.path.join(args.outdir, 'golden_acc.bin')
    scales   = os.path.join(args.outdir, 'scales_conv1.npz')
    meta_js  = os.path.join(args.outdir, 'pack_order.json')

    x_pack.tofile(in_bin)
    w_pack.tofile(w_bin)
    y_acc.astype(np.int32).tofile(golden)
    np.savez(scales, scale_in=scale_in, scale_w=scale_w.astype(np.float32),
             stride=stride, pad=pad, in_shape=x_q.shape, w_shape=Wq.shape, out_shape=y_acc.shape)

    # 패킹/전달 순서(PL과 일치 필수)를 json으로 남김
    meta = {
        "input_order": "per-pixel C-fast (i=0..H-1, j=0..W-1, then c=0..C-1), int8, pack 4×int8 -> u32",
        "weight_order": "for co in [0..Co-1], for ci in [0..Ci-1], for ky in [0..2], for kx in [0..2]; int8, pack 4×int8 -> u32",
        "dtype_packed": "uint32 little-endian",
        "conv1": {
            "Cin": int(Cin),
            "Cout": int(Co),
            "kernel": [3,3],
            "stride": int(stride),
            "pad": int(pad),
        }
    }
    with open(meta_js, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 로그
    Hout, Wout = y_acc.shape[2], y_acc.shape[3]
    print("== DONE ==")
    print(f"Input : {x_f.shape} -> x_q int8, packed to {in_bin}, bytes={x_pack.nbytes}")
    print(f"Weights: {Wf.shape} -> Wq int8, packed to {w_bin}, bytes={w_pack.nbytes}")
    print(f"Golden: [1,{Co},{Hout},{Wout}] int32 -> {golden}, bytes={y_acc.size*4}")
    print(f"Scales: scale_in(float), scale_w(float[{Co}]) -> {scales}")
    print(f"Meta  : pack/ordering -> {meta_js}")


if __name__ == '__main__':
    main()
