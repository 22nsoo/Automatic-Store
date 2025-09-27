#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
packbits.py
- AXI-Stream 전송을 위한 int8 <-> u32/u64 패킹/언패킹
- NCHW 텐서(Conv 입력)와 [Co,Ci,3,3](Conv 가중치)의 평탄화/복원 유틸

패킹 규칙(기본):
- int8 4개 -> uint32 1개 (리틀엔디안, LSB부터 채움)
- 길이가 4의 배수가 아니면 0으로 패딩

입력 평탄화 규칙(기본):
- per-pixel C-fast: (i=0..H-1, j=0..W-1, c=0..C-1) 순서로 1D int8 생성

가중치 평탄화 규칙(기본):
- for co in [0..Co-1]:
    for ci in [0..Ci-1]:
      for ky in [0..Kh-1]:
        for kx in [0..Kw-1]:
          emit W[co,ci,ky,kx]
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


# =========================
# Endianness helper
# =========================
def is_little_endian() -> bool:
    return np.little_endian


# =========================
# Packing / Unpacking
# =========================
def pack_int8_to_u32(arr_int8: np.ndarray) -> np.ndarray:
    """
    4 * int8 -> 1 * uint32 (little-endian)
    길이가 4의 배수가 아니면 0 패딩
    """
    if arr_int8.dtype != np.int8 or arr_int8.ndim != 1:
        raise ValueError("arr_int8 must be 1D np.int8 array")
    pad = (-arr_int8.size) % 4
    if pad:
        arr_int8 = np.concatenate([arr_int8, np.zeros(pad, dtype=np.int8)])
    # 리틀엔디안 환경 가정
    return np.frombuffer(arr_int8.tobytes(), dtype=np.uint32)


def unpack_u32_to_int8(arr_u32: np.ndarray, length: Optional[int] = None) -> np.ndarray:
    """
    1 * uint32 -> 4 * int8 (little-endian)
    length가 주어지면 마지막에 잘라냄(패딩 제거용)
    """
    if arr_u32.dtype != np.uint32 or arr_u32.ndim != 1:
        raise ValueError("arr_u32 must be 1D np.uint32 array")
    out = np.frombuffer(arr_u32.tobytes(), dtype=np.int8)
    if length is not None:
        out = out[:length]
    return out


def pack_int8_to_u64(arr_int8: np.ndarray) -> np.ndarray:
    """
    8 * int8 -> 1 * uint64 (little-endian)
    길이가 8의 배수가 아니면 0 패딩
    """
    if arr_int8.dtype != np.int8 or arr_int8.ndim != 1:
        raise ValueError("arr_int8 must be 1D np.int8 array")
    pad = (-arr_int8.size) % 8
    if pad:
        arr_int8 = np.concatenate([arr_int8, np.zeros(pad, dtype=np.int8)])
    return np.frombuffer(arr_int8.tobytes(), dtype=np.uint64)


def unpack_u64_to_int8(arr_u64: np.ndarray, length: Optional[int] = None) -> np.ndarray:
    """
    1 * uint64 -> 8 * int8 (little-endian)
    length가 주어지면 마지막에 잘라냄
    """
    if arr_u64.dtype != np.uint64 or arr_u64.ndim != 1:
        raise ValueError("arr_u64 must be 1D np.uint64 array")
    out = np.frombuffer(arr_u64.tobytes(), dtype=np.int8)
    if length is not None:
        out = out[:length]
    return out


# =========================
# Flatten / Inflate (Inputs)
# =========================
def flatten_feature_cfast(x_q: np.ndarray) -> np.ndarray:
    """
    입력 특징맵 평탄화 (per-pixel C-fast).
    x_q: [1,C,H,W] int8
    return: 1D int8 of length H*W*C
    순서: for i in H: for j in W: for c in C: emit x_q[0,c,i,j]
    """
    if x_q.dtype != np.int8 or x_q.ndim != 4 or x_q.shape[0] != 1:
        raise ValueError("x_q must be [1,C,H,W] int8")
    _, C, H, W = x_q.shape
    out = np.empty((H * W * C,), dtype=np.int8)
    idx = 0
    for i in range(H):
        for j in range(W):
            out[idx:idx + C] = x_q[0, :, i, j]
            idx += C
    return out


def inflate_feature_cfast(flat: np.ndarray, C: int, H: int, W: int) -> np.ndarray:
    """
    평탄화된 per-pixel C-fast 1D int8 -> [1,C,H,W] 복원
    """
    if flat.dtype != np.int8 or flat.ndim != 1 or flat.size != C * H * W:
        raise ValueError("size mismatch")
    out = np.empty((1, C, H, W), dtype=np.int8)
    idx = 0
    for i in range(H):
        for j in range(W):
            out[0, :, i, j] = flat[idx:idx + C]
            idx += C
    return out


# =========================
# Flatten / Inflate (Weights)
# =========================
def flatten_weight_standard(Wq: np.ndarray) -> np.ndarray:
    """
    가중치 평탄화.
    Wq: [Co,Ci,Kh,Kw] int8
    순서:
      for co in [0..Co-1]:
        for ci in [0..Ci-1]:
          for ky in [0..Kh-1]:
            for kx in [0..Kw-1]:
              emit Wq[co,ci,ky,kx]
    return: 1D int8 of length Co*Ci*Kh*Kw
    """
    if Wq.dtype != np.int8 or Wq.ndim != 4:
        raise ValueError("Wq must be [Co,Ci,Kh,Kw] int8")
    Co, Ci, Kh, Kw = Wq.shape
    out = np.empty((Co * Ci * Kh * Kw,), dtype=np.int8)
    idx = 0
    for co in range(Co):
        for ci in range(Ci):
            k = Wq[co, ci]
            out[idx:idx + Kh * Kw] = k.reshape(-1)
            idx += Kh * Kw
    return out


def inflate_weight_standard(flat: np.ndarray, Co: int, Ci: int, Kh: int, Kw: int) -> np.ndarray:
    """
    평탄화된 가중치 1D int8 -> [Co,Ci,Kh,Kw] 복원
    """
    if flat.dtype != np.int8 or flat.ndim != 1 or flat.size != Co * Ci * Kh * Kw:
        raise ValueError("size mismatch")
    out = np.empty((Co, Ci, Kh, Kw), dtype=np.int8)
    idx = 0
    for co in range(Co):
        for ci in range(Ci):
            out[co, ci] = flat[idx:idx + Kh * Kw].reshape(Kh, Kw)
            idx += Kh * Kw
    return out


# =========================
# File I/O helpers
# =========================
def save_bin_u32(path: str, arr_u32: np.ndarray) -> None:
    if arr_u32.dtype != np.uint32:
        raise ValueError("arr_u32 must be np.uint32")
    arr_u32.tofile(path)


def load_bin_u32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32)


def save_bin_i32(path: str, arr_i32: np.ndarray) -> None:
    if arr_i32.dtype != np.int32:
        raise ValueError("arr_i32 must be np.int32")
    arr_i32.tofile(path)


def load_bin_i32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32)


def save_bin_i8(path: str, arr_i8: np.ndarray) -> None:
    if arr_i8.dtype != np.int8:
        raise ValueError("arr_i8 must be np.int8")
    arr_i8.tofile(path)


def load_bin_i8(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.int8)


# =========================
# Quick self-test (optional)
# =========================
if __name__ == "__main__":
    print("little-endian:", is_little_endian())

    # pack/unpack sanity
    a = np.array([1, -2, 3, -4, 5, -6, 7], dtype=np.int8)
    p32 = pack_int8_to_u32(a)
    b = unpack_u32_to_int8(p32, length=a.size)
    assert np.all(a == b), "u32 pack/unpack mismatch"

    p64 = pack_int8_to_u64(a)
    c = unpack_u64_to_int8(p64, length=a.size)
    assert np.all(a == c), "u64 pack/unpack mismatch"

    # feature flatten/inflate sanity
    x = np.arange(1*3*2*2, dtype=np.int8).reshape(1,3,2,2)
    xf = flatten_feature_cfast(x)
    xr = inflate_feature_cfast(xf, C=3, H=2, W=2)
    assert np.all(x == xr), "feature flatten/inflate mismatch"

    # weight flatten/inflate sanity
    W = np.arange(4*3*3*3, dtype=np.int8).reshape(4,3,3,3)
    Wf = flatten_weight_standard(W)
    Wr = inflate_weight_standard(Wf, Co=4, Ci=3, Kh=3, Kw=3)
    assert np.all(W == Wr), "weight flatten/inflate mismatch"

    print("All tests passed.")
