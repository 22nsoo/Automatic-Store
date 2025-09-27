#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_compare_out.py
- 보드 출력(out.bin, int32) vs 골든(golden_acc.bin, int32) 비교
- scales_conv1.npz의 out_shape로 자동 reshape (없으면 --shape로 수동 지정)

사용 예:
  python 02_compare_out.py \
    --out bins/out.bin \
    --golden bins/golden_acc.bin \
    --scales bins/scales_conv1.npz \
    --abs_tol 1 \
    --rmse_tol 0.1
"""

import os, argparse, numpy as np

def load_shape_from_scales(scales_path: str):
    z = np.load(scales_path, allow_pickle=True)
    if 'out_shape' in z:
        return tuple(int(x) for x in z['out_shape'])
    raise ValueError("scales npz에 'out_shape'가 없습니다.")

def parse_shape(s: str):
    # "1,16,416,416" 형태
    parts = [p.strip() for p in s.split(',')]
    return tuple(int(p) for p in parts)

def load_bin_i32(path: str):
    return np.fromfile(path, dtype=np.int32)

def summarize_diff(a: np.ndarray, b: np.ndarray, abs_tol: int = 0, topk: int = 10):
    diff = (a.astype(np.int64) - b.astype(np.int64))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae  = float(np.mean(np.abs(diff)))
    mx   = int(np.max(np.abs(diff)))
    idx_max = int(np.argmax(np.abs(diff)))
    flat = a.size

    # 상위 차이 인덱스들
    if flat > 0:
        order = np.argsort(-np.abs(diff))[:topk]
        top_list = [(int(i), int(a.flat[i]), int(b.flat[i]), int(diff.flat[i])) for i in order]
    else:
        top_list = []

    passed = (mx <= abs_tol)
    return {
        "rmse": rmse,
        "mae": mae,
        "max_abs": mx,
        "idx_max": idx_max,
        "topk": top_list,
        "passed": passed
    }

def channel_stats(a: np.ndarray, b: np.ndarray):
    # a,b: [1,Co,H,W] or [Co,H,W] 형태 지원
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
        b = b[0]
    Co = a.shape[0]
    stats = []
    for co in range(Co):
        d = (a[co].astype(np.int64) - b[co].astype(np.int64))
        rmse = float(np.sqrt(np.mean(d*d)))
        mx   = int(np.max(np.abs(d)))
        stats.append((co, rmse, mx))
    # RMSE 내림차순 상위 5개만 리턴
    stats.sort(key=lambda x: -x[1])
    return stats[:min(5, Co)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",    required=True, help="보드 출력 out.bin (int32)")
    ap.add_argument("--golden", required=True, help="PC에서 만든 golden_acc.bin (int32)")
    ap.add_argument("--scales", default=None,  help="scales_conv1.npz (out_shape 자동 로드용)")
    ap.add_argument("--shape",  default=None,  help="수동 지정 shape. 예: 1,16,416,416")
    ap.add_argument("--abs_tol", type=int, default=1, help="최대 절대오차 허용 (정수 LSB)")
    ap.add_argument("--rmse_tol", type=float, default=None, help="RMSE 허용(선택)")
    ap.add_argument("--print_topk", type=int, default=10, help="상위 오차 항목 출력 개수")
    args = ap.parse_args()

    # 1) 데이터 로드
    out_arr    = load_bin_i32(args.out)
    golden_arr = load_bin_i32(args.golden)

    # 2) shape 결정
    if args.scales:
        shape = load_shape_from_scales(args.scales)
    elif args.shape:
        shape = parse_shape(args.shape)
    else:
        raise SystemExit("shape 정보를 --scales 또는 --shape로 제공하세요.")

    # 3) 길이 체크 후 reshape
    need = int(np.prod(shape))
    if out_arr.size != need or golden_arr.size != need:
        raise SystemExit(f"크기 불일치: 필요 {need}개, out={out_arr.size}, golden={golden_arr.size}. "
                         f"--shape 혹은 RTL 출력 순서를 점검하세요.")
    out_v    = out_arr.reshape(shape)
    golden_v = golden_arr.reshape(shape)

    # 4) 전체 비교
    res = summarize_diff(out_v, golden_v, abs_tol=args.abs_tol, topk=args.print_topk)

    print("==== Compare out.bin vs golden_acc.bin ====")
    print(f"Shape        : {shape}")
    print(f"RMSE         : {res['rmse']:.6f}")
    print(f"MAE          : {res['mae']:.6f}")
    print(f"MaxAbs       : {res['max_abs']} (idx={res['idx_max']})")
    print(f"AbsTol(pass) : <= {args.abs_tol} -> {'PASS' if res['passed'] else 'FAIL'}")

    if args.rmse_tol is not None:
        print(f"RMS Tolerance: <= {args.rmse_tol} -> {'PASS' if res['rmse'] <= args.rmse_tol else 'FAIL'}")

    print(f"\nTop-{args.print_topk} mismatches (flat_index, out, golden, diff):")
    for i, o, g, d in res['topk']:
        print(f"  {i:>9d}: {o:>8d} vs {g:>8d} (diff {d:+d})")

    # 5) 채널별 통계(선택: 4D 또는 3D일 때만)
    if len(shape) == 4 or (len(shape) == 3):
        try:
            top_channels = channel_stats(out_v, golden_v)
            print("\n[채널별 통계 상위 5개]  (co, rmse, max_abs)")
            for co, rmse, mx in top_channels:
                print(f"  co={co:>3d}: rmse={rmse:.6f}, max_abs={mx}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
