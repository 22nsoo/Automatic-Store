/**************************************************************
 * app_conv1.c  (Zynq Bare-metal)
 * - SD카드에서 in.bin, w.bin 읽기
 * - AXI DMA MM2S: (1) weights, (2) input  순서로 전송
 * - AXI DMA S2MM: 출력(int32 누적) 수신 → out.bin 저장
 * - 가속기 AXI-Lite 레지스터로 파라미터 설정 후 START
 *
 * 준비물:
 *  - Vivado BD: Zynq PS + AXI DMA(단일) + conv1_accel
 *  - Vitis BSP: xilffs(FatFs) 포함, SD 드라이버 활성화
 **************************************************************/

#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xaxidma.h"

/* --- FATFS/SD --- */
#include "ff.h"
#include "xsdps.h"

/* ===================== 사용자 환경 정의 ===================== */

/* AXI DMA Device ID (xparameters.h에서 확인) */
#ifndef DMA_DEV_ID
#define DMA_DEV_ID      XPAR_AXIDMA_0_DEVICE_ID
#endif

/* 가속기 AXI-Lite Base Address (Address Editor에서 할당한 값) */
#ifndef ACCEL_BASE
#define ACCEL_BASE      0x43C00000U
#endif

/* 가속기 레지스터 오프셋(예시: 필요에 맞게 수정 가능) */
#define REG_CTRL        0x00  /* [0]START, [1]DONE, [2]IDLE, [3]IRQ_EN */
#define REG_IN_W        0x10
#define REG_IN_H        0x14
#define REG_IN_C        0x18
#define REG_PAD_STR     0x1C  /* [31:16]=pad, [15:0]=stride */
#define REG_SCALE_IN    0x20  /* 옵션: Q0.24 고정소수점 스케일 */
#define REG_MISC        0x24  /* 옵션(예: flags) */

/* CTRL 비트 */
#define CTRL_START      (1u<<0)
#define CTRL_DONE       (1u<<1)
#define CTRL_IDLE       (1u<<2)

/* 입력/가중치/출력 파일명 */
#define IN_BIN_PATH     "in.bin"
#define W_BIN_PATH      "w.bin"
#define OUT_BIN_PATH    "out.bin"

/* 버퍼 최대 크기 (바이트) — 보수적으로 넉넉히 잡기
   416x416x3 int8 packed→u32 = ceil(519,168B) ≈ 520KB
   Conv1 출력(예: Cout=16, 416x416, int32) ≈ 10.56MB
*/
#define MAX_IN_BYTES    (1*1024*1024)     /* 1MB */
#define MAX_W_BYTES     (256*1024)        /* 256KB (Conv1 weights는 수KB 수준) */
#define MAX_OUT_BYTES   (16*1024*1024)    /* 16MB */

/* 클럭/리셋은 Vivado BD에서 PS FCLK0로 통일되었다고 가정 */

/* ===================== 전역 ===================== */
static XAxiDma AxiDma;

/* 64B 정렬 버퍼 (DCache 라인 정렬) */
static u8  in_buf [MAX_IN_BYTES]  __attribute__((aligned(64)));
static u8  w_buf  [MAX_W_BYTES]   __attribute__((aligned(64)));
static u8  out_buf[MAX_OUT_BYTES] __attribute__((aligned(64)));

/* FATFS 핸들 */
static FATFS fatfs;

/* ===================== 유틸 ===================== */
static inline void reg_write(u32 off, u32 val) { Xil_Out32(ACCEL_BASE + off, val); }
static inline u32  reg_read (u32 off) { return Xil_In32(ACCEL_BASE + off); }

/* Q0.24 고정소수점 변환(필요 시 사용) */
static inline u32 to_q024(float s) {
    if (s < 0) s = 0;
    double v = (double)s * (1u<<24);
    if (v > 4294967295.0) v = 4294967295.0;
    return (u32)(v + 0.5);
}

/* DMA 초기화 */
static int dma_init(u16 DevId) {
    XAxiDma_Config *Cfg = XAxiDma_LookupConfig(DevId);
    if (!Cfg) { xil_printf("DMA lookup failed\r\n"); return XST_FAILURE; }
    int st = XAxiDma_CfgInitialize(&AxiDma, Cfg);
    if (st != XST_SUCCESS) { xil_printf("DMA init failed %d\r\n", st); return st; }
    /* SG 모드가 아니어야 함 */
    if (XAxiDma_HasSg(&AxiDma)) {
        xil_printf("DMA in SG mode — set to Simple DMA in Vivado\r\n");
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

/* SD카드 마운트 */
static int sd_mount(void) {
    FRESULT fr;
    fr = f_mount(&fatfs, "0:/", 1);
    if (fr != FR_OK) { xil_printf("f_mount failed: %d\r\n", fr); return XST_FAILURE; }
    return XST_SUCCESS;
}

/* 파일 전체 읽기 → buf에 저장, *bytes에 실제 바이트 수 기록 */
static int read_file_to_buf(const char *path, u8 *buf, UINT max_bytes, UINT *bytes) {
    FRESULT fr;
    FIL fil;
    UINT br = 0;
    fr = f_open(&fil, path, FA_READ);
    if (fr != FR_OK) { xil_printf("open %s failed: %d\r\n", path, fr); return XST_FAILURE; }
    /* 파일 크기 확인 */
    FSIZE_t fsz = f_size(&fil);
    if (fsz > max_bytes) {
        xil_printf("file %s too large (%lu bytes > %u)\r\n", path, (u32)fsz, (u32)max_bytes);
        f_close(&fil);
        return XST_FAILURE;
    }
    /* 읽기 */
    fr = f_read(&fil, buf, (UINT)fsz, &br);
    f_close(&fil);
    if (fr != FR_OK || br != (UINT)fsz) {
        xil_printf("read %s failed: fr=%d br=%u fsz=%lu\r\n", path, fr, br, (u32)fsz);
        return XST_FAILURE;
    }
    *bytes = br;
    return XST_SUCCESS;
}

/* 버퍼를 파일로 저장 (out.bin) */
static int write_buf_to_file(const char *path, const u8 *buf, UINT bytes) {
    FRESULT fr;
    FIL fil;
    UINT bw = 0;
    fr = f_open(&fil, path, FA_WRITE | FA_CREATE_ALWAYS);
    if (fr != FR_OK) { xil_printf("open %s for write failed: %d\r\n", path, fr); return XST_FAILURE; }
    fr = f_write(&fil, buf, bytes, &bw);
    f_close(&fil);
    if (fr != FR_OK || bw != bytes) {
        xil_printf("write %s failed: fr=%d bw=%u/%u\r\n", path, fr, bw, bytes);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

/* ===================== 메인 ===================== */
int main(void)
{
    xil_printf("\r\n[app_conv1] start\r\n");

    /* 1) SD 마운트 */
    if (sd_mount() != XST_SUCCESS) {
        xil_printf("SD mount failed. Check SD card & BSP(xilffs)\r\n");
        return -1;
    }

    /* 2) in.bin, w.bin 읽기 */
    UINT in_bytes = 0, w_bytes = 0;
    if (read_file_to_buf(IN_BIN_PATH, in_buf, MAX_IN_BYTES, &in_bytes) != XST_SUCCESS) return -2;
    if (read_file_to_buf(W_BIN_PATH , w_buf , MAX_W_BYTES , &w_bytes ) != XST_SUCCESS) return -3;
    xil_printf("Loaded in.bin=%u bytes, w.bin=%u bytes\r\n", (u32)in_bytes, (u32)w_bytes);

    /* 3) DMA 초기화 */
    if (dma_init(DMA_DEV_ID) != XST_SUCCESS) return -4;

    /* 4) (선택) 파라미터 세팅 — 실제 Conv1 파라미터로 수정 */
    /*    크기/패딩/스트라이드는 Python npz(meta)에서 읽어 하드코딩하거나, 나중에 파일로 전달 가능 */
    const u32 IN_W = 416, IN_H = 416, IN_C = 3;
    const u16 STRIDE = 1, PAD = 1;
    /* 입력 스케일을 하드웨어가 필요로 한다면 적절히 설정 (없으면 무시됨) */
    /* scale_in은 float → Q0.24. 여기서는 예시로 1.0 */
    const float scale_in_f = 1.0f;

    reg_write(REG_IN_W, IN_W);
    reg_write(REG_IN_H, IN_H);
    reg_write(REG_IN_C, IN_C);
    reg_write(REG_PAD_STR, ((u32)PAD << 16) | (u32)STRIDE);
    reg_write(REG_SCALE_IN, to_q024(scale_in_f));
    /* 필요시 REG_MISC 등 추가 */

    /* 5) 캐시 플러시 (보내기 전), 수신 버퍼 무효화 */
    Xil_DCacheFlushRange((INTPTR)in_buf,  in_bytes);
    Xil_DCacheFlushRange((INTPTR)w_buf,   w_bytes);
    /* out_buf는 수신 전 invalidate */
    Xil_DCacheInvalidateRange((INTPTR)out_buf, MAX_OUT_BYTES);

    /* 6) S2MM(수신) 먼저 arm — 출력 바이트 수는 “가속기 실제 출력 길이”여야 함
          Conv1 누적 결과(out)는 int32 형태이므로, 바이트 수 = Cout*Hout*Wout*4
          처음엔 대략 값을 알고 넣는 게 좋아(예: Cout=16, Hout=416, Wout=416).
          여기선 보수적으로 out_bytes를 계산해 사용자가 수정하도록 안내.
    */
    /* TODO: 너의 Conv1 실제 출력 크기에 맞게 수정 ↓ */
    u32 Cout = 16;  /* 모델의 Conv1 출력 채널 수 */
    u32 Hout = 416; /* stride=1,pad=1이면 입력과 동일 */
    u32 Wout = 416;
    u32 out_bytes = Cout * Hout * Wout * 4u;
    if (out_bytes > MAX_OUT_BYTES) {
        xil_printf("out_bytes too large (%u)\r\n", (u32)out_bytes);
        return -5;
    }

    int st;
    st = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)out_buf, out_bytes, XAXIDMA_DEVICE_TO_DMA);
    if (st != XST_SUCCESS) { xil_printf("S2MM start failed\r\n"); return -6; }

    /* 7) MM2S로 “가중치 → 입력” 순서 전송
          중요: 네 RTL이 한 개의 AXI-Stream에서
                (1) weight 블록, (2) feature 블록
                의 순서를 **기대**한다고 가정 (TLAST는 각 블록 끝에서 1회)
          만약 가중치와 입력 스트림 포트가 분리돼있다면
          → MM2S IP를 두 개 쓰거나, AXI Stream Switch를 사용해야 함(구성 변경 필요)
    */
    st = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)w_buf,  w_bytes,  XAXIDMA_DMA_TO_DEVICE);
    if (st != XST_SUCCESS) { xil_printf("MM2S(w) start failed\r\n"); return -7; }

    /* MM2S 가중치 전송 완료 대기 */
    while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) { /* spin */ }

    /* 이어서 입력 전송 */
    st = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)in_buf, in_bytes, XAXIDMA_DMA_TO_DEVICE);
    if (st != XST_SUCCESS) { xil_printf("MM2S(in) start failed\r\n"); return -8; }

    /* 8) 가속기 START */
    reg_write(REG_CTRL, CTRL_START);

    /* 9) MM2S 입력 완료 대기 */
    while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) { /* spin */ }

    /* 10) S2MM(수신) 완료 대기 */
    while (XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) { /* spin */ }

    /* (선택) DONE 폴링 */
    u32 ctrl = 0;
    do { ctrl = reg_read(REG_CTRL); } while ((ctrl & CTRL_DONE) == 0);

    /* 11) 수신 데이터 캐시 무효화 후 SD에 저장 */
    Xil_DCacheInvalidateRange((INTPTR)out_buf, out_bytes);
    if (write_buf_to_file(OUT_BIN_PATH, out_buf, out_bytes) != XST_SUCCESS) {
        xil_printf("write out.bin failed\r\n"); return -9;
    }

    xil_printf("OK: out.bin saved (%u bytes)\r\n", (u32)out_bytes);
    xil_printf("[app_conv1] done\r\n");
    return 0;
}
