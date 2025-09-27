/**************************************************************
 * dma_utils.c  — Zynq AXI DMA (Simple Mode) 유틸
 * - 초기화, 송신(MM2S), 수신(S2MM), 바쁜지 체크, 완료 대기
 * - DCache flush / invalidate 헬퍼 포함
 *
 * 사용 전제:
 *  - Vivado에서 AXI DMA를 Scatter-Gather 비활성(Simple)로 구성
 *  - 같은 클럭 도메인(FCLK0)에서 DMA/가속기 동작
 *
 * 함께 사용할 헤더 예시(dma_utils.h):
 * ------------------------------------------------------------
 * #pragma once
 * #include "xaxidma.h"
 * #include "xil_types.h"
 * int  dma_init(u16 DevId, XAxiDma *pDma);
 * int  dma_send(XAxiDma *pDma, const void *buf, u32 bytes);
 * int  dma_recv(XAxiDma *pDma,       void *buf, u32 bytes);
 * int  dma_busy_tx(XAxiDma *pDma);
 * int  dma_busy_rx(XAxiDma *pDma);
 * int  dma_wait_tx(XAxiDma *pDma);
 * int  dma_wait_rx(XAxiDma *pDma);
 * void dma_cache_flush(const void *buf, u32 bytes);
 * void dma_cache_invalidate(void *buf, u32 bytes);
 * ------------------------------------------------------------
 **************************************************************/

#include "dma_utils.h"
#include "xparameters.h"
#include "xil_cache.h"
#include "xil_printf.h"

/* -----------------------------------------------------------
 * 초기화
 * ---------------------------------------------------------*/
int dma_init(u16 DevId, XAxiDma *pDma)
{
    XAxiDma_Config *Cfg = XAxiDma_LookupConfig(DevId);
    if (!Cfg) {
        xil_printf("[dma] LookupConfig failed (DevId=%u)\r\n", DevId);
        return XST_FAILURE;
    }
    int st = XAxiDma_CfgInitialize(pDma, Cfg);
    if (st != XST_SUCCESS) {
        xil_printf("[dma] CfgInitialize failed: %d\r\n", st);
        return st;
    }
    if (XAxiDma_HasSg(pDma)) {
        xil_printf("[dma] ERROR: DMA is in SG mode. Re-generate IP as Simple Mode.\r\n");
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

/* -----------------------------------------------------------
 * 송신(MM2S): DDR -> AXI-Stream
 *  - buf는 64B 정렬 권장
 *  - bytes는 실제 전송할 바이트 수 (프레임 끝에서 TLAST)
 * ---------------------------------------------------------*/
int dma_send(XAxiDma *pDma, const void *buf, u32 bytes)
{
    if (bytes == 0) return XST_SUCCESS;
    /* Cache flush: CPU가 쓴 내용을 DDR에 반영 */
    Xil_DCacheFlushRange((INTPTR)buf, bytes);

    int st = XAxiDma_SimpleTransfer(pDma, (UINTPTR)buf, bytes, XAXIDMA_DMA_TO_DEVICE);
    if (st != XST_SUCCESS) {
        xil_printf("[dma] MM2S start failed: %d\r\n", st);
        return st;
    }
    return XST_SUCCESS;
}

/* -----------------------------------------------------------
 * 수신(S2MM): AXI-Stream -> DDR
 *  - buf는 64B 정렬 권장
 *  - bytes는 기대하는 수신 바이트 수 (프레임 길이와 일치)
 * ---------------------------------------------------------*/
int dma_recv(XAxiDma *pDma, void *buf, u32 bytes)
{
    if (bytes == 0) return XST_SUCCESS;
    /* Cache invalidate는 완료 후에 하지만,
       일부 BSP에선 시작 전에 미리 invalidate 해두기도 함 */
    Xil_DCacheInvalidateRange((INTPTR)buf, bytes);

    int st = XAxiDma_SimpleTransfer(pDma, (UINTPTR)buf, bytes, XAXIDMA_DEVICE_TO_DMA);
    if (st != XST_SUCCESS) {
        xil_printf("[dma] S2MM start failed: %d\r\n", st);
        return st;
    }
    return XST_SUCCESS;
}

/* -----------------------------------------------------------
 * Busy / Wait
 * ---------------------------------------------------------*/
int dma_busy_tx(XAxiDma *pDma)
{
    return XAxiDma_Busy(pDma, XAXIDMA_DMA_TO_DEVICE);
}

int dma_busy_rx(XAxiDma *pDma)
{
    return XAxiDma_Busy(pDma, XAXIDMA_DEVICE_TO_DMA);
}

int dma_wait_tx(XAxiDma *pDma)
{
    while (XAxiDma_Busy(pDma, XAXIDMA_DMA_TO_DEVICE)) { /* spin */ }
    return XST_SUCCESS;
}

int dma_wait_rx(XAxiDma *pDma)
{
    while (XAxiDma_Busy(pDma, XAXIDMA_DEVICE_TO_DMA)) { /* spin */ }
    return XST_SUCCESS;
}

/* -----------------------------------------------------------
 * Cache helpers
 * ---------------------------------------------------------*/
void dma_cache_flush(const void *buf, u32 bytes)
{
    if (bytes) Xil_DCacheFlushRange((INTPTR)buf, bytes);
}

void dma_cache_invalidate(void *buf, u32 bytes)
{
    if (bytes) Xil_DCacheInvalidateRange((INTPTR)buf, bytes);
}
