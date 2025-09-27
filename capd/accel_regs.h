#pragma once
/**************************************************************
 * accel_regs.h — Conv1 RTL 가속기 AXI-Lite 레지스터 정의
 * 맞춤 포인트:
 *  - ACCEL_BASE: Address Editor에서 지정한 Base Address로 맞추기
 *  - 오프셋/필드가 RTL과 다르면 아래 매크로를 수정
 **************************************************************/

#include <stdint.h>
#include "xil_io.h"   /* Xil_In32 / Xil_Out32 */

/* ---------- Base Address ---------- */
/* 필요하면 컴파일 옵션으로 -DACCEL_BASE=0x43C10000 지정 */
#ifndef ACCEL_BASE
#define ACCEL_BASE      0x43C00000U
#endif

/* ---------- Register Map (Offsets) ---------- */
/* 아래 오프셋은 예시이며, 네 RTL과 반드시 일치해야 함 */
#define REG_CTRL        0x00  /* [0]START, [1]DONE, [2]IDLE, [3]IRQ_EN */
#define REG_STATUS      0x04  /* (선택) 확장 상태 */
#define REG_IN_W        0x10  /* 입력 너비  */
#define REG_IN_H        0x14  /* 입력 높이  */
#define REG_IN_C        0x18  /* 입력 채널  */
#define REG_PAD_STR     0x1C  /* [31:16]=PAD, [15:0]=STRIDE */
#define REG_SCALE_IN    0x20  /* 입력 스케일(Q0.24 등) */
#define REG_MISC        0x24  /* (선택) 플래그/옵션 필드 */
#define REG_USER0       0x28  /* (선택) 사용자 정의 */
#define REG_USER1       0x2C  /* (선택) 사용자 정의 */

/* 필요하면 채널별 scale_w/bias용 베이스/길이 등도 정의
   예: scale_w 테이블을 스트림으로 보내지 않고 AXI-Lite로 쓰는 구조라면:
   #define REG_SW_BASE  0x40  // 테이블 시작 주소
   #define REG_SW_LEN   0x44  // 항목 개수(Cout)
*/

/* ---------- CTRL Bits ---------- */
#define CTRL_START      (1u << 0)
#define CTRL_DONE       (1u << 1)
#define CTRL_IDLE       (1u << 2)
#define CTRL_IRQ_EN     (1u << 3)

/* ---------- Field pack/unpack helpers ---------- */
#define PACK_PAD_STR(pad, stride)   ((((uint32_t)(pad)) << 16) | ((uint32_t)(stride)))
#define UNPACK_PAD(x)               ( (uint16_t)(((x) >> 16) & 0xFFFF) )
#define UNPACK_STRIDE(x)            ( (uint16_t)((x) & 0xFFFF) )

/* ---------- Q-format helpers ---------- */
/* Q0.24: 0<=s 가정. 필요 시 부호/범위 체크 추가 */
static inline uint32_t to_q024(float s) {
    if (s < 0.0f) s = 0.0f;
    double v = (double)s * (1u << 24);
    if (v > 4294967295.0) v = 4294967295.0;
    return (uint32_t)(v + 0.5);
}

/* 일반화된 Qm.n 변환(부호 없는 버전). n은 0~30 권장 */
static inline uint32_t to_qn_u(float s, unsigned n) {
    if (s < 0.0f) s = 0.0f;
    double v = (double)s * (1u << n);
    if (v > 4294967295.0) v = 4294967295.0;
    return (uint32_t)(v + 0.5);
}

/* ---------- MMIO helpers ---------- */
static inline void accel_write(uint32_t offset, uint32_t value) {
    Xil_Out32(ACCEL_BASE + offset, value);
}

static inline uint32_t accel_read(uint32_t offset) {
    return Xil_In32(ACCEL_BASE + offset);
}

/* ---------- Convenience setters ---------- */
static inline void accel_set_input_shape(uint32_t w, uint32_t h, uint32_t c) {
    accel_write(REG_IN_W, w);
    accel_write(REG_IN_H, h);
    accel_write(REG_IN_C, c);
}

static inline void accel_set_pad_stride(uint16_t pad, uint16_t stride) {
    accel_write(REG_PAD_STR, PACK_PAD_STR(pad, stride));
}

static inline void accel_set_scale_in_q024(float scale_in) {
    accel_write(REG_SCALE_IN, to_q024(scale_in));
}

static inline void accel_start(void) {
    accel_write(REG_CTRL, CTRL_START);
}

static inline uint32_t accel_status(void) {
    return accel_read(REG_CTRL);
}

static inline int accel_is_done(void) {
    return (accel_read(REG_CTRL) & CTRL_DONE) != 0;
}

static inline int accel_is_idle(void) {
    return (accel_read(REG_CTRL) & CTRL_IDLE) != 0;
}
