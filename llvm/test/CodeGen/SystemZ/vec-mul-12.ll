; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; Test high-part vector multiplication on z17
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z17 | FileCheck %s

; Test a v2i64 unsigned high-part multiplication.
define <2 x i64> @f1(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmlhg %v24, %v24, %v26
; CHECK-NEXT:    br %r14
  %zext1 = zext <2 x i64> %val1 to <2 x i128>
  %zext2 = zext <2 x i64> %val2 to <2 x i128>
  %mulx = mul <2 x i128> %zext1, %zext2
  %highx = lshr <2 x i128> %mulx, splat(i128 64)
  %high = trunc <2 x i128> %highx to <2 x i64>
  ret <2 x i64> %high
}

; Test a v2i64 signed high-part multiplication.
define <2 x i64> @f2(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmhg %v24, %v24, %v26
; CHECK-NEXT:    br %r14
  %sext1 = sext <2 x i64> %val1 to <2 x i128>
  %sext2 = sext <2 x i64> %val2 to <2 x i128>
  %mulx = mul <2 x i128> %sext1, %sext2
  %highx = lshr <2 x i128> %mulx, splat(i128 64)
  %high = trunc <2 x i128> %highx to <2 x i64>
  ret <2 x i64> %high
}
