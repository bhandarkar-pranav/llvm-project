// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @simple_var.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "simple_var", pinned} : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @target_map_single_private() attributes {fir.internal_name = "_QPtarget_map_single_private"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "simple_var"} : (i64) -> !llvm.ptr
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %4, %3 : i32, !llvm.ptr
  %5 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "a"}
  omp.target map_entries(%5 -> %arg0 : !llvm.ptr) private(@simple_var.privatizer %1 -> %arg1 : !llvm.ptr) {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %6 = llvm.mlir.constant(10 : i32) : i32
    %7 = llvm.load %arg0 : !llvm.ptr -> i32
    %8 = llvm.add %7, %6 : i32
    llvm.store %8, %arg1 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @__omp_offloading_fd00
// CHECK-NOT: define {{.*}}
// CHECK-DAG: %[[PRIV_ALLOC:.*]] = alloca i32, i64 1, align 4
// CHECK-DAG: %[[ADD:.*]] = add i32 {{.*}}, 10
// CHECK-DAG: store i32 %[[ADD]], ptr %[[PRIV_ALLOC]], align 4

omp.private {type = private} @n.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x f32 {bindc_name = "n", pinned} : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @target_map_2_privates() attributes {fir.internal_name = "_QPtarget_map_2_privates"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "simple_var"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x f32 {bindc_name = "n"} : (i64) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %6, %5 : i32, !llvm.ptr
  %7 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "a"}
  omp.target map_entries(%7 -> %arg0 : !llvm.ptr) private(@simple_var.privatizer %1 -> %arg1 : !llvm.ptr, @n.privatizer %3 -> %arg2 : !llvm.ptr) {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr):
    %8 = llvm.mlir.constant(1.100000e+01 : f32) : f32
    %9 = llvm.mlir.constant(10 : i32) : i32
    %10 = llvm.load %arg0 : !llvm.ptr -> i32
    %11 = llvm.add %10, %9 : i32
    llvm.store %11, %arg1 : i32, !llvm.ptr
    %12 = llvm.load %arg1 : !llvm.ptr -> i32
    %13 = llvm.sitofp %12 : i32 to f32
    %14 = llvm.fadd %13, %8  {fastmathFlags = #llvm.fastmath<contract>} : f32
    llvm.store %14, %arg2 : f32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK: define internal void @__omp_offloading_fd00
// CHECK-DAG: %[[PRIV_I32_ALLOC:.*]] = alloca i32, i64 1, align 4
// CHECK-DAG: %[[PRIV_FLOAT_ALLOC:.*]] = alloca float, i64 1, align 4
// CHECK-DAG: %[[ADD_I32:.*]] = add i32 {{.*}}, 10
// CHECK-DAG: store i32 %[[ADD_I32]], ptr %[[PRIV_I32_ALLOC]], align 4
// CHECK-DAG: %[[LOAD_I32_AGAIN:.*]] = load i32, ptr %6, align 4
// CHECK-DAG: %[[CAST_TO_FLOAT:.*]] = sitofp i32 %[[LOAD_I32_AGAIN]] to float
// CHECK-DAG: %[[ADD_FLOAT:.*]] = fadd contract float %[[CAST_TO_FLOAT]], 1.100000e+01
// CHECK-DAG: store float %[[ADD_FLOAT]], ptr %[[PRIV_FLOAT_ALLOC]], align 4
      
