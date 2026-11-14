// Tests the rule that decides, for a `do concurrent` loop that is already
// directly inside a user-written omp.target region, whether a
// `teams distribute parallel do` nest may be emitted there (canEmitTeamsInTarget)
// or the pass must fall back to a `parallel do` nest.
//
// Besides the loop, the target body may only contain ops that serve the loop
// (its bounds or values used in its body) or declare variables, plus
// constants. Anything else, even a side-effect-free read, makes the pass fall
// back: code outside the teams region is executed once per team on the GPU,
// and OpenMP does not allow code between `target` and a nested `teams`.
//
// The inputs are derived from the IR that flang produces for
//
//   !$omp target map(tofrom: a) map(to: m)
//   do concurrent (i = 1:100)
//     a(i) = 1
//   end do
//   !$omp end target
//
// with extra ops inserted before the loop; such ops cannot be written in
// Fortran without a statement (which the rule would reject anyway).
//
// When the teams nest is emitted, the code left before omp.teams is only
// constants, shapes and declarations (the loop's bound computations are
// re-evaluated inside omp.parallel and the originals erased).

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=device" %s | FileCheck %s

module attributes {omp.is_target_device = false} {

  func.func @only_loop(%arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<f32> {fir.bindc_name = "m"}) {
    %0 = fir.dummy_scope : !fir.dscope
    %c100 = arith.constant 100 : index
    %1 = fir.shape %c100 : (index) -> !fir.shape<1>
    %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 arg 1 uniq_name("_QFunrelated_readEa") : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
    %3 = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFunrelated_readEi"}>
    %4:2 = hlfir.declare %3 uniq_name("_QFunrelated_readEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFunrelated_readEm") : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %6 = arith.subi %c100, %c1 : index
    %7 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%6 : index) extent(%c100 : index) stride(%c1 : index) start_idx(%c1 : index)
    %8 = omp.map.info var_ptr(%2#1 : !fir.ref<!fir.array<100xf32>>, !fir.array<100xf32>) map_clauses(tofrom) capture(ByRef) bounds(%7) name("a") -> !fir.ref<!fir.array<100xf32>>
    %9 = omp.map.info var_ptr(%5#1 : !fir.ref<f32>, f32) map_clauses(to) capture(ByRef) name("m") -> !fir.ref<f32>
    omp.target kernel_type(generic) map_entries(%8 -> %arg2, %9 -> %arg3 : !fir.ref<!fir.array<100xf32>>, !fir.ref<f32>) {
      %c100_0 = arith.constant 100 : index
      %10 = fir.shape %c100_0 : (index) -> !fir.shape<1>
      %11:2 = hlfir.declare %arg2(%10) uniq_name("_QFunrelated_readEa") : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
      %12:2 = hlfir.declare %arg3 uniq_name("_QFunrelated_readEm") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %c1_i32 = arith.constant 1 : i32
      %13 = fir.convert %c1_i32 : (i32) -> index
      %c100_i32 = arith.constant 100 : i32
      %14 = fir.convert %c100_i32 : (i32) -> index
      %c1_1 = arith.constant 1 : index
      fir.do_concurrent {
        %15 = fir.alloca i32 <{bindc_name = "i", pinned}>
        %16:2 = hlfir.declare %15 uniq_name("_QFunrelated_readEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        fir.do_concurrent.loop (%arg4) = (%13) to (%14) step (%c1_1) {
          %17 = fir.convert %arg4 : (index) -> i32
          fir.store %17 to %16#0 : !fir.ref<i32>
          %cst = arith.constant 1.000000e+00 : f32
          %18 = fir.load %16#0 : !fir.ref<i32>
          %19 = fir.convert %18 : (i32) -> i64
          %20 = hlfir.designate %11#0 (%19)  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
          hlfir.assign %cst to %20 : f32, !fir.ref<f32>
        }
      }
      omp.terminator
    }
    return
  }

// CHECK-LABEL: func.func @only_loop
// CHECK:         omp.target kernel_type(generic)
// CHECK-NOT:       fir.load
// CHECK-NOT:       fir.alloca
// CHECK-NOT:       fir.convert
// CHECK:           omp.teams {
// CHECK-NEXT:        omp.parallel {
// CHECK:               omp.distribute {
// CHECK:           } {omp.combined}
// CHECK:           omp.terminator
// CHECK-NEXT:    } {omp.combined}

  func.func @unused_read(%arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<f32> {fir.bindc_name = "m"}) {
    %0 = fir.dummy_scope : !fir.dscope
    %c100 = arith.constant 100 : index
    %1 = fir.shape %c100 : (index) -> !fir.shape<1>
    %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 arg 1 uniq_name("_QFunrelated_readEa") : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
    %3 = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFunrelated_readEi"}>
    %4:2 = hlfir.declare %3 uniq_name("_QFunrelated_readEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFunrelated_readEm") : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %6 = arith.subi %c100, %c1 : index
    %7 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%6 : index) extent(%c100 : index) stride(%c1 : index) start_idx(%c1 : index)
    %8 = omp.map.info var_ptr(%2#1 : !fir.ref<!fir.array<100xf32>>, !fir.array<100xf32>) map_clauses(tofrom) capture(ByRef) bounds(%7) name("a") -> !fir.ref<!fir.array<100xf32>>
    %9 = omp.map.info var_ptr(%5#1 : !fir.ref<f32>, f32) map_clauses(to) capture(ByRef) name("m") -> !fir.ref<f32>
    omp.target kernel_type(generic) map_entries(%8 -> %arg2, %9 -> %arg3 : !fir.ref<!fir.array<100xf32>>, !fir.ref<f32>) {
      %c100_0 = arith.constant 100 : index
      %10 = fir.shape %c100_0 : (index) -> !fir.shape<1>
      %11:2 = hlfir.declare %arg2(%10) uniq_name("_QFunrelated_readEa") : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
      %12:2 = hlfir.declare %arg3 uniq_name("_QFunrelated_readEm") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %m = fir.load %12#0 : !fir.ref<f32>
      %c1_i32 = arith.constant 1 : i32
      %13 = fir.convert %c1_i32 : (i32) -> index
      %c100_i32 = arith.constant 100 : i32
      %14 = fir.convert %c100_i32 : (i32) -> index
      %c1_1 = arith.constant 1 : index
      fir.do_concurrent {
        %15 = fir.alloca i32 <{bindc_name = "i", pinned}>
        %16:2 = hlfir.declare %15 uniq_name("_QFunrelated_readEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        fir.do_concurrent.loop (%arg4) = (%13) to (%14) step (%c1_1) {
          %17 = fir.convert %arg4 : (index) -> i32
          fir.store %17 to %16#0 : !fir.ref<i32>
          %cst = arith.constant 1.000000e+00 : f32
          %18 = fir.load %16#0 : !fir.ref<i32>
          %19 = fir.convert %18 : (i32) -> i64
          %20 = hlfir.designate %11#0 (%19)  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
          hlfir.assign %cst to %20 : f32, !fir.ref<f32>
        }
      }
      omp.terminator
    }
    return
  }

// CHECK-LABEL: func.func @unused_read
// CHECK:         omp.target kernel_type(generic)
// CHECK-NOT:       omp.teams
// CHECK-NOT:       omp.distribute
// CHECK:           omp.parallel {
// CHECK:             omp.wsloop {
// CHECK:           } {omp.combined}
// CHECK:           omp.terminator
// CHECK-NEXT:    {{^ *}}}{{$}}

  func.func @read_feeding_unused_op(%arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<f32> {fir.bindc_name = "m"}) {
    %0 = fir.dummy_scope : !fir.dscope
    %c100 = arith.constant 100 : index
    %1 = fir.shape %c100 : (index) -> !fir.shape<1>
    %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 arg 1 uniq_name("_QFunrelated_readEa") : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
    %3 = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFunrelated_readEi"}>
    %4:2 = hlfir.declare %3 uniq_name("_QFunrelated_readEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFunrelated_readEm") : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %6 = arith.subi %c100, %c1 : index
    %7 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%6 : index) extent(%c100 : index) stride(%c1 : index) start_idx(%c1 : index)
    %8 = omp.map.info var_ptr(%2#1 : !fir.ref<!fir.array<100xf32>>, !fir.array<100xf32>) map_clauses(tofrom) capture(ByRef) bounds(%7) name("a") -> !fir.ref<!fir.array<100xf32>>
    %9 = omp.map.info var_ptr(%5#1 : !fir.ref<f32>, f32) map_clauses(to) capture(ByRef) name("m") -> !fir.ref<f32>
    omp.target kernel_type(generic) map_entries(%8 -> %arg2, %9 -> %arg3 : !fir.ref<!fir.array<100xf32>>, !fir.ref<f32>) {
      %c100_0 = arith.constant 100 : index
      %10 = fir.shape %c100_0 : (index) -> !fir.shape<1>
      %11:2 = hlfir.declare %arg2(%10) uniq_name("_QFunrelated_readEa") : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
      %12:2 = hlfir.declare %arg3 uniq_name("_QFunrelated_readEm") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %m = fir.load %12#0 : !fir.ref<f32>
      %m2 = arith.addf %m, %m : f32
      %c1_i32 = arith.constant 1 : i32
      %13 = fir.convert %c1_i32 : (i32) -> index
      %c100_i32 = arith.constant 100 : i32
      %14 = fir.convert %c100_i32 : (i32) -> index
      %c1_1 = arith.constant 1 : index
      fir.do_concurrent {
        %15 = fir.alloca i32 <{bindc_name = "i", pinned}>
        %16:2 = hlfir.declare %15 uniq_name("_QFunrelated_readEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        fir.do_concurrent.loop (%arg4) = (%13) to (%14) step (%c1_1) {
          %17 = fir.convert %arg4 : (index) -> i32
          fir.store %17 to %16#0 : !fir.ref<i32>
          %cst = arith.constant 1.000000e+00 : f32
          %18 = fir.load %16#0 : !fir.ref<i32>
          %19 = fir.convert %18 : (i32) -> i64
          %20 = hlfir.designate %11#0 (%19)  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
          hlfir.assign %cst to %20 : f32, !fir.ref<f32>
        }
      }
      omp.terminator
    }
    return
  }

// CHECK-LABEL: func.func @read_feeding_unused_op
// CHECK:         omp.target kernel_type(generic)
// CHECK-NOT:       omp.teams
// CHECK-NOT:       omp.distribute
// CHECK:           omp.parallel {
// CHECK:             omp.wsloop {
// CHECK:           } {omp.combined}
// CHECK:           omp.terminator
// CHECK-NEXT:    {{^ *}}}{{$}}

  func.func @store(%arg0: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<f32> {fir.bindc_name = "m"}) {
    %0 = fir.dummy_scope : !fir.dscope
    %c100 = arith.constant 100 : index
    %1 = fir.shape %c100 : (index) -> !fir.shape<1>
    %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 arg 1 uniq_name("_QFunrelated_readEa") : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
    %3 = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFunrelated_readEi"}>
    %4:2 = hlfir.declare %3 uniq_name("_QFunrelated_readEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFunrelated_readEm") : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %6 = arith.subi %c100, %c1 : index
    %7 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%6 : index) extent(%c100 : index) stride(%c1 : index) start_idx(%c1 : index)
    %8 = omp.map.info var_ptr(%2#1 : !fir.ref<!fir.array<100xf32>>, !fir.array<100xf32>) map_clauses(tofrom) capture(ByRef) bounds(%7) name("a") -> !fir.ref<!fir.array<100xf32>>
    %9 = omp.map.info var_ptr(%5#1 : !fir.ref<f32>, f32) map_clauses(to) capture(ByRef) name("m") -> !fir.ref<f32>
    omp.target kernel_type(generic) map_entries(%8 -> %arg2, %9 -> %arg3 : !fir.ref<!fir.array<100xf32>>, !fir.ref<f32>) {
      %c100_0 = arith.constant 100 : index
      %10 = fir.shape %c100_0 : (index) -> !fir.shape<1>
      %11:2 = hlfir.declare %arg2(%10) uniq_name("_QFunrelated_readEa") : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
      %12:2 = hlfir.declare %arg3 uniq_name("_QFunrelated_readEm") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %two = arith.constant 2.0 : f32
      fir.store %two to %12#0 : !fir.ref<f32>
      %c1_i32 = arith.constant 1 : i32
      %13 = fir.convert %c1_i32 : (i32) -> index
      %c100_i32 = arith.constant 100 : i32
      %14 = fir.convert %c100_i32 : (i32) -> index
      %c1_1 = arith.constant 1 : index
      fir.do_concurrent {
        %15 = fir.alloca i32 <{bindc_name = "i", pinned}>
        %16:2 = hlfir.declare %15 uniq_name("_QFunrelated_readEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        fir.do_concurrent.loop (%arg4) = (%13) to (%14) step (%c1_1) {
          %17 = fir.convert %arg4 : (index) -> i32
          fir.store %17 to %16#0 : !fir.ref<i32>
          %cst = arith.constant 1.000000e+00 : f32
          %18 = fir.load %16#0 : !fir.ref<i32>
          %19 = fir.convert %18 : (i32) -> i64
          %20 = hlfir.designate %11#0 (%19)  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
          hlfir.assign %cst to %20 : f32, !fir.ref<f32>
        }
      }
      omp.terminator
    }
    return
  }

// CHECK-LABEL: func.func @store
// CHECK:         omp.target kernel_type(generic)
// CHECK-NOT:       omp.teams
// CHECK-NOT:       omp.distribute
// CHECK:           omp.parallel {
// CHECK:             omp.wsloop {
// CHECK:           } {omp.combined}
// CHECK:           omp.terminator
// CHECK-NEXT:    {{^ *}}}{{$}}

}
