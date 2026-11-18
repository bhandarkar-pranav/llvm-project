! Tests the conversion of a `do concurrent` loop that is already inside a
! user-written `!$omp target` region, when apart from the loop the region
! only contains variable declarations (and the code computing their shapes)
! and constants, and the loop's bounds do not read memory. (A bound that reads
! memory, e.g. a mapped `n`, gets the `parallel do` fallback for now; see
! in_target_fallback.f90.)
!
! With -fdo-concurrent-to-openmp=device, the pass must not create a new
! (nested) omp.target. Instead it emits the `teams distribute parallel do`
! nest inside the user's target region:
!   - no new omp.target, map_entries or host_eval;
!   - the user's target keeps its kernel type (generic) and is marked
!     omp.combined, as lowering does for `!$omp target` + nested
!     `!$omp teams ...`, so that the target captures the teams construct;
!   - privates go on omp.parallel, reductions on omp.teams and omp.wsloop,
!     where the wsloop reduces into the teams reduction block argument.
!
! The host module and the device module must agree on the generated nest, so
! both are checked with the same prefix.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   -fdo-concurrent-to-openmp=device %s -o - \
! RUN: | FileCheck %s

! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp \
! RUN:   -fopenmp-is-target-device -fdo-concurrent-to-openmp=device %s -o - \
! RUN: | FileCheck %s

! Constant bounds: the bound computations (converts of constants) are
! re-evaluated inside omp.parallel, before omp.distribute, and the originals
! are erased, so only constants, shapes and declarations precede omp.teams.
subroutine constant_bounds(a)
  implicit none
  integer :: i
  real :: a(1000)
  !$omp target map(tofrom: a)
  do concurrent (i = 1:1000)
    a(i) = a(i) + 1
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPconstant_bounds
! CHECK:         omp.target kernel_type(generic)
! CHECK-NOT:     host_eval
! CHECK-SAME:    {
! CHECK:           hlfir.declare %{{.*}} {{.*}}uniq_name("_QFconstant_boundsEa")
! CHECK-NOT:       fir.load
! CHECK-NOT:       fir.alloca
! CHECK-NOT:       fir.convert
! CHECK-NOT:       omp.target
! CHECK:           omp.teams {
! CHECK-NEXT:        omp.parallel {
! CHECK:               %[[LB:.*]] = fir.convert %{{.*}} : (i32) -> index
! CHECK:               %[[UB:.*]] = fir.convert %{{.*}} : (i32) -> index
! CHECK:               omp.distribute {
! CHECK-NEXT:            omp.wsloop {
! CHECK-NEXT:              omp.loop_nest (%{{.*}}) : index = (%[[LB]]) to (%[[UB]]) inclusive step (%{{.*}}) {
! CHECK-NOT:       fir.do_concurrent
! CHECK:                 } {omp.composite}
! CHECK-NEXT:          } {omp.composite}
! CHECK:             } {omp.composite}
! CHECK:           } {omp.combined}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {omp.combined}

! LOCAL and REDUCE inside the user's target.
subroutine with_local_and_reduce(a, s, t)
  implicit none
  integer :: i
  real :: a(100), s, t
  !$omp target map(to: a) map(tofrom: s) map(to: t)
  do concurrent (i = 1:100) local(t) reduce(+:s)
    t = a(i) * 2
    s = s + t
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPwith_local_and_reduce
! CHECK:         omp.target kernel_type(generic)
! CHECK:           %[[S:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFwith_local_and_reduceEs")
! CHECK:           %[[T:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFwith_local_and_reduceEt")
! CHECK-NOT:       omp.target
! CHECK:           omp.teams reduction(@[[RED:.*]] %[[S]]#0 -> %[[TEAMS_S:.*]] : !fir.ref<f32>) {
! CHECK-NEXT:        omp.parallel private(@{{.*}}Et_private_f32.omp %[[T]]#0 -> %[[PRIV_T:.*]] : !fir.ref<f32>) {
! CHECK:               omp.distribute {
! CHECK-NEXT:            omp.wsloop reduction(@[[RED]] %[[TEAMS_S]] -> %[[WS_S:.*]] : !fir.ref<f32>) {
! CHECK-NEXT:              omp.loop_nest
! CHECK:                     hlfir.declare %[[PRIV_T]] uniq_name("_QFwith_local_and_reduceEt")
! CHECK:                     hlfir.declare %[[WS_S]] uniq_name("_QFwith_local_and_reduceEs")
! CHECK:           } {omp.combined}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {omp.combined}

! Explicit-shape dummy array with constant loop bounds: the extent `n` is
! loaded in the target body only to build the shape of `a`'s declaration.
! Code that only feeds a declaration is allowed before omp.teams (lowering
! emits the same code for the hand-written `!$omp target` +
! `!$omp teams distribute parallel do`), so this gets the teams nest.
subroutine explicit_shape_dummy(a, n)
  implicit none
  integer :: n, i
  real :: a(n)
  !$omp target map(tofrom: a) map(to: n)
  do concurrent (i = 1:100)
    a(i) = a(i) + real(i)
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPexplicit_shape_dummy
! CHECK:         omp.target kernel_type(generic)
! CHECK:           %[[EXT:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:           hlfir.declare %{{.*}}(%[[SHAPE]]) {{.*}}uniq_name("_QFexplicit_shape_dummyEa")
! CHECK:           hlfir.declare %{{.*}} {{.*}}uniq_name("_QFexplicit_shape_dummyEn")
! CHECK-NOT:       fir.load
! CHECK-NOT:       omp.target
! CHECK:           omp.teams {
! CHECK-NEXT:        omp.parallel {
! CHECK:               omp.distribute {
! CHECK-NEXT:            omp.wsloop {
! CHECK-NEXT:              omp.loop_nest
! CHECK:           } {omp.combined}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {omp.combined}
