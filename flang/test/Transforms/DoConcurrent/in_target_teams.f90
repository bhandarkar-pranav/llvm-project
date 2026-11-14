! Tests the conversion of a `do concurrent` loop that is already inside a
! user-written `!$omp target` region, when apart from the loop the region
! only contains code that can be executed once per team without changing the
! program's behavior (computations without side effects, memory reads and
! variable declarations).
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

! The upper bound is read from a mapped variable, i.e. there is a fir.load
! inside the target region. Reads are safe to repeat per team, so this must
! still produce the teams nest. The computations the loop depends on are
! re-evaluated inside omp.parallel, before omp.distribute, which is where
! lowering places loop bounds for `!$omp target` + `!$omp teams` +
! `!$omp distribute parallel do`.
subroutine bound_from_mapped_var(a, n)
  implicit none
  integer :: n, i
  real :: a(n)
  !$omp target map(tofrom: a) map(to: n)
  do concurrent (i = 1:n)
    a(i) = a(i) + 1
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPbound_from_mapped_var
! CHECK:         omp.target kernel_type(generic)
! CHECK-NOT:     host_eval
! CHECK-SAME:    {
! CHECK:           %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name("_QFbound_from_mapped_varEn")
! Only constants, shapes and declarations remain before omp.teams: the
! original bound computations (and the original loop index allocation) are
! not left in the target body.
! CHECK-NOT:       fir.load
! CHECK-NOT:       fir.alloca
! CHECK-NOT:       fir.convert
! CHECK-NOT:       omp.target
! CHECK:           omp.teams {
! CHECK-NEXT:        omp.parallel {
! CHECK:               %[[N:.*]] = fir.load %[[N_DECL]]#0 : !fir.ref<i32>
! CHECK-NEXT:          %[[UB:.*]] = fir.convert %[[N]] : (i32) -> index
! CHECK:               omp.distribute {
! CHECK-NEXT:            omp.wsloop {
! CHECK-NEXT:              omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%[[UB]]) inclusive step (%{{.*}}) {
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

! Explicit-shape dummy array: the extent `n` is loaded in the target body to
! build the shape of `a`'s declaration. Code that only feeds a declaration is
! allowed before omp.teams (lowering emits the same code for the hand-written
! `!$omp target` + `!$omp teams distribute parallel do`), so this still gets
! the teams nest. The loop's own upper bound is a separate load, re-evaluated
! inside omp.parallel.
subroutine explicit_shape_dummy(a, n)
  implicit none
  integer :: n, i
  real :: a(n)
  !$omp target map(tofrom: a) map(to: n)
  do concurrent (i = 1:n)
    a(i) = a(i) + real(i)
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPexplicit_shape_dummy
! CHECK:         omp.target kernel_type(generic)
! CHECK:           %[[EXT:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:           hlfir.declare %{{.*}}(%[[SHAPE]]) {{.*}}uniq_name("_QFexplicit_shape_dummyEa")
! CHECK:           %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name("_QFexplicit_shape_dummyEn")
! CHECK-NOT:       fir.load
! CHECK-NOT:       omp.target
! CHECK:           omp.teams {
! CHECK-NEXT:        omp.parallel {
! CHECK:               %[[N:.*]] = fir.load %[[N_DECL]]#0 : !fir.ref<i32>
! CHECK-NEXT:          %[[UB:.*]] = fir.convert %[[N]] : (i32) -> index
! CHECK:               omp.distribute {
! CHECK-NEXT:            omp.wsloop {
! CHECK-NEXT:              omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%[[UB]]) inclusive step (%{{.*}}) {
! CHECK:           } {omp.combined}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {omp.combined}
