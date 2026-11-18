! Tests the conversion of a `do concurrent` loop inside a user-written
! `!$omp target` region when a `teams distribute parallel do` nest is NOT
! possible there. OpenMP requires a teams construct nested in a target to be
! the only thing in it, and code outside the teams region would run once per
! team. So whenever the loop is not directly in the target's body, or the
! target contains other code that must not be repeated (a store, another
! loop), or the loop's bounds read memory (e.g. a mapped `n`: re-reading it in
! every thread could observe a write made by an iteration, while Fortran
! evaluates the bounds once before the loop), the pass falls back to a
! `parallel do` nest inside the target:
!   - no new (nested) omp.target and no omp.teams / omp.distribute;
!   - omp.parallel (marked omp.combined) + omp.wsloop, as for
!     -fdo-concurrent-to-openmp=host;
!   - the user's target is left unchanged (in particular, not marked
!     omp.combined).
!
! The host module and the device module must agree, so both are checked
! with the same prefix.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   -fdo-concurrent-to-openmp=device %s -o - \
! RUN: | FileCheck %s

! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp \
! RUN:   -fopenmp-is-target-device -fdo-concurrent-to-openmp=device %s -o - \
! RUN: | FileCheck %s

! A store (`x = 2.0`) precedes the loop: it must not run once per team.
subroutine stmt_before_loop(a, x)
  implicit none
  integer :: i
  real :: a(100), x
  !$omp target map(tofrom: a, x)
  x = 2.0
  do concurrent (i = 1:100)
    a(i) = a(i) + x
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPstmt_before_loop
! CHECK:         omp.target kernel_type(generic)
! CHECK-NOT:       omp.target
! CHECK-NOT:       omp.teams
! CHECK:           omp.parallel {
! CHECK-NOT:         omp.distribute
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest
! CHECK:           } {omp.combined}
! CHECK-NOT:       fir.do_concurrent
! CHECK:           omp.terminator
! CHECK-NEXT:    {{^ *}}}{{$}}

! Two loops in one target: a target may contain at most one teams region.
subroutine two_loops(a, b)
  implicit none
  integer :: i
  real :: a(100), b(100)
  !$omp target map(tofrom: a, b)
  do concurrent (i = 1:100)
    a(i) = 1
  end do
  do concurrent (i = 1:100)
    b(i) = a(i) + 1
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtwo_loops
! CHECK:         omp.target kernel_type(generic)
! CHECK-NOT:       omp.target
! CHECK-NOT:       omp.teams
! CHECK:           omp.parallel {
! CHECK-NOT:         omp.distribute
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest
! CHECK:           } {omp.combined}
! CHECK-NOT:       omp.target
! CHECK-NOT:       omp.teams
! CHECK:           omp.parallel {
! CHECK-NOT:         omp.distribute
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest
! CHECK:           } {omp.combined}
! CHECK-NOT:       fir.do_concurrent
! CHECK:           omp.terminator
! CHECK-NEXT:    {{^ *}}}{{$}}

! The loop is inside an `if`: a teams region there would be illegal.
subroutine loop_inside_if(a, flag)
  implicit none
  integer :: i
  real :: a(100)
  logical :: flag
  !$omp target map(tofrom: a) map(to: flag)
  if (flag) then
    do concurrent (i = 1:100)
      a(i) = 3
    end do
  end if
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPloop_inside_if
! CHECK:         omp.target kernel_type(generic)
! CHECK:           fir.if
! CHECK-NOT:       omp.target
! CHECK-NOT:       omp.teams
! CHECK:           omp.parallel {
! CHECK-NOT:         omp.distribute
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest
! CHECK:           } {omp.combined}
! CHECK-NOT:       fir.do_concurrent
! CHECK:           omp.terminator
! CHECK-NEXT:    {{^ *}}}{{$}}

! The loop is inside a sequential loop in the target.
subroutine loop_inside_sequential_do(a)
  implicit none
  integer :: i, k
  real :: a(100)
  !$omp target map(tofrom: a)
  do k = 1, 3
    do concurrent (i = 1:100)
      a(i) = a(i) + 1
    end do
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPloop_inside_sequential_do
! CHECK:         omp.target kernel_type(generic)
! CHECK:           fir.do_loop
! CHECK-NOT:       omp.target
! CHECK-NOT:       omp.teams
! CHECK:           omp.parallel {
! CHECK-NOT:         omp.distribute
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest
! CHECK:           } {omp.combined}
! CHECK-NOT:       fir.do_concurrent
! CHECK:           omp.terminator
! CHECK-NEXT:    {{^ *}}}{{$}}

! The upper bound is read from a mapped variable. The fallback reads it once,
! in the target body, before omp.parallel.
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
! CHECK:           %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name("_QFbound_from_mapped_varEn")
! CHECK:           %[[N:.*]] = fir.load %[[N_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[UB:.*]] = fir.convert %[[N]] : (i32) -> index
! CHECK-NOT:       omp.teams
! CHECK:           omp.parallel {
! CHECK-NOT:         fir.load %[[N_DECL]]
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%[[UB]]) inclusive step (%{{.*}}) {
! CHECK:           } {omp.combined}
! CHECK:           omp.terminator
! CHECK-NEXT:    {{^ *}}}{{$}}

! Explicit-shape dummy array whose loop bound is also read from `n`.
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
! CHECK-NOT:       omp.teams
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop {
! CHECK:           } {omp.combined}
! CHECK:           omp.terminator
! CHECK-NEXT:    {{^ *}}}{{$}}
