! Offloading test for `do concurrent` loops that are already inside a
! user-written `!$omp target` region, converted with
! -fdo-concurrent-to-openmp=device. Such loops must not get a nested target
! region; depending on what else is in the target, they become a
! `teams distribute parallel do` nest (loop alone in the target) or a
! `parallel do` nest (fallback). Each case checks the computed result.
!
! REDUCE inside a user target is not covered here: without optimization the
! resulting generic kernel with a teams reduction computes a wrong result, the
! same as the equivalent hand-written `!$omp teams reduction` /
! `!$omp distribute parallel do reduction` code; that is a separate issue in
! generic-mode teams reductions, not in the do concurrent conversion.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -fdo-concurrent-to-openmp=device
! RUN: env OMP_TARGET_OFFLOAD=MANDATORY %libomptarget-run-generic 2>&1 | %fcheck-generic
program do_concurrent_in_target
  implicit none
  integer, parameter :: n = 10000
  integer :: i, j, k, nn
  real :: a(n), b(n), t, x
  real :: c(100, 100)
  logical :: flag
  nn = n

  ! Teams nest: the loop is alone in the target; the upper bound is read from
  ! a mapped variable inside the target region.
  a = 0
  !$omp target map(tofrom: a) map(to: nn)
  do concurrent (i = 1:nn)
    a(i) = a(i) + 1
  end do
  !$omp end target
  print *, "teams, bound from mapped variable:", sum(a)

  ! Teams nest in a subroutine with an explicit-shape dummy array: both the
  ! loop bound and the array's shape come from a mapped dummy argument.
  a = 0
  call add_index(a, n)
  print *, "teams, explicit-shape dummy:", sum(a)

  ! Teams nest with a LOCAL variable. Each iteration writes its own value to
  ! `t` and reads it back; a shared `t` would mix up iterations. The values
  ! are small so that the sum is exact in single precision.
  a = 0
  !$omp target map(tofrom: a) map(to: t)
  do concurrent (i = 1:n) local(t)
    t = real(mod(i, 7))
    a(i) = t * 2
  end do
  !$omp end target
  print *, "teams, local:", sum(a)

  ! Teams nest with a LOCAL_INIT variable (re-initialized every iteration).
  a = 0; t = 5.0
  !$omp target map(tofrom: a) map(to: t)
  do concurrent (i = 1:n) local_init(t)
    t = t + 1
    a(i) = t
  end do
  !$omp end target
  print *, "teams, local_init:", sum(a)

  ! Teams nest with a collapsed 2-D iteration space.
  c = 0
  !$omp target map(tofrom: c)
  do concurrent (i = 1:100, j = 1:100)
    c(i, j) = 1
  end do
  !$omp end target
  print *, "teams, 2-D:", sum(c)

  ! Fallback: a statement precedes the loop in the target.
  a = 0; x = 0
  !$omp target map(tofrom: a, x)
  x = 2.0
  do concurrent (i = 1:n)
    a(i) = a(i) + x
  end do
  !$omp end target
  print *, "fallback, statement before loop:", sum(a)

  ! Fallback: two loops in one target.
  a = 0; b = 0
  !$omp target map(tofrom: a, b)
  do concurrent (i = 1:n)
    a(i) = 1
  end do
  do concurrent (i = 1:n)
    b(i) = a(i) + 1
  end do
  !$omp end target
  print *, "fallback, two loops:", sum(b)

  ! Fallback: the loop is inside an `if`.
  a = 0; flag = .true.
  !$omp target map(tofrom: a) map(to: flag)
  if (flag) then
    do concurrent (i = 1:n)
      a(i) = 3
    end do
  end if
  !$omp end target
  print *, "fallback, loop inside if:", sum(a)

  ! Fallback: the loop is inside a sequential loop.
  a = 0
  !$omp target map(tofrom: a)
  do k = 1, 3
    do concurrent (i = 1:n)
      a(i) = a(i) + 1
    end do
  end do
  !$omp end target
  print *, "fallback, loop inside sequential do:", sum(a)
contains
  subroutine add_index(v, m)
    integer :: m, i
    real :: v(m)
    !$omp target map(tofrom: v) map(to: m)
    do concurrent (i = 1:m)
      v(i) = v(i) + mod(i, 5)
    end do
    !$omp end target
  end subroutine
end program

! CHECK: teams, bound from mapped variable: 10000.
! CHECK: teams, explicit-shape dummy: 20000.
! CHECK: teams, local: 59996.
! CHECK: teams, local_init: 60000.
! CHECK: teams, 2-D: 10000.
! CHECK: fallback, statement before loop: 20000.
! CHECK: fallback, two loops: 20000.
! CHECK: fallback, loop inside if: 30000.
! CHECK: fallback, loop inside sequential do: 30000.
