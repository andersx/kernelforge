module kernelforge_mod

  use, intrinsic :: iso_c_binding
  implicit none

contains
  
  ! Example kernel: inverse distance (packed upper triangle)
subroutine compute_inverse_distance(x, n, d) bind(C, name="compute_inverse_distance")
    integer(c_int), value       :: n
    real(c_double), intent(in)  :: x(3,n)               ! expect (3,n)
    real(c_double), intent(out) :: d(n*(n-1)/2)         ! packed upper triangle

    integer :: i, j, idx
    real(c_double) :: dx, dy, dz, rij2, rij

    idx = 0
    do j = 2, n
        do i = 1, j-1
            idx = idx + 1
            dx = x(1,i) - x(1,j)
            dy = x(2,i) - x(2,j)
            dz = x(3,i) - x(3,j)
            rij2 = dx*dx + dy*dy + dz*dz
            rij  = sqrt(rij2)
            d(idx) = 1.0d0 / rij
        end do
    end do
end subroutine compute_inverse_distance


subroutine kernel_simple(x, lda, n, k, ldk, alpha) bind(C, name="kernel_simple")
  ! X: (lda, n)  column-major
  ! K: (ldk, n)  column-major, only upper triangle (i<=j) is written
  integer(c_int), value       :: lda, n, ldk
  real(c_double), intent(in)  :: x(lda, *)
  real(c_double), intent(inout) :: k(ldk, *)
  real(c_double), value       :: alpha

  integer :: i, j, p
  real(c_double) :: dx, rij2

  !$omp parallel do private(i,j,p,dx,rij2) schedule(guided)
  do j = 1, n
    do i = 1, j
      rij2 = 0.0d0
      do p = 1, lda
        dx = x(p,i) - x(p,j)
        rij2 = rij2 + dx*dx
      end do
      k(i,j) = exp(alpha * rij2)
    end do
  end do
  !$omp end parallel do
end subroutine kernel_simple


end module kernelforge_mod

