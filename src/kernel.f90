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
end module kernelforge_mod

