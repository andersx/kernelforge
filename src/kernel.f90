module kernelforge_mod

  use, intrinsic :: iso_c_binding
  implicit none

contains

  ! Example kernel: inverse distance (packed upper triangle)
subroutine compute_inverse_distance(x, n, d) bind(C, name="compute_inverse_distance")

    implicit none

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


subroutine kernel_symm_simple(X, lda, n, K, ldk, alpha) bind(C, name="kernel_symm_simple")

  integer(c_int), value       :: lda, n, ldk
  real(c_double), intent(in)  :: X(lda, *)
  real(c_double), intent(inout) :: K(ldk, *)
  real(c_double), value       :: alpha

  integer :: i, j, p
  real(c_double) :: dx, rij2, dist2

  !$omp parallel do private(i, j, dist2) shared(X, K, alpha, n) schedule(guided)
  do j = 1, n
      do i = 1, j
          dist2 = sum((X(:, i) - X(:, j))**2)
          K(i, j) = exp(alpha * dist2)
      end do
  end do
  !$omp end parallel do

end subroutine kernel_symm_simple


subroutine kernel_symm_blas(X, lda, n, K, ldk, alpha) bind(C, name="kernel_symm_blas")

  use, intrinsic :: iso_c_binding, only: c_int, c_double
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use omp_lib

  implicit none

  ! C ABI args
  integer(c_int), value        :: lda, n, ldk
  real(c_double), intent(in)   :: X(lda,*)
  real(c_double), intent(inout):: K(ldk,*)
  real(c_double), value        :: alpha

  ! Fortran default integers for BLAS calls
  integer :: lda_f, n_f, ldk_f, rep_size_f
  integer :: i, j
  real(c_double), allocatable :: diag(:), onevec(:)

  ! Copy c_int (by-value) to default INTEGERs for BLAS (expects default INTEGER by ref)
  lda_f = int(lda)
  n_f   = int(n)
  ldk_f = int(ldk)

  ! Rep size is the first dim of X; keep as default INTEGER
  rep_size_f = lda_f

  ! Gram matrix computation using DGEMM/DSYRK
  call dsyrk('U', 'T', int(n), int(lda), -2.0_dp * alpha, X, int(lda), 0.0_dp, K, int(n))

  allocate(diag(n_f), onevec(n_f))
  diag(:) = -0.5_dp * [ (K(i,i), i = 1, n) ]
  onevec(:) = 1.0_dp

  ! Add the (diagonal) self-inner products the matrix to form the distance matrix
  call dsyr2('U', n_f, 1.0_dp, onevec, 1, diag, 1, K, n_f)
  deallocate(diag, onevec)

  ! EXP double loop is fast compared to dsyrk anyway.
  !$omp parallel do private(i, j) shared(K, n) schedule(guided)
  do j = 1, n
      do i = 1, j
          K(i, j) = exp(K(i, j))
      end do
  end do
  !$omp end parallel do

end subroutine kernel_symm_blas



!subroutine fatomic_local_gradient_kernel(x1, dx1, x2, n1, nd1, n2 sigma, K)
!
!    implicit none
!
!    double precision, dimension(:,:,:), intent(in) :: x1
!    double precision, dimension(:,:,:), intent(in) :: x2
!
!    double precision, dimension(:,:,:,:,:), intent(in) :: dx2
!
!    double precision, intent(in) :: sigma
!
!    double precision, dimension(nd1,n2), intent(out) :: kernel
!
!    integer :: i2, j1, j2
!    integer :: xyz2
!
!    integer :: rep_size
!
!    double precision :: expd
!    double precision :: inv_2sigma2
!    double precision :: inv_sigma2
!
!    double precision, allocatable, dimension(:) :: d
!
!    rep_size = size(x1, dim=2)
!    allocate(d(rep_size))
!
!    inv_2sigma2 = -1.0d0 / (2 * sigma**2)
!    inv_sigma2 = -1.0d0 / (sigma**2)
!
!    kernel = 0.0d0
!
!    !$OMP PARALLEL DO PRIVATE(idx2_end,idx2_start,d,expd,idx1_start,idx1) schedule(dynamic)
!    do a = 1, nm1
!
!        idx1_start = sum(n1(:a)) - n1(a) + 1
!
!        do b = 1, nm2
!
!            idx2_start = (sum(n2(:b)) - n2(b))*3 + 1
!            idx2_end = sum(n2(:b))*3
!
!               ! Calculate the distance vector, and some intermediate results
!               d(:) = x1(a,j1,:)- x2(b,j2,:)
!               expd = inv_sigma2 * exp(sum(d**2) * inv_2sigma2)
!
!               ! Add the dot product to the kernel in one BLAS call
!               call dgemv("T", rep_size, n2(b)*3, expd, sorted_derivs(:,:n2(b)*3,j2,b), &
!                   & rep_size, d, 1, 1.0d0, kernel(idx2_start:idx2_end,idx1), 1)
!
!       enddo
!    enddo
!    !$OMP END PARALLEL do
!
!    deallocate(sorted_derivs)
!    deallocate(d)
!
!end subroutine fatomic_local_gradient_kernel


end module kernelforge_mod

