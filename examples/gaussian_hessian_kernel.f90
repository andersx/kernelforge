subroutine kernel_gaussian_hessian(x1, x2, dx1, dx2, q1, q2, n1, n2, nm1, nm2, na1, na2, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, dimension(:,:,:,:,:), intent(in) :: dx1
    double precision, dimension(:,:,:,:,:), intent(in) :: dx2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    integer, intent(in) :: na1
    integer, intent(in) :: na2

    double precision, intent(in) :: sigma

    double precision, dimension(na2*3,na1*3), intent(out) :: kernel

    integer :: i1, i2, j2, k
    integer :: xyz2
    integer :: a, b
    integer :: idx1_end, idx1_start, idx2_end, idx2_start, idx2

    integer :: rep_size

    double precision :: expd, expdiag

    double precision :: inv_2sigma2
    double precision :: inv_sigma4
    double precision :: sigma2

    double precision, allocatable, dimension(:) :: d

    double precision, allocatable, dimension(:,:) :: hess
    double precision, allocatable, dimension(:,:) :: partial

    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs1
    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs2

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))
    allocate(partial(rep_size,maxval(n2)*3))
    partial = 0.0d0
    allocate(hess(rep_size, rep_size))

    allocate(sorted_derivs1(rep_size,maxval(n1)*3,maxval(n1),nm1))
    allocate(sorted_derivs2(rep_size,maxval(n2)*3,maxval(n2),nm2))

    sorted_derivs1 = 0.0d0
    sorted_derivs2 = 0.0d0

    ! Presort the representation derivatives
    do b = 1, nm2
        do i2 = 1, n2(b)
            idx2 = 0

            do j2 = 1, n2(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs2(:,idx2,i2,b) = dx2(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    ! Presort the representation derivatives
    do b = 1, nm1
        do i2 = 1, n1(b)
            idx2 = 0

            do j2 = 1, n1(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs1(:,idx2,i2,b) = dx1(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    ! Reset kernel
    kernel = 0.0d0

    ! Calculate these only once
    inv_2sigma2 = -1.0d0 / (2 * sigma**2)
    inv_sigma4 = -1.0d0 / (sigma**4)
    sigma2 = -1.0d0 * sigma**2

    !$OMP PARALLEL DO PRIVATE(idx1_start,idx2_start,d,expd,expdiag,hess,idx1_end,idx2_end,partial) schedule(dynamic)
    do a = 1, nm1
        idx1_start = (sum(n1(:a)) - n1(a))*3 + 1
        idx1_end = sum(n1(:a))*3

        do b = 1, nm2
            idx2_start = (sum(n2(:b)) - n2(b))*3 + 1
            idx2_end = sum(n2(:b))*3

            ! Atoms A and B
            do i1 = 1, n1(a)
                do i2 = 1, n2(b)

                    if (q1(i1,a) == q2(i2,b)) then

                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,i1,:)- x2(b,i2,:)
                        expd = inv_sigma4 * exp(sum(d**2) * inv_2sigma2)
                        expdiag = sigma2 * expd

                        ! Calculate the outer product of the distance
                        hess = 0.0d0
                        call dsyr("U", rep_size, expd, d, 1, hess, rep_size)

                        do k = 1, rep_size
                           hess(k,k) = hess(k,k) + expdiag
                        enddo

                        ! ! Do the first half of the dot product, save in partial(:,:)
                        ! call dsymm("L", "U", rep_size, n1(a)*3, 1.0d0, hess(:,:), &
                        !     & rep_size, sorted_derivs1(:,:n1(a)*3,i1,a), rep_size, &
                        !     & 0.0d0, partial(:,:n1(a)*3), rep_size)

                        ! ! Add the dot product to the kernel in one BLAS call
                        ! call dgemm("T", "N", n1(a)*3, n2(b)*3, rep_size, 1.0d0, &
                        !     & partial(:,:n1(a)*3), rep_size, &
                        !     & sorted_derivs2(:,:n2(b)*3,i2,b), rep_size, 1.0d0, &
                        !     & kernel(idx1_start:idx1_end,idx2_start:idx2_end), n1(a)*3, 1)

                        ! Do the first half of the dot product, save in partial(:,:)
                        call dsymm("L", "U", rep_size, n2(b)*3, 1.0d0, hess(:,:), &
                            & rep_size, sorted_derivs2(:,:n2(b)*3,i2,b), rep_size, &
                            & 0.0d0, partial(:,:n2(b)*3), rep_size)

                        ! Add the dot product to the kernel in one BLAS call
                        call dgemm("T", "N", n2(b)*3, n1(a)*3, rep_size, 1.0d0, &
                            & partial(:,:n2(b)*3), rep_size, &
                            & sorted_derivs1(:,:n1(a)*3,i1,a), rep_size, 1.0d0, &
                            & kernel(idx2_start:idx2_end,idx1_start:idx1_end), n2(b)*3, 1)

                    endif

                enddo
            enddo

        enddo
    enddo
    !$OMP END PARALLEL do

    deallocate(hess)
    deallocate(sorted_derivs1)
    deallocate(sorted_derivs2)
    deallocate(partial)
    deallocate(d)

end subroutine kernel_gaussian_hessian 
