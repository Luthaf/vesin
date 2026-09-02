program vesin_test
    use vesin
    use, intrinsic :: iso_c_binding
    implicit none

    type(NeighborList) :: neighbor_list
    integer(c_size_t), parameter :: n_points=2
    real :: points(3, n_points)
    real :: box(3, 3)
    integer :: ierr

    integer(c_size_t) :: expected_pairs(2, 10)
    integer(c_int32_t) :: expected_shifts(3, 10)
    real(c_double) :: expected_vectors(3, 10)
    real(c_double) :: expected_distances(10)

    integer(c_int32_t) :: zero_shifts(3, 10) = 0
    real(c_double) :: zero_vectors(3, 10) = 0.0
    real(c_double) :: zero_distances(10) = 0.0

    integer :: i


    points(:, 1) = [0.0, 0.0, 0.0]
    points(:, 2) = [0.0, 1.3, 1.3]

    box(:, 1) = [3.2, 0.0, 0.0]
    box(:, 2) = [0.0, 3.2, 0.0]
    box(:, 3) = [0.0, 0.0, 3.2]


    call neighbor_list%compute(points, box, periodic=[.true., .true., .true.], status=ierr)
    if (ierr == 0) call print_and_exit("expected error")
    if (neighbor_list%errmsg /= "NeighborList has to be initialized before calling compute") then
        call print_and_exit("got wrong error")
    endif

    ! check that we can initialize with float or double
    neighbor_list = NeighborList(cutoff=3.3, full=.true.)

    call neighbor_list%free()
    neighbor_list = NeighborList(cutoff=real(3.3, c_double), algorithm=VesinAutoAlgorithm, full=.false.)

    call neighbor_list%free()
    neighbor_list = NeighborList(       &
        cutoff=real(3.3, c_float),      &
        full=.false.,                   &
        return_distances=.true.,        &
        sorted=.true.                   &
    )

    call neighbor_list%compute(points, box, periodic=[.true., .true., .true.], status=ierr)
    if (ierr /= 0) call print_and_exit(neighbor_list%errmsg)

    if (neighbor_list%length /= 10) call print_and_exit("wrong number of pairs")

    expected_pairs = reshape([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [2, 10])
    expected_distances = [&
        2.6870059, 2.3021729, 2.3021729, 1.8384775, 3.2000000, &
        3.2000000, 3.2000000, 3.2000000, 3.2000000, 3.2000000  &
    ]
    expected_shifts = reshape([ &
         0, -1, -1, &
         0,  0, -1, &
         0, -1,  0, &
         0,  0,  0, &
         1,  0,  0, &
         0,  1,  0, &
         0,  0,  1, &
         1,  0,  0, &
         0,  1,  0, &
         0,  0,  1  &
    ], [3, 10])

    expected_vectors = reshape([ &
         0.0000000, -1.9000000, -1.9000000, &
         0.0000000,  1.2999999, -1.9000000, &
         0.0000000, -1.9000000,  1.2999999, &
         0.0000000,  1.2999999,  1.2999999, &
         3.2000000,  0.0000000,  0.0000000, &
         0.0000000,  3.2000000,  0.0000000, &
         0.0000000,  0.0000000,  3.2000000, &
         3.2000000,  0.0000000,  0.0000000, &
         0.0000000,  3.2000000,  0.0000000, &
         0.0000000,  0.0000000,  3.2000000  &
    ], [3, 10])

    ! Half-list pair ordering is not guaranteed to be stable -- which point
    ! ends up as `i` vs `j` for a given pair can depend on which cell it
    ! falls into -- so compare the canonicalized entries as an
    ! order-independent collection instead of relying on exact positions.
    call check_half_list(                                    &
        neighbor_list%pairs, zero_shifts, neighbor_list%distances, zero_vectors, &
        expected_pairs, zero_shifts, expected_distances, zero_vectors,           &
        10_c_size_t, .false., .true., .false.                 &
    )

    if (associated(neighbor_list%shifts)) call print_and_exit("shifts should not be there")
    if (associated(neighbor_list%vectors)) call print_and_exit("vectors should not be there")

    call neighbor_list%free()
    neighbor_list = NeighborList(           &
        cutoff=3.3,                         &
        full=.false.,                       &
        skin=0.0,                           &
        return_distances=.false.,           &
        return_vectors=.true.,              &
        return_shifts=.true.,               &
        sorted=.true.                       &
    )

    call neighbor_list%compute(points, box, periodic=[.true., .true., .true.], status=ierr)
    if (ierr /= 0) call print_and_exit(neighbor_list%errmsg)

    print *, "shifts: ["
    do i=1,neighbor_list%length
        print *, neighbor_list%shifts(:, i)
    end do
    print *, "]"

    print *, "vectors: ["
    do i=1,neighbor_list%length
        print *, neighbor_list%vectors(:, i)
    end do
    print *, "]"

    if (neighbor_list%length /= 10) call print_and_exit("wrong number of pairs")

    call check_half_list(                                                     &
        neighbor_list%pairs, neighbor_list%shifts, zero_distances, neighbor_list%vectors, &
        expected_pairs, expected_shifts, zero_distances, expected_vectors,     &
        10_c_size_t, .true., .false., .true.                                  &
    )

    if (associated(neighbor_list%distances)) call print_and_exit("distances should not be there")

    call neighbor_list%free()

contains

    subroutine print_and_exit(message)
        implicit none
        character(*) :: message

        print *, message
        stop 1
    end subroutine

    !> Sign (+1/-1) to apply to `(i, j)`/`shift`/`vector` to bring a half-list
    !! entry into canonical form with `i <= j`.
    integer function canonical_sign(i, j, shift) result(sign)
        implicit none
        integer(c_size_t), intent(in) :: i, j
        integer(c_int32_t), intent(in) :: shift(3)
        integer(c_int32_t) :: total

        sign = 1
        if (i > j) then
            sign = -1
        else if (i == j) then
            total = shift(1) + shift(2) + shift(3)
            if (total < 0 .or. (total == 0 .and. (shift(3) < 0 .or. (shift(3) == 0 .and. shift(2) < 0)))) then
                sign = -1
            end if
        end if
    end function

    !> Compare `actual` against `expected` as order-independent collections
    !! of half-list entries: for each expected entry, find a not-yet-matched
    !! actual entry whose canonicalized `(i, j)` (and, when requested,
    !! shift/distance/vector) match, exiting with an error if none is found.
    subroutine check_half_list(                                                &
        actual_pairs, actual_shifts, actual_distances, actual_vectors,         &
        expected_pairs, expected_shifts, expected_distances, expected_vectors, &
        n, check_shifts, check_distances, check_vectors                        &
    )
        implicit none
        integer(c_size_t), intent(in) :: n
        integer(c_size_t), intent(in) :: actual_pairs(2, n), expected_pairs(2, n)
        integer(c_int32_t), intent(in) :: actual_shifts(3, n), expected_shifts(3, n)
        real(c_double), intent(in) :: actual_distances(n), expected_distances(n)
        real(c_double), intent(in) :: actual_vectors(3, n), expected_vectors(3, n)
        logical, intent(in) :: check_shifts, check_distances, check_vectors

        integer(c_size_t) :: ai, aj, ei, ej
        integer(c_int32_t) :: a_sign, e_sign
        integer(c_int32_t) :: ashift(3), eshift(3)
        real(c_double) :: avector(3), evector(3)
        logical :: used(n)
        logical :: found
        integer(c_size_t) :: e, a
        real(c_double), parameter :: tol = 1e-5

        used = .false.

        do e = 1, n
            e_sign = canonical_sign(expected_pairs(1, e), expected_pairs(2, e), expected_shifts(:, e))
            if (e_sign == 1) then
                ei = expected_pairs(1, e); ej = expected_pairs(2, e)
                eshift = expected_shifts(:, e); evector = expected_vectors(:, e)
            else
                ei = expected_pairs(2, e); ej = expected_pairs(1, e)
                eshift = -expected_shifts(:, e); evector = -expected_vectors(:, e)
            end if

            found = .false.
            do a = 1, n
                if (used(a)) cycle

                a_sign = canonical_sign(actual_pairs(1, a), actual_pairs(2, a), actual_shifts(:, a))
                if (a_sign == 1) then
                    ai = actual_pairs(1, a); aj = actual_pairs(2, a)
                    ashift = actual_shifts(:, a); avector = actual_vectors(:, a)
                else
                    ai = actual_pairs(2, a); aj = actual_pairs(1, a)
                    ashift = -actual_shifts(:, a); avector = -actual_vectors(:, a)
                end if

                if (ai /= ei .or. aj /= ej) cycle
                if (check_shifts .and. any(ashift /= eshift)) cycle
                if (check_distances .and. abs(actual_distances(a) - expected_distances(e)) > tol) cycle
                if (check_vectors .and. any(abs(avector - evector) > tol)) cycle

                used(a) = .true.
                found = .true.
                exit
            end do

            if (.not. found) call print_and_exit("wrong pairs")
        end do
    end subroutine

end program
