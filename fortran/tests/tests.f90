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

    integer :: k
    integer :: padding_count

    integer(c_size_t) :: i


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
    call check_half_list(                                  &
        n_pairs=10,                                        &
        actual_pairs=neighbor_list%pairs,                  &
        expected_pairs=expected_pairs,                     &
        actual_distances=neighbor_list%distances,          &
        expected_distances=expected_distances              &
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

    call check_half_list(                                  &
        n_pairs=10,                                        &
        actual_pairs=neighbor_list%pairs,                  &
        expected_pairs=expected_pairs,                     &
        actual_shifts=neighbor_list%shifts,                &
        expected_shifts=expected_shifts,                   &
        actual_vectors=neighbor_list%vectors,              &
        expected_vectors=expected_vectors                  &
    )

    if (associated(neighbor_list%distances)) call print_and_exit("distances should not be there")

    call neighbor_list%free()

    neighbor_list = NeighborList(           &
        cutoff=2.5,                         &
        full=.true.,                        &
        return_distances=.true.,            &
        return_vectors=.true.,              &
        return_shifts=.true.,               &
        padded_to=4                         &
    )

    call neighbor_list%compute(points, box, periodic=[.true., .true., .true.], status=ierr)
    if (ierr /= 0) call print_and_exit(neighbor_list%errmsg)
    if (neighbor_list%length /= 8) call print_and_exit("wrong number of pairs with padded_to")

    if (.not. associated(neighbor_list%pairs)) call print_and_exit("pairs should be present")
    if (.not. associated(neighbor_list%shifts)) call print_and_exit("shifts should be present")
    if (.not. associated(neighbor_list%vectors)) call print_and_exit("vectors should be present")
    if (.not. associated(neighbor_list%distances)) call print_and_exit("distances should be present")

    padding_count = 0
    do k = 1, int(neighbor_list%length)
        if (neighbor_list%pairs(1, k) == -1 .or. neighbor_list%pairs(2, k) == -1) then
            padding_count = padding_count + 1

            if (any(neighbor_list%shifts(:, k) /= 0)) call print_and_exit("padding shifts should be zero")
            if (any(abs(neighbor_list%vectors(:, k)) /= 0.0)) call print_and_exit("padding vectors should be zero")
            if (abs(neighbor_list%distances(k)) /= 0.0) call print_and_exit("padding distances should be zero")
        end if
    end do

    if (padding_count /= 2) then
        call print_and_exit("wrong number of padding pairs")
    end if

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
    !! actual entry whose canonicalized `(i, j)` (and shift/distance/vector,
    !! when the corresponding `expected_*` is given) match, exiting with an
    !! error if none is found. Each `actual_*` must be given together with the
    !! matching `expected_*`.
    subroutine check_half_list(                                                &
        n_pairs,                                                               &
        actual_pairs,                                                          &
        expected_pairs,                                                        &
        actual_shifts,                                                         &
        expected_shifts,                                                       &
        actual_distances,                                                      &
        expected_distances,                                                    &
        actual_vectors,                                                        &
        expected_vectors                                                       &
    )
        implicit none
        integer, intent(in) :: n_pairs
        integer(c_size_t), intent(in) :: actual_pairs(2, n_pairs)
        integer(c_size_t), intent(in) :: expected_pairs(2, n_pairs)
        integer(c_int32_t), intent(in), optional :: actual_shifts(3, n_pairs)
        integer(c_int32_t), intent(in), optional :: expected_shifts(3, n_pairs)
        real(c_double), intent(in), optional :: actual_distances(n_pairs)
        real(c_double), intent(in), optional :: expected_distances(n_pairs)
        real(c_double), intent(in), optional :: actual_vectors(3, n_pairs)
        real(c_double), intent(in), optional :: expected_vectors(3, n_pairs)

        integer(c_size_t) :: ai, aj, ei, ej
        integer(c_int32_t) :: ashift(3), eshift(3)
        real(c_double) :: avector(3), evector(3)
        logical :: used(n_pairs)
        logical :: found
        integer(c_size_t) :: e, a
        real(c_double), parameter :: tol = 1e-5

        used = .false.

        do e = 1, n_pairs
            ei = expected_pairs(1, e); ej = expected_pairs(2, e)
            eshift = 0; evector = 0.0
            if (present(expected_shifts)) eshift = expected_shifts(:, e)
            if (present(expected_vectors)) evector = expected_vectors(:, e)
            if (canonical_sign(ei, ej, eshift) == -1) then
                ei = expected_pairs(2, e); ej = expected_pairs(1, e)
                eshift = -eshift; evector = -evector
            end if

            found = .false.
            do a = 1, n_pairs
                if (used(a)) cycle

                ai = actual_pairs(1, a); aj = actual_pairs(2, a)
                ashift = 0; avector = 0.0
                if (present(actual_shifts)) ashift = actual_shifts(:, a)
                if (present(actual_vectors)) avector = actual_vectors(:, a)
                if (canonical_sign(ai, aj, ashift) == -1) then
                    ai = actual_pairs(2, a); aj = actual_pairs(1, a)
                    ashift = -ashift; avector = -avector
                end if

                if (ai /= ei .or. aj /= ej) cycle
                if (present(expected_shifts)) then
                    if (any(ashift /= eshift)) cycle
                end if
                if (present(expected_distances)) then
                    if (abs(actual_distances(a) - expected_distances(e)) > tol) cycle
                end if
                if (present(expected_vectors)) then
                    if (any(abs(avector - evector) > tol)) cycle
                end if

                used(a) = .true.
                found = .true.
                exit
            end do

            if (.not. found) call print_and_exit("wrong pairs")
        end do
    end subroutine

end program
