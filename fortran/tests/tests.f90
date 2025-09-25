program vesin_test
    use vesin
    use, intrinsic :: iso_c_binding
    implicit none

    type(NeighborList) :: neighbor_list
    integer(c_size_t), parameter :: n_points=2
    real :: points(3, n_points)
    real :: box(3, 3)
    integer :: ierr

    integer :: expected_pairs(2, 10)
    integer :: expected_shifts(3, 10)
    real :: expected_vectors(3, 10)
    real :: expected_distances(10)


    points(:, 1) = [0.0, 0.0, 0.0]
    points(:, 2) = [0.0, 1.3, 1.3]

    box(:, 1) = [3.2, 0.0, 0.0]
    box(:, 2) = [0.0, 3.2, 0.0]
    box(:, 3) = [0.0, 0.0, 3.2]


    call neighbor_list%compute(points, box, periodic=.true., status=ierr)
    if (ierr == 0) call print_and_exit("expected error")
    if (neighbor_list%errmsg /= "NeighborList has to be initialized before calling compute") then
        call print_and_exit("got wrong error")
    endif

    ! check that we can initialize with float or double
    neighbor_list = NeighborList(cutoff=3.3, full=.true.)

    call neighbor_list%free()
    neighbor_list = NeighborList(cutoff=real(3.3, c_double), full=.false.)

    call neighbor_list%free()
    neighbor_list = NeighborList(cutoff=real(3.3, c_float), full=.false., return_distances=.true., sorted=.true.)

    call neighbor_list%compute(points, box, periodic=.true., status=ierr)
    if (ierr /= 0) call print_and_exit(neighbor_list%errmsg)

    if (neighbor_list%length /= 10) call print_and_exit("wrong number of pairs")

    expected_pairs = reshape([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], [2, 10])
    expected_distances = [&
        3.20000004, 3.2000000, 3.2000000, 2.6870059, 2.3021729, &
        2.30217293, 1.8384776, 3.2000000, 3.2000000, 3.2000000  &
    ]
    expected_shifts = reshape([ &
        0, 0, 1,    &
        0, 1, 0,    &
        1, 0, 0,    &
        0, -1, -1,  &
        0, -1, 0,   &
        0, 0, -1,   &
        0, 0, 0,    &
        0, 0, 1,    &
        0, 1, 0,    &
        1, 0, 0     &
    ], [3, 10])

    expected_vectors = reshape([ &
        0.0000000,  0.0000000,  3.2000000, &
        0.0000000,  3.2000000,  0.0000000, &
        3.2000000,  0.0000000,  0.0000000, &
        0.0000000, -1.9000000, -1.9000000, &
        0.0000000, -1.9000000,  1.3000000, &
        0.0000000,  1.3000000, -1.9000000, &
        0.0000000,  1.3000000,  1.3000000, &
        0.0000000,  0.0000000,  3.2000000, &
        0.0000000,  3.2000000,  0.0000000, &
        3.2000000,  0.0000000,  0.0000000  &
    ], [3, 10])

    if (any(neighbor_list%pairs /= expected_pairs)) call print_and_exit("wrong pairs")
    if (any(abs(neighbor_list%distances - expected_distances) > 1e-6)) call print_and_exit("wrong distances")

    if (associated(neighbor_list%shifts)) call print_and_exit("shifts should not be there")
    if (associated(neighbor_list%vectors)) call print_and_exit("vectors should not be there")

    call neighbor_list%free()
    neighbor_list = NeighborList(cutoff=3.3, full=.false., return_distances=.false., return_vectors=.true., return_shifts=.true., sorted=.true.)

    call neighbor_list%compute(points, box, periodic=.true., status=ierr)
    if (ierr /= 0) call print_and_exit(neighbor_list%errmsg)

    if (neighbor_list%length /= 10) call print_and_exit("wrong number of pairs")

    if (any(neighbor_list%pairs /= expected_pairs)) call print_and_exit("wrong pairs")

    if (any(neighbor_list%shifts /= expected_shifts)) call print_and_exit("wrong shifts")
    if (any(abs(neighbor_list%vectors - expected_vectors) > 1e-6)) call print_and_exit("wrong vectors")

    if (associated(neighbor_list%distances)) call print_and_exit("distances should not be there")

    call neighbor_list%free()

contains

    subroutine print_and_exit(message)
        implicit none
        character(*) :: message

        print *, message
        stop 1
    end subroutine

end program
