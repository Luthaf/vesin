!> @brief
!! This is an example wrapper to the fortran interface of `vesin`.
!!
!! @details
!! It is suggested to copy and modify this file as needed by your application.
!! The basic functionalities are there, but for example the data conversion
!! from C precision to your precision has to be added. Likewise any additional
!! functionality to process the neighbor list.
!!
!! Example usage:
!!
!!~~~~~~~~~~~~~{.f90}
!! program main
!!     use vesin, only: NeighborList
!!
!!     implicit none
!!
!!     real, allocatable :: positions(:,:)
!!     real :: box(3,3)
!!     integer :: i, ierr
!!     type(NeighborList) :: neighbor_list
!!
!!     ! we have 2000 points in this example
!!     allocate(positions(3, 2000))
!!
!!     ! set the values for points `positions` and bounding `box` here
!!     ! positions = ...
!!     ! box = ...
!!
!!     ! initialize `neighbor_list` with some options
!!     neighbor_list = NeighborList(cutoff=4.2, full=.true., sorted=.true.)
!!     ! compute
!!     call neighbor_list%compute(positions, box, periodic=.true., status=ierr)
!!     if (ierr /= 0) then
!!         write(*,*) neighbor_list%errmsg
!!         stop
!!     end if
!!
!!     write(*,*) "we got ", neighbor_list%length, "pairs"
!!     do i=1,neighbor_list%length
!!         write(*,*) " - ", i, ":", neighbor_list%pairs(:, i)
!!     end do
!!
!!     ! release allocated memory
!!     call neighbor_list%free()
!! end program main
!!~~~~~~~~~~~~~

module vesin
    use, intrinsic :: iso_c_binding

    ! import the C-interoperable vesin interface
    use vesin_cdef, only: VesinOptions, VesinNeighborList, vesin_neighbors, vesin_free, VesinUnknownDevice, VesinCPU


    implicit none

    private
    public :: NeighborList

    !> @brief A neighbor list calculator.
    !!
    !! This type contains pointers to the computed C data, in C precision.
    type :: NeighborList
        !> .true. when instance has been initialized
        logical, private :: initialized = .false.

        !> Options for the calculation
        type(VesinOptions), private :: options

        !> The C struct holding neighbor data
        type(VesinNeighborList), private :: c_neighbors

        !> Latest error message
        character(:), allocatable :: errmsg

        !> Number of pairs in the list
        integer(c_size_t) :: length = 0_c_size_t

        !> Array of pairs (storing the indices of the first and second point in
        !! the pair). The shape is `[2, length]`.
        integer(c_size_t), pointer :: pairs(:,:) => null()

        !> Array of box shifts, one for each `pair`. This is only set if
        !! `return_pairs` was set to `.true.` during intialization. The shape
        !! is `[3, length]`
        integer(c_int32_t), pointer :: shifts(:,:) => null()

        !> Array of pair distance (i.e. distance between the two points), one
        !! for each `pair`. This is only set if `return_distances` was set to
        !! `.true.` during intialization. The shape is `[length]`
        real(c_double), pointer :: distances(:) => null()

        !> Array of pair vector (i.e. vector between the two points), one
        !! for each `pair`. This is only set if `return_vectors` was set to
        !! `.true.` during intialization. The shape is `[3, length]`
        real(c_double), pointer :: vectors(:, :) => null()

    contains
        procedure, private :: vesin_compute_c_double
        procedure, private :: vesin_compute_c_float

        generic, public :: compute => vesin_compute_c_double
        generic, public :: compute => vesin_compute_c_float

        procedure, public :: free => vesin_destroy
    end type NeighborList

    interface NeighborList
        procedure :: vesin_construct_c_float
        procedure :: vesin_construct_c_double
    end interface NeighborList

contains
    function vesin_construct_c_double(cutoff, full, sorted, return_shifts, return_distances, return_vectors) result(self)
        real(c_double), intent(in) :: cutoff
        logical, intent(in), optional :: full
        logical, intent(in), optional :: sorted
        logical, intent(in), optional :: return_shifts
        logical, intent(in), optional :: return_distances
        logical, intent(in), optional :: return_vectors
        type(neighborlist) :: self

        self%options%cutoff = cutoff

        if (present(full)) self%options%full = full
        if (present(sorted)) self%options%sorted = sorted
        if (present(return_shifts)) self%options%return_shifts = return_shifts
        if (present(return_distances)) self%options%return_distances = return_distances
        if (present(return_vectors)) self%options%return_vectors = return_vectors

        self%initialized = .true.
    end function vesin_construct_c_double

    function vesin_construct_c_float(cutoff, full, sorted, return_shifts, return_distances, return_vectors) result(self)
        real(c_float), intent(in) :: cutoff
        logical, intent(in), optional :: full
        logical, intent(in), optional :: sorted
        logical, intent(in), optional :: return_shifts
        logical, intent(in), optional :: return_distances
        logical, intent(in), optional :: return_vectors
        type(neighborlist) :: self
        ! set the cutoff and other stuff
        self = vesin_construct_c_double(real(cutoff, c_double), full, sorted, return_shifts, return_distances, return_vectors)
    end function vesin_construct_c_float

    !> Compute the neighbor list for data in `c_double` precision
    subroutine vesin_compute_c_double(self, points, box, periodic, status)
        implicit none
        class(NeighborList), intent(inout) :: self
        real(c_double), intent(in) :: points(:, :)
        real(c_double), intent(in) :: box(3, 3)
        logical, intent(in) :: periodic
        integer, optional :: status

        integer :: points_shape(2)
        logical(c_bool) :: c_periodic
        type(c_ptr) :: c_errmsg = c_null_ptr

        self%errmsg = ""

        if (.not. self%initialized) then
            self%errmsg = "NeighborList has to be initialized before calling compute"
            status = -1
            return
        end if

        c_periodic = periodic
        points_shape = shape(points)

        status = int(vesin_neighbors(           &
            points,                             &
            int(points_shape(2), c_size_t),     &
            box,                                &
            c_periodic,                         &
            VesinCPU,                           &
            self%options,                       &
            self%c_neighbors,                   &
            c_errmsg                            &
        ))

        if (status /= 0) then
            self%errmsg = c2f_string(c_errmsg)
            return
        end if

        self%length = self%c_neighbors%length

        ! reset all data in self
        if (associated(self%pairs)) nullify(self%pairs)
        if (associated(self%shifts)) nullify(self%shifts)
        if (associated(self%distances)) nullify(self%distances)
        if (associated(self%vectors)) nullify(self%vectors)

        if (c_associated(self%c_neighbors%pairs)) then
            call c_f_pointer(self%c_neighbors%pairs, self%pairs, shape=[2, int(self%length)])
        endif
        if (c_associated(self%c_neighbors%shifts)) then
            call c_f_pointer(self%c_neighbors%shifts, self%shifts, shape=[3, int(self%length)])
        endif
        if (c_associated(self%c_neighbors%distances)) then
            call c_f_pointer(self%c_neighbors%distances, self%distances, shape=[int(self%length)])
        endif
        if (c_associated(self%c_neighbors%vectors)) then
            call c_f_pointer(self%c_neighbors%vectors, self%vectors, shape=[3, int(self%length)])
        endif
    end subroutine vesin_compute_c_double

    !> Compute the neighbor list for data in `c_float` precision
    subroutine vesin_compute_c_float(self, points, box, periodic, status)
        implicit none
        class(NeighborList), intent(inout) :: self
        real(c_float), intent(in) :: points(:, :)
        real(c_float), intent(in) :: box(3, 3)
        logical, intent(in) :: periodic
        integer, optional :: status

        call vesin_compute_c_double(self, real(points, c_double), real(box, c_double), periodic, status)
    end subroutine vesin_compute_c_float

    !> Release all data allocated on the C side, and reset the `NeighborList`.
    !!
    !! This function should be called when you no longer need the data.
    subroutine vesin_destroy(self)
        class(NeighborList), intent(inout) :: self
        if (associated(self%pairs)) nullify(self%pairs)
        if (associated(self%shifts)) nullify(self%shifts)
        if (associated(self%distances)) nullify(self%distances)
        if (associated(self%vectors)) nullify(self%vectors)
        call vesin_free(self%c_neighbors)
        self%initialized = .false.
    end subroutine vesin_destroy


    ! transform `type(c_ptr)` string to fortran `character(:),allocatable` string
    function c2f_string(ptr) result(f_string)
        implicit none
        interface
        function c_strlen(str) bind(c, name='strlen')
            use iso_c_binding, only: c_ptr, c_size_t
            implicit none
            type(c_ptr), intent(in), value :: str
            integer(c_size_t) :: c_strlen
        end function c_strlen
        end interface
        type(c_ptr), intent(in) :: ptr
        character(len=:), allocatable :: f_string
        character(len=1, kind=c_char), dimension(:), pointer :: c_string
        integer :: n, i

        if (.not. c_associated(ptr)) then
        f_string = ' '
        else
        n = int(c_strlen(ptr), kind=kind(n))
        call c_f_pointer(ptr, c_string, [n+1])
        allocate(character(len=n)::f_string)
        do i = 1, n
            f_string(i:i) = c_string(i)
        end do
        end if
    end function c2f_string

end module vesin
