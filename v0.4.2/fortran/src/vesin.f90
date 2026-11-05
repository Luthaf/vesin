!> High-level fortran interface to vesin.
!!
!! This is the recomended interface to vesin, taking care of data conversion
!! from and to the C API for you.
module vesin
    use, intrinsic :: iso_c_binding
    use vesin_c, only: VesinOptions, VesinNeighborList, &
                       vesin_neighbors, vesin_free, &
                       VesinUnknownDevice, VesinCPU
    implicit none

    private
    public :: NeighborList

    !> A neighbor list calculator.
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

    !> Initialize a `NeighborList`.
    !!
    !! The `cutoff` and `full` options are mandatory, the other are optional
    !! and default to `.false.`
    interface NeighborList
        procedure :: vesin_construct_c_float
        procedure :: vesin_construct_c_double
    end interface NeighborList

contains
    function vesin_construct_c_double(cutoff, full, sorted, return_shifts, return_distances, return_vectors) result(self)
        !> Spherical cutoff, only pairs below this cutoff will be included
        real(c_double), intent(in) :: cutoff

        !> Should the returned neighbor list be a full list (include both `i -> j`
        !! and `j -> i` pairs) or a half list (include only `i -> j`)?
        logical, intent(in) :: full

        !> Should the neighbor list be sorted? If yes, the returned pairs will be
        !! sorted using lexicographic order.
        logical, intent(in), optional :: sorted

        !> Should the returned `VesinNeighborList` contain `shifts`?
        logical, intent(in), optional :: return_shifts

        !> Should the returned `VesinNeighborList` contain `distances`?
        logical, intent(in), optional :: return_distances

        !> Should the returned `VesinNeighborList` contain `vector`?
        logical, intent(in), optional :: return_vectors

        type(neighborlist) :: self

        self%options%cutoff = cutoff
        self%options%full = full

        if (present(sorted)) self%options%sorted = sorted
        if (present(return_shifts)) self%options%return_shifts = return_shifts
        if (present(return_distances)) self%options%return_distances = return_distances
        if (present(return_vectors)) self%options%return_vectors = return_vectors

        self%initialized = .true.
    end function vesin_construct_c_double

    function vesin_construct_c_float(cutoff, full, sorted, return_shifts, return_distances, return_vectors) result(self)
        !> Spherical cutoff, only pairs below this cutoff will be included
        real(c_float), intent(in) :: cutoff

        !> Should the returned neighbor list be a full list (include both `i -> j`
        !! and `j -> i` pairs) or a half list (include only `i -> j`)?
        logical, intent(in) :: full

        !> Should the neighbor list be sorted? If yes, the returned pairs will be
        !! sorted using lexicographic order.
        logical, intent(in), optional :: sorted

        !> Should the returned `VesinNeighborList` contain `shifts`?
        logical, intent(in), optional :: return_shifts

        !> Should the returned `VesinNeighborList` contain `distances`?
        logical, intent(in), optional :: return_distances

        !> Should the returned `VesinNeighborList` contain `vector`?
        logical, intent(in), optional :: return_vectors

        type(neighborlist) :: self

        self = vesin_construct_c_double(        &
            real(cutoff, c_double),             &
            full,                               &
            sorted,                             &
            return_shifts,                      &
            return_distances,                   &
            return_vectors                      &
        )
    end function vesin_construct_c_float

    !> Compute the neighbor list for data in `c_double`/`real64` precision
    subroutine vesin_compute_c_double(self, points, box, periodic, status)
        implicit none
        !> The neighbor list calculator
        class(NeighborList), intent(inout) :: self
        !> Positions of all points to consider, this must be a `3 x n_points`
        !! array.
        real(c_double), intent(in) :: points(:, :)
        !> Bounding box for the system. If the system is non-periodic,
        !! this is ignored. This should contain the three vectors of the
        !! bounding box, one vector per column of the matrix.
        real(c_double), intent(in) :: box(3, 3)
        !> Is the system using periodic boundary conditions? This
        !! should be an array of three logical, one for each dimension.
        logical, intent(in) :: periodic(3)
        !> Status code of the operation, this will be 0 if there are no error,
        !! and non-zero otherwise. The full error message will be stored in
        !! `self%errmsg`.
        integer, optional :: status

        integer :: points_shape(2)
        logical(c_bool) :: c_periodic(3)
        type(c_ptr) :: c_errmsg = c_null_ptr

        self%errmsg = ""

        if (.not. self%initialized) then
            self%errmsg = "NeighborList has to be initialized before calling compute"
            status = -1
            return
        end if

        c_periodic = periodic
        points_shape = shape(points)

        if (points_shape(1) /= 3) then
            self%errmsg = "`points` should be a [3, n_points] array"
            status = -1
            return
        end if

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

    !> Compute the neighbor list for data in `c_float`/`real32` precision
    subroutine vesin_compute_c_float(self, points, box, periodic, status)
        implicit none
        !> The neighbor list calculator
        class(NeighborList), intent(inout) :: self
        !> Positions of all points to consider, this must be a `3 x n_points`
        !! array.
        real(c_float), intent(in) :: points(:, :)
        !> Bounding box for the system. If the system is non-periodic,
        !! this is ignored. This should contain the three vectors of the
        !! bounding box, one vector per column of the matrix.
        real(c_float), intent(in) :: box(3, 3)
        !> Is the system using periodic boundary conditions? This
        !! should be an array of three booleans, one for each dimension.
        logical, intent(in) :: periodic(3)
        !> Status code of the operation, this will be 0 if there are no error,
        !! and non-zero otherwise. The full error message will be stored in
        !! `self%errmsg`.
        integer, optional :: status

        call vesin_compute_c_double(self, real(points, c_double), real(box, c_double), periodic, status)
    end subroutine vesin_compute_c_float

    !> Release all data allocated on the C side, and reset the `NeighborList`.
    !!
    !! This function should be called when you no longer need the data.
    subroutine vesin_destroy(self)
        !> The neighbor list calculator
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
