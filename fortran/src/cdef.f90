!> This is a Fortran interface to the C-API of `vesin`, from the header `vesin.h`.
module vesin_c
    use, intrinsic :: iso_c_binding
    implicit none

    private
    public :: VesinUnknownDevice, VesinCPU
    public :: VesinAutoAlgorithm, VesinBruteForce, VesinCellList
    public :: VesinDevice
    public :: VesinOptions
    public :: VesinNeighborList
    public :: vesin_neighbors
    public :: vesin_free

    !> Device type on which the data can be
    enum, bind(c)
        !> Unknown device, used for default initialization and to indicate no
        !! allocated data.
        enumerator :: VesinUnknownDevice = 0

        !> CPU device
        enumerator :: VesinCPU = 1
    end enum

    !> Algorithm to use for neighbor list construction (CUDA only)
    enum, bind(c)
        !> Automatically select algorithm based on system size
        enumerator :: VesinAutoAlgorithm = 0

        !> Brute-force O(n^2) algorithm with minimum image convention
        enumerator :: VesinBruteForce = 1

        !> Cell list algorithm with O(n) scaling
        enumerator :: VesinCellList = 2
    end enum

    !> Device on which data can be allocated
    type, bind(c) :: VesinDevice
        !> Device type (VesinUnknownDevice, VesinCPU, etc.)
        integer(c_int) :: type = VesinUnknownDevice

        !> Device index (0 for CPU, GPU index for CUDA)
        integer(c_int) :: device_id = 0
    end type VesinDevice

    !> Used for storing Vesin options.
    type, bind(c) :: VesinOptions
        !> Spherical cutoff, only pairs below this cutoff will be included
        real(c_double) :: cutoff = 0.0_c_double

        !> Should the returned neighbor list be a full list (include both `i -> j`
        !! and `j -> i` pairs) or a half list (include only `i -> j`)?
        logical(c_bool) :: full

        !> Should the neighbor list be sorted? If yes, the returned pairs will be
        !! sorted using lexicographic order.
        logical(c_bool) :: sorted = .false.

        !> Which algorithm to use for the calculation
        integer(c_int) :: algorithm = VesinAutoAlgorithm

        !> Should the returned `VesinNeighborList` contain `shifts`?
        logical(c_bool) :: return_shifts = .false.

        !> Should the returned `VesinNeighborList` contain `distances`?
        logical(c_bool) :: return_distances = .false.

        !> Should the returned `VesinNeighborList` contain `vector`?
        logical(c_bool) :: return_vectors = .false.

    end type VesinOptions

    !> Used as return type from `vesin_neighbors()`.
    !!
    !! Data is returned in the various `type(c_ptr)` pointers to memory allocated
    !! by C. They need to be transformed into fortran pointers in order to read
    !! the values.
    type, bind(c) :: VesinNeighborList
        !> Number of pairs in this neighbor list
        integer(c_size_t) :: length = 0_c_size_t

        !> Device used for the data allocations
        type(VesinDevice) :: device

        !> Array of pairs (storing the indices of the first and second point in
        !! the pair), containing `length` elements.
        type(c_ptr) :: pairs = c_null_ptr

        !> Array of box shifts, one for each `pair`. This is only set if
        !! `options.return_pairs` was `true` during the calculation.
        type(c_ptr) :: shifts = c_null_ptr

        !> Array of pair distance (i.e. distance between the two points), one for
        !! each pair. This is only set if `options.return_distances` was `true`
        !! during the calculation.
        type(c_ptr) :: distances = c_null_ptr

        !> Array of pair vector (i.e. vector between the two points), one for
        !! each pair. This is only set if `options.return_vector` was `true`
        !! during the calculation.
        type(c_ptr) :: vectors = c_null_ptr

        !> Private pointer used to hold additional internal data
        type(c_ptr) :: opaque = c_null_ptr
    end type VesinNeighborList

    !> Compute a neighbor list.
    !!
    !! The data is returned in a `VesinNeighborList`. For an initial call, the
    !! `VesinNeighborList` should be default-initalized. The `VesinNeighborList`
    !! can be re-used across calls to this functions to re-use memory allocations,
    !! and once it is no longer needed, users should call `vesin_free` to
    !! release the corresponding memory.
    interface
        function vesin_neighbors(&
            points,        &
            n_points,      &
            box,           &
            periodic,      &
            device,        &
            options,       &
            neighbors,     &
            error_message  &
        ) result(status) bind(c, name="vesin_neighbors")
            import:: c_double, c_size_t, c_bool, c_ptr, c_int
            import:: VesinDevice, VesinOptions, VesinNeighborList

            !> Number of elements in the `points` array
            integer(c_size_t), value   :: n_points

            !> Positions of all points in the system;
            real(c_double), intent(in) :: points(3, n_points)

            !> Bounding box for the system. If the system is non-periodic,
            !! this is ignored. This should contain the three vectors of the
            !! bounding box, one vector per column of the matrix.
            real(c_double), intent(in) :: box(3,3)

            !> Is the system using periodic boundary conditions? This
            !! should be an array of three booleans, one for each dimension.
            logical(c_bool)            :: periodic(3)

            !> Device where the `points` and `box` data is allocated.
            type(VesinDevice), value   :: device

            !> Options for the calculation
            type(VesinOptions), value  :: options

            !> A `type(VesinNeighborList)` instance, that will be used
            !! to store the computed list of neighbors.
            type(VesinNeighborList)    :: neighbors

            !> A `type(c_ptr)` to a null-terminated `char*` containing the
            !! error message of this function if any.
            type(c_ptr), intent(in)    :: error_message

            !> Non-zero integer upon error; zero otherwise.
            integer(c_int)             :: status

        end function vesin_neighbors
    end interface

    !> Free all allocated memory inside a `VesinNeighborList`, according to it's
    !! `device`.
    interface
        subroutine vesin_free(neighbors) bind(C, name="vesin_free")
        import:: VesinNeighborList
        type(VesinNeighborList) :: neighbors
        end subroutine vesin_free
    end interface
contains

end module vesin_c
