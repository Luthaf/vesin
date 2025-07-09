module vesin

  use, intrinsic :: iso_c_binding
  implicit none

  private
  public :: vesin_t


  !> @details
  !! Fortran interface to `vesin`.
  !! Defines the derived type `vesin_t`, which holds the options and pointers to the
  !! return values of the `vesin` neighbor list, in the C precision.
  !!
  !! @author Miha Gunde
  !!
  !! test program:
  !!
  !!~~~~~~~~{.f90}
  !!
  !! program main
  !!   use vesin, only: vesin_t
  !!   use, intrinsic :: iso_c_binding
  !!   implicit none
  !!   type( vesin_t ), pointer :: neigh
  !!   integer( c_size_t ), parameter :: nat=2
  !!   real(c_double) :: pos(3,nat)
  !!   real(c_double) :: box(3,3)
  !!   integer(c_int) :: ierr
  !!
  !!   ! positions
  !!   pos(:,1) = [ 0.0_c_double, 0.0_c_double, 0.0_c_double ]
  !!   pos(:,2) = [ 0.0_c_double, 1.3_c_double, 1.3_c_double ]
  !!
  !!   ! lattice
  !!   box(:,1) = [ 3.2_c_double, 0.0_c_double, 0.0_c_double ]
  !!   box(:,2) = [ 0.0_c_double, 3.2_c_double, 0.0_c_double ]
  !!   box(:,3) = [ 0.0_c_double, 0.0_c_double, 3.2_c_double ]
  !!
  !!   ! create the instance, set options
  !!   neigh => vesin_t( cutoff=4.2_c_double, full=.true., return_shifts=.true., return_distances=.true. )
  !!
  !!   ! launch computation of neighbor list
  !!   ierr = neigh% compute( nat, pos, box )
  !!   if( ierr/= 0 ) then
  !!      write(*,*) neigh% errmsg
  !!      stop
  !!   end if
  !!
  !!   ! data is inside `neigh`:
  !!   write(*,*) "got length:", neigh% length
  !!
  !!   ! destroy data
  !!   deallocate( neigh )
  !!
  !! end program main
  !!
  !!~~~~~~~~


  ! /// Device on which the data can be
  ! enum VesinDevice {
  ! /// Unknown device, used for default initialization and to indicate no
  ! /// allocated data.
  ! VesinUnknownDevice = 0,
  integer( c_int ), parameter :: VesinUnknownDevice = 0
  ! /// CPU device
  ! VesinCPU = 1,
  integer( c_int ), parameter :: VesinCPU = 1
  ! };


  !> @details
  !! Used for storing Vesin options.
  !! Equivalent to:
  !!
  !! struct VesinOptions {} from `vesin.h`
  !!
  type, bind(c) ::  VesinOptions

     real( c_double ) :: &
          ! /// Spherical cutoff, only pairs below this cutoff will be included
          ! double cutoff;
          cutoff = 0.0_c_double

     logical( c_bool ) :: &

          ! /// Should the returned neighbor list be a full list (include both `i -> j`
          ! /// and `j -> i` pairs) or a half list (include only `i -> j`)?
          ! bool full;
          full = .false., &

          ! /// Should the neighbor list be sorted? If yes, the returned pairs will be
          ! /// sorted using lexicographic order.
          ! bool sorted;
          sorted = .true., &

          ! /// Should the returned `VesinNeighborList` contain `shifts`?
          ! bool return_shifts;
          return_shifts = .false., &

          ! /// Should the returned `VesinNeighborList` contain `distances`?
          ! bool return_distances;
          return_distances = .false., &

          ! /// Should the returned `VesinNeighborList` contain `vector`?
          ! bool return_vectors;
          return_vectors = .false.

  end type VesinOptions


  !> @details
  !! Used as return type from `vesin_neighbors()`.
  !! Equvalent to:
  !!
  !! struct VESIN_API VesinNeighborList {} from `vesin.h`
  !!
  type, bind(c) :: VesinNeighborList

     ! /// Number of pairs in this neighbor list
     ! size_t length;
     integer( c_size_t ) :: length = 0_c_size_t

     ! /// Device used for the data allocations
     ! VesinDevice device;
     integer( c_int ) :: device = VesinUnknownDevice

     ! /// Array of pairs (storing the indices of the first and second point in the
     ! /// pair), containing `length` elements.
     ! size_t (*pairs)[2];
     type( c_ptr ) :: pairs = c_null_ptr

     ! /// Array of box shifts, one for each `pair`. This is only set if
     ! /// `options.return_pairs` was `true` during the calculation.
     ! int32_t (*shifts)[3];
     type( c_ptr ) :: shifts = c_null_ptr

     ! /// Array of pair distance (i.e. distance between the two points), one for
     ! /// each pair. This is only set if `options.return_distances` was `true`
     ! /// during the calculation.
     ! double *distances;
     type( c_ptr ) :: distances = c_null_ptr

     ! /// Array of pair vector (i.e. vector between the two points), one for
     ! /// each pair. This is only set if `options.return_vector` was `true`
     ! /// during the calculation.
     ! double (*vectors)[3];
     type( c_ptr ) :: vectors = c_null_ptr

  end type VesinNeighborList


  ! C-header:
  !
  !~~~~~~~~~~~~~~~~~{.c}
  ! int VESIN_API vesin_neighbors(
  !     const double (*points)[3],
  !     size_t n_points,
  !     const double box[3][3],
  !     bool periodic,
  !     VesinDevice device,
  !     struct VesinOptions options,
  !     struct VesinNeighborList* neighbors,
  !     const char** error_message
  ! );
  !~~~~~~~~~~~~~~~~~
  interface
     function fvesin_neighbors( &
          points,        &
          n_points,      &
          box,           &
          periodic,      &
          device,        &
          options,       &
          neighbors,     &
          error_message  &
       )result(res)bind( c, name="vesin_neighbors" )
       import :: c_double, c_size_t, c_bool, c_ptr, c_int, VesinOptions, VesinNeighborList
       integer( c_size_t ), value :: n_points
       real( c_double ), intent(in) :: points(3, n_points)
       real( c_double ), intent(in) :: box(3,3)
       logical( c_bool ), value :: periodic
       integer(c_int), value :: device
       type( VesinOptions ), value :: options
       type( VesinNeighborList ) :: neighbors
       type( c_ptr ) :: error_message
       integer( c_int ) :: res
     end function fvesin_neighbors
  end interface


  ! C-header:
  !
  !~~~~~~~~~~~~{.c}
  ! void VESIN_API vesin_free(struct VesinNeighborList* neighbors);
  !~~~~~~~~~~~~
  interface
     subroutine fvesin_free( neighbors ) bind(C, name="vesin_free")
       import :: VesinNeighborList
       type( VesinNeighborList ) :: neighbors
     end subroutine fvesin_free
  end interface



  !> @details
  !! fortran derived type, holding the input options, instance of `VesinNeighborList`,
  !! and pointers to the output data from Vesin in the C precision.
  !!
  type :: vesin_t

     ! options
     type( VesinOptions ), private :: opts

     ! returned C data, store the instance for re-use
     type( VesinNeighborList ), private :: cdata

     !! error message
     character(:), allocatable, public :: errmsg

     !! number of elements in the neighbor list
     integer( c_size_t ), public :: length

     !! Array of pairs (storing the indices of the first and second point in the
     !! pair), containing `length` elements.
     integer( c_size_t ),  pointer, public :: pairs(:,:) => null()

     !! Array of box shifts, one for each `pair`. This is only set if
     !! `return_pairs` option was `true` during the calculation.
     integer( c_int32_t ),  pointer, public :: shifts(:,:) => null()

     !! Array of pair distance (i.e. distance between the two points), one for
     !! each pair. This is only set if `return_distances` option was `true`
     !! during the calculation.
     real( c_double ), pointer, public :: distances(:) => null()

     !! Array of pair vector (i.e. vector between the two points), one for
     !! each pair. This is only set if `return_vectors` option was `true`
     !! during the calculation.
     real( c_double ), pointer, public :: vectors(:,:) => null()

   contains
     procedure, public :: compute => vesin_t_compute
     final :: vesin_t_destroy
  end type vesin_t

  ! overload name
  interface vesin_t
     procedure vesin_set_options
  end interface vesin_t


contains

  function vesin_set_options( &
       cutoff, &
       full, &
       sorted, &
       return_shifts, &
       return_distances, &
       return_vectors )result(self)
    !> @details
    !! Construct and set the options for `vesin_t`. This directly sets values to the private
    !! member `vesin_t% opts`, which is a `type( VesinOptions )` instance, used in the
    !! calculation of the neighbor list.
    !!
    !! @param `cutoff`, real(c_double) :: Spherical cutoff, only pairs below this cutoff will be included. Default=0.0
    !! @param `full`, logical, optional :: Should the returned neighbor list be a full
    !!       list (include both `i -> j` and `j -> i` pairs) or a half
    !!       list (include only `i -> j`)? Default=.false.
    !! @param `sorted`, logical, optional :: Should the neighbor list be sorted? If yes,
    !!       the returned pairs will be sorted using lexicographic order. Default=.true.
    !! @param `return_shifts`, logical, optional :: Should `vesin_t` contain `shifts`? Default=.false.
    !! @param `return_distances`, logical, optional :: Should `vesin_t` contain `distances`? Default=.false.
    !! @param `return_vectors`, logical, optional :: Should `vesin_t` contain `vectors`? Default=.false.
    !! @returns `self`, type( vesin_t ), pointer :: pointer to created `vesin_t` instance
    !!
    implicit none
    real( c_double ), intent(in) :: cutoff
    logical, intent(in), optional :: full
    logical, intent(in), optional :: sorted
    logical, intent(in), optional :: return_shifts
    logical, intent(in), optional :: return_distances
    logical, intent(in), optional :: return_vectors
    type( vesin_t ), pointer :: self

    allocate( vesin_t :: self )

    self% opts% cutoff = real( cutoff, c_double )
    if(present(full)            ) self% opts% full = logical( full, c_bool )
    if(present(sorted)          ) self% opts% sorted = logical( sorted, c_bool )
    if(present(return_shifts)   ) self% opts% return_shifts = logical( return_shifts, c_bool )
    if(present(return_distances)) self% opts% return_distances = logical( return_distances, c_bool )
    if(present(return_vectors)  ) self% opts% return_vectors = logical( return_vectors, c_bool )

  end function vesin_set_options


  function vesin_t_compute( self, nat, pos, box, periodic )result(ierr)
    !> @details
    !! Compute the neighbor list with options provided in `vesin_t% opts`.
    !! The data is recorded first in C format in `type(VesinNeighborList)`, then it is
    !! transferred to fortran format into `self`. The C data is destroyed at the end
    !! of this function.
    !!
    !! @param `nat`, integer(c_size_t) :: number of atoms
    !! @param `pos`, real(c_double), [3,nat] :: atomic positions
    !! @param `box`, real(c_double), [3,3] :: periodic box vectors
    !! @param `periodic`, logical, optional :: flag for (non)-periodic calculation. Default=.true.
    !! @returns `ierr`, integer(c_int) :: nonzero on error
    implicit none
    class( vesin_t ), intent(inout) :: self
    integer( c_size_t ), intent(in)  :: nat
    real( c_double ), intent(in) :: pos(3,nat)
    real( c_double ), intent(in) :: box(3,3)
    logical, intent(in), optional :: periodic
    integer( c_int ) :: ierr

    logical( c_bool ) :: c_periodic
    integer( c_int ) :: dev
    type( c_ptr ) :: c_errmsg

    integer :: n
    integer( c_size_t ), pointer :: pairs(:,:) => null()
    integer( c_int32_t ), pointer :: shifts(:,:) => null()
    real(c_double ), pointer :: distances(:) => null()
    real( c_double ), pointer :: vectors(:,:) => null()

    ! perform periodic calc by default
    c_periodic = .true.
    if(present(periodic)) c_periodic=logical(periodic, c_bool)

    ! set device
    dev = VesinCPU
    self%cdata% device = VesinCPU

    ! compute the neighbor list
    ierr = fvesin_neighbors(pos, nat, box, c_periodic, dev, self%opts, self%cdata, c_errmsg )
    if( ierr/= 0 ) then
       self% errmsg = c2f_string( c_errmsg )
       return
    end if

    ! set pointers to self%cdata
    self%length = int(self%cdata%length, kind(self%length))
    n = int( self%cdata%length )
    ! pairs
    if( c_associated(self%cdata%pairs)) then
       call c_f_pointer( self%cdata%pairs, self%pairs, shape=[2,n] )
    end if
    ! shifts
    if( c_associated(self%cdata%shifts)) then
       call c_f_pointer( self%cdata%shifts, self%shifts, shape=[3,n])
    end if
    ! distances
    if( c_associated(self%cdata%distances)) then
       call c_f_pointer( self%cdata%distances, self%distances, shape=[n] )
    end if
    ! vectors
    if( c_associated(self%cdata%vectors))then
       call c_f_pointer( self%cdata%vectors, self%vectors, shape=[3,n])
    end if

  end function vesin_t_compute


  subroutine vesin_t_destroy( self )
    !! destructor
    implicit none
    type( vesin_t ), intent(inout) :: self
    if( associated( self%pairs))     nullify( self%pairs )
    if( associated( self%shifts))    nullify( self%shifts )
    if( associated( self%distances)) nullify( self%distances )
    if( associated( self%vectors))   nullify( self%vectors )
    call fvesin_free( self%cdata )
  end subroutine vesin_t_destroy


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
       allocate( character(len=n)::f_string)
       do i = 1, n
          f_string(i:i) = c_string(i)
       end do
    end if
  end function c2f_string


end module vesin
