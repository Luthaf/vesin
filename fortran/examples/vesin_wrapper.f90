!> @brief
!! This is an example wrapper to the fortran interface of `vesin`.
!!
!! @details
!! It is suggested to copy and modify this file as needed by your application.
!! The basic functionalities are there, but for example the data conversion
!! from C precision to your precision has to be added. Likewise any additional
!! functionality to process the neighbor list.
!!
!! Example (pseudo)-program:
!!
!!~~~~~~~~~~~~~{.f90}
!!  use vesin_wrapper
!!  integer(ip) :: nat
!!  real(rp), allocatable :: pos(:,:)
!!  real(rp) :: box(3,3)
!!  type( NeighborList ) :: neigh
!!  integer :: ierr, i
!!
!!  ! set values to nat, pos, box, ...
!!
!!  ! initialize with options
!!  neigh = NeighborList( cutoff=4.2_rp, full=.true., sorted=.true. )
!!
!!  ! compute neighbor list
!!  ierr = neigh% compute( nat, pos, lat, periodic=.true. )
!!  if( ierr /= 0 ) then
!!     write(*,*) neigh%errmsg
!!     stop
!!  end if
!!  write(*,*) "length:",neigh% length
!!  do i = 1, neigh% length
!!     write(*,*) i, ":", neigh% pairs(:,i)
!!  end do
!!
!!  ! set new options to the same allocation
!!  call neigh% options( cutoff=2.4_rp, return_distances=.true. )
!!  ierr = neigh% compute( nat, pos, lat, periodic=.true. )
!!  if( ierr /= 0 ) then
!!     write(*,*) neigh%errmsg
!!     stop
!!  end if
!!  write(*,*) "length:",neigh% length
!!  do i = 1, neigh% length
!!     write(*,*) i, ":", neigh%pairs(:,i), neigh% distances(i)
!!  end do
!!
!!  ! free the memory
!!  call neigh% free()
!!~~~~~~~~~~~~~
!!
module vesin_wrapper

  use, intrinsic :: iso_c_binding

  ! import the C-interoperable vesin interface
  use vesin, only: &
       c_VesinOptions => VesinOptions, &
       c_VesinNeighborList => VesinNeighborList, &
       c_vesin_neighbors => vesin_neighbors, &
       c_vesin_free => vesin_free, &
       VesinUnknownDevice, VesinCPU


  implicit none

  private
  public :: NeighborList



  !> @brief Neighbor list computed by `vesin`.
  !! @details Contains pointers to the computed C data, in C precision.
  type :: NeighborList

     type( c_vesinOptions ), private :: opts
     !< Computation options

     type( c_vesinNeighborList ), private :: cdata
     !< Returned C data

     integer, private :: device = VesinCPU
     !< device where data is coming from

     logical, private :: active = .false.
     !< .true. when instance has been initialized

     character(:), allocatable :: errmsg
     !< error message

     ! pointers to C data, in C precision
     integer(c_size_t) :: length = 0_c_size_t
     !< size of the list

     integer( c_size_t ), pointer :: pairs(:,:) => null()
     !< pair indices, shape[2, length]

     integer( c_int32_t ), pointer :: shifts(:,:) => null()
     !< periodic image shifts, shape[3, length]

     real( c_double ), pointer :: distances(:) => null()
     !< distances between corresponding pair of points, shape[length]

     real( c_double ), pointer :: vectors(:,:) => null()
     !< vectors corresponding to pair positions, shape[3, length]

   contains

     !> Set or change one or more Vesin options.
     procedure, public :: options => vesin_set_options

     !> Compute the neighbor list.
     procedure, private :: vesin_compute_c_size_t_c_float
     procedure, private :: vesin_compute_c_size_t_c_double
     procedure, private :: vesin_compute_c_int_c_float
     procedure, private :: vesin_compute_c_int_c_double
     generic, public :: compute => vesin_compute_c_size_t_c_float
     generic, public :: compute => vesin_compute_c_size_t_c_double
     generic, public :: compute => vesin_compute_c_int_c_float
     generic, public :: compute => vesin_compute_c_int_c_double

     !> Destructor.
     procedure, public :: free => vesin_destroy
  end type NeighborList

  !> @brief Constructor for `type(NeighborList)`.
  !! @details Creates a new instance of `NeighborList`.
  !! The value for `cutoff` is mandatory, other options can optionally be set.
  interface NeighborList
     procedure :: vesin_construct_c_float
     procedure :: vesin_construct_c_double
  end interface NeighborList

contains

  function vesin_construct_c_float( cutoff, full, sorted, return_shifts, &
         return_distances, return_vectors, device ) result( self )
    real(c_float), intent(in) :: cutoff
    logical, intent(in), optional :: full
    logical, intent(in), optional :: sorted
    logical, intent(in), optional :: return_shifts
    logical, intent(in), optional :: return_distances
    logical, intent(in), optional :: return_vectors
    integer, intent(in), optional :: device
    type( neighborlist ) :: self
    ! set the cutoff and other stuff
    call self% options( &
         cutoff=real(cutoff, c_double), &
         full=full, &
         sorted=sorted, &
         return_shifts=return_shifts, &
         return_distances=return_distances, &
         return_vectors=return_vectors, &
         device=device )
    ! flag as active
    self% active = .true.
  end function vesin_construct_c_float
  function vesin_construct_c_double( cutoff, full, sorted, return_shifts, &
         return_distances, return_vectors, device ) result( self )
    real(c_double), intent(in) :: cutoff
    logical, intent(in), optional :: full
    logical, intent(in), optional :: sorted
    logical, intent(in), optional :: return_shifts
    logical, intent(in), optional :: return_distances
    logical, intent(in), optional :: return_vectors
    integer, intent(in), optional :: device
    type( neighborlist ) :: self
    ! set the cutoff and other stuff
    call self% options( &
         cutoff=real(cutoff, c_double), &
         full=full, &
         sorted=sorted, &
         return_shifts=return_shifts, &
         return_distances=return_distances, &
         return_vectors=return_vectors, &
         device=device )
    ! flag as active
    self% active = .true.
  end function vesin_construct_c_double


  !> @brief Set or change one or more Vesin options.
  subroutine vesin_set_options( self, cutoff, full, sorted, return_shifts, &
                              return_distances, return_vectors, device )
    implicit none
    class( neighborlist ), intent(inout) :: self
    class(*), intent(in), optional :: cutoff
    logical, intent(in), optional :: full
    logical, intent(in), optional :: sorted
    logical, intent(in), optional :: return_shifts
    logical, intent(in), optional :: return_distances
    logical, intent(in), optional :: return_vectors
    integer, intent(in), optional :: device

    if(present(cutoff)) then
       select type(cutoff)
       type is( real(c_float) ); self%opts%cutoff = real( cutoff, c_double )
       type is( real(c_double) ); self%opts%cutoff = real( cutoff, c_double )
       end select
    end if
    if(present(full)) self%opts%full = full
    if(present(sorted)) self%opts%sorted = sorted
    if(present(return_shifts)) self%opts%return_shifts = return_shifts
    if(present(return_distances)) self%opts%return_distances = return_distances
    if(present(return_vectors)) self%opts%return_vectors = return_vectors
    if(present(device)) self%device = int( device, c_int )
  end subroutine vesin_set_options


  ! Set the pointers in `self` to computed C data.
  function vesin_compute_c_size_t_c_float( self, nat, pos, box, periodic ) result( ierr )
    implicit none
    class( NeighborList ), intent(inout) :: self
    integer( c_size_t ), intent(in) :: nat
    real( c_float ), intent(in) :: pos(3,nat)
    real( c_float ), intent(in) :: box(3,3)
    logical, intent(in), optional :: periodic
    integer :: ierr

    logical( c_bool ) :: c_periodic
    type( c_ptr ) :: c_errmsg = c_null_ptr
    integer :: n
    ! self has not been initialized
    if( .not. self% active ) then
       self% errmsg="NeighborList has to be initialized before computing."
       ierr = -1
       return
    end if
    ! periodic by default
    c_periodic = .true.
    if( present(periodic))c_periodic = periodic
    ! call iterface to c, with data in prescribed c precision
    ierr = int( c_vesin_neighbors( &
         real(pos, c_double), &
         int(nat, c_size_t), &
         real(box, c_double), &
         c_periodic, &
         self%device, &
         self%opts, &
         self%cdata, &
         c_errmsg) )
    if( ierr /= 0 ) then
       self% errmsg = c2f_string(c_errmsg)
       return
    end if
    ! cast cdata to f, in the returned C precision
    n = int( self%cdata%length )
    self% length = self%cdata%length
    ! nullify pointers in self
    if( associated(self%pairs))nullify(self%pairs)
    if( associated(self%shifts))nullify(self%shifts)
    if( associated(self%distances))nullify(self%distances)
    if( associated(self%vectors))nullify(self%vectors)
    ! set
    if(c_associated(self%cdata%pairs)) call c_f_pointer(self%cdata%pairs, self%pairs, shape=[2,n])
    if(c_associated(self%cdata%shifts)) call c_f_pointer(self%cdata%shifts, self%shifts, shape=[3,n])
    if(c_associated(self%cdata%distances)) call c_f_pointer(self%cdata%distances, self%distances, shape=[n])
    if(c_associated(self%cdata%vectors)) call c_f_pointer(self%cdata%vectors, self%vectors, shape=[3,n])
  end function vesin_compute_c_size_t_c_float
  function vesin_compute_c_int_c_float( self, nat, pos, box, periodic ) result( ierr )
    implicit none
    class( NeighborList ), intent(inout) :: self
    integer( c_int ), intent(in) :: nat
    real( c_float ), intent(in) :: pos(3,nat)
    real( c_float ), intent(in) :: box(3,3)
    logical, intent(in), optional :: periodic
    integer :: ierr

    logical( c_bool ) :: c_periodic
    type( c_ptr ) :: c_errmsg = c_null_ptr
    integer :: n
    ! self has not been initialized
    if( .not. self% active ) then
       self% errmsg="NeighborList has to be initialized before computing."
       ierr = -1
       return
    end if
    ! periodic by default
    c_periodic = .true.
    if( present(periodic))c_periodic = periodic
    ! call iterface to c, with data in prescribed c precision
    ierr = int( c_vesin_neighbors( &
         real(pos, c_double), &
         int(nat, c_size_t), &
         real(box, c_double), &
         c_periodic, &
         self%device, &
         self%opts, &
         self%cdata, &
         c_errmsg) )
    if( ierr /= 0 ) then
       self% errmsg = c2f_string(c_errmsg)
       return
    end if
    ! cast cdata to f, in the returned C precision
    n = int( self%cdata%length )
    self% length = self%cdata%length
    ! nullify pointers in self
    if( associated(self%pairs))nullify(self%pairs)
    if( associated(self%shifts))nullify(self%shifts)
    if( associated(self%distances))nullify(self%distances)
    if( associated(self%vectors))nullify(self%vectors)
    ! set
    if(c_associated(self%cdata%pairs)) call c_f_pointer(self%cdata%pairs, self%pairs, shape=[2,n])
    if(c_associated(self%cdata%shifts)) call c_f_pointer(self%cdata%shifts, self%shifts, shape=[3,n])
    if(c_associated(self%cdata%distances)) call c_f_pointer(self%cdata%distances, self%distances, shape=[n])
    if(c_associated(self%cdata%vectors)) call c_f_pointer(self%cdata%vectors, self%vectors, shape=[3,n])
  end function vesin_compute_c_int_c_float
  function vesin_compute_c_size_t_c_double( self, nat, pos, box, periodic ) result( ierr )
    implicit none
    class( NeighborList ), intent(inout) :: self
    integer( c_size_t ), intent(in) :: nat
    real( c_double ), intent(in) :: pos(3,nat)
    real( c_double ), intent(in) :: box(3,3)
    logical, intent(in), optional :: periodic
    integer :: ierr

    logical( c_bool ) :: c_periodic
    type( c_ptr ) :: c_errmsg = c_null_ptr
    integer :: n
    ! self has not been initialized
    if( .not. self% active ) then
       self% errmsg="NeighborList has to be initialized before computing."
       ierr = -1
       return
    end if
    ! periodic by default
    c_periodic = .true.
    if( present(periodic))c_periodic = periodic
    ! call iterface to c, with data in prescribed c precision
    ierr = int( c_vesin_neighbors( &
         real(pos, c_double), &
         int(nat, c_size_t), &
         real(box, c_double), &
         c_periodic, &
         self%device, &
         self%opts, &
         self%cdata, &
         c_errmsg) )
    if( ierr /= 0 ) then
       self% errmsg = c2f_string(c_errmsg)
       return
    end if
    ! cast cdata to f, in the returned C precision
    n = int( self%cdata%length )
    self% length = self%cdata%length
    ! nullify pointers in self
    if( associated(self%pairs))nullify(self%pairs)
    if( associated(self%shifts))nullify(self%shifts)
    if( associated(self%distances))nullify(self%distances)
    if( associated(self%vectors))nullify(self%vectors)
    ! set
    if(c_associated(self%cdata%pairs)) call c_f_pointer(self%cdata%pairs, self%pairs, shape=[2,n])
    if(c_associated(self%cdata%shifts)) call c_f_pointer(self%cdata%shifts, self%shifts, shape=[3,n])
    if(c_associated(self%cdata%distances)) call c_f_pointer(self%cdata%distances, self%distances, shape=[n])
    if(c_associated(self%cdata%vectors)) call c_f_pointer(self%cdata%vectors, self%vectors, shape=[3,n])
  end function vesin_compute_c_size_t_c_double
  function vesin_compute_c_int_c_double( self, nat, pos, box, periodic ) result( ierr )
    implicit none
    class( NeighborList ), intent(inout) :: self
    integer( c_int ), intent(in) :: nat
    real( c_double ), intent(in) :: pos(3,nat)
    real( c_double ), intent(in) :: box(3,3)
    logical, intent(in), optional :: periodic
    integer :: ierr

    logical( c_bool ) :: c_periodic
    type( c_ptr ) :: c_errmsg = c_null_ptr
    integer :: n
    ! self has not been initialized
    if( .not. self% active ) then
       self% errmsg="NeighborList has to be initialized before computing."
       ierr = -1
       return
    end if
    ! periodic by default
    c_periodic = .true.
    if( present(periodic))c_periodic = periodic
    ! call iterface to c, with data in prescribed c precision
    ierr = int( c_vesin_neighbors( &
         real(pos, c_double), &
         int(nat, c_size_t), &
         real(box, c_double), &
         c_periodic, &
         self%device, &
         self%opts, &
         self%cdata, &
         c_errmsg) )
    if( ierr /= 0 ) then
       self% errmsg = c2f_string(c_errmsg)
       return
    end if
    ! cast cdata to f, in the returned C precision
    n = int( self%cdata%length )
    self% length = self%cdata%length
    ! nullify pointers in self
    if( associated(self%pairs))nullify(self%pairs)
    if( associated(self%shifts))nullify(self%shifts)
    if( associated(self%distances))nullify(self%distances)
    if( associated(self%vectors))nullify(self%vectors)
    ! set
    if(c_associated(self%cdata%pairs)) call c_f_pointer(self%cdata%pairs, self%pairs, shape=[2,n])
    if(c_associated(self%cdata%shifts)) call c_f_pointer(self%cdata%shifts, self%shifts, shape=[3,n])
    if(c_associated(self%cdata%distances)) call c_f_pointer(self%cdata%distances, self%distances, shape=[n])
    if(c_associated(self%cdata%vectors)) call c_f_pointer(self%cdata%vectors, self%vectors, shape=[3,n])
  end function vesin_compute_c_int_c_double


  !> @brief Destructor.
  !! @details nullify the fortran pointers to C data, and free the C data.
  subroutine vesin_destroy( self )
    class( NeighborList ), intent(inout) :: self
    if( associated(self%pairs))nullify(self%pairs)
    if( associated(self%shifts))nullify(self%shifts)
    if( associated(self%distances))nullify(self%distances)
    if( associated(self%vectors))nullify(self%vectors)
    call c_vesin_free(self%cdata)
    self% active = .false.
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
       allocate( character(len=n)::f_string)
       do i = 1, n
          f_string(i:i) = c_string(i)
       end do
    end if
  end function c2f_string

end module vesin_wrapper

