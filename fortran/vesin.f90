!> @brief
!!
!! This is a Fortran interface to the C-API of `Vesin`, from the header `vesin.h`.
!!
!! @details
!! This module only defines interfacees which are C-interoperable, therefore keeping
!! the variable precision as defined by C. No memory is allocated here. The returned
!! data still needs to be processed (casted to proper fortran data).
!!
!! Example program
!!~~~~~~~~~~~~{.f90}
!! program main
!!   use vesin
!!   use, intrinsic :: iso_c_binding
!!   implicit none
!!   type( VesinOptions ) :: options
!!   type( VesinNeighborList ) :: neigh
!!   type( VesinDevice ) :: device
!!   integer( c_size_t ), parameter :: nat=2
!!   real(c_double) :: pos(3,nat)
!!   real(c_double) :: box(3,3)
!!   integer(c_int) :: ierr
!!   logical(c_bool) :: periodic
!!   type( c_ptr ) :: errmsg
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
!!   ! options
!!   options% cutoff = 4.2_c_double
!!   options% return_vectors = .true.
!!   options% full = .true.
!!
!!   ! periodic calc
!!   periodic = .true.
!!
!!   ! compute neighbors: returned data in `neigh`
!!   ierr = vesin_neighbors( pos, nat, box, periodic, device%vesincpu, options, neigh, errmsg )
!!
!!   if( ierr /= 0_c_int ) write(*,*) c2f_string( errmsg )
!!   write(*,*) "length:", neigh% length
!!
!!   ! cast data from `neigh` ... see docs of `type(VesinNeighborList)`
!!
!!   ! free the C-memory
!!   call vesin_free( neigh )
!!
!! end program main
!!~~~~~~~~~~~~
!!
!! @author Miha Gunde
!!
module vesin

  use, intrinsic :: iso_c_binding
  implicit none

  private
  public :: VesinDevice
  public :: VesinOptions
  public :: VesinNeighborList
  public :: vesin_neighbors
  public :: vesin_free


  !> @brief Device on which the data can be
  type, bind(c) :: VesinDevice

     integer( c_int ) :: VesinUnknownDevice = 0
     !< Unknown device, used for default initialization and to indicate no
     !! allocated data.

     integer( c_int ) :: VesinCPU = 1
     !< CPU device

  end type VesinDevice


  !> @brief
  !! Used for storing Vesin options.
  !!
  !! @details
  !! Equivalent to:
  !!~~~~~~~~~{.c}
  !! struct VesinOptions {}
  !!~~~~~~~~~
  !!
  type, bind(c) ::  VesinOptions

     real( c_double ) :: cutoff = 0.0_c_double
     !< Spherical cutoff, only pairs below this cutoff will be included

     logical( c_bool ) :: full = .false.
     !< Should the returned neighbor list be a full list (include both `i -> j`
     !! and `j -> i` pairs) or a half list (include only `i -> j`)?

     logical( c_bool ) :: sorted = .true.
     !< Should the neighbor list be sorted? If yes, the returned pairs will be
     !! sorted using lexicographic order.

     logical( c_bool ) :: return_shifts = .false.
     !< Should the returned `VesinNeighborList` contain `shifts`?

     logical( c_bool ) :: return_distances = .false.
     !< Should the returned `VesinNeighborList` contain `distances`?

     logical( c_bool ) :: return_vectors = .false.
     !< Should the returned `VesinNeighborList` contain `vector`?

  end type VesinOptions


  !> @brief
  !! Used as return type from `vesin_neighbors()`.
  !!
  !! @details Returned data are `type(c_ptr)` pointers to
  !! the memory allocated by C. They need to be transformed into fortran pointers in
  !! order to read the values.
  !!
  !! Equvalent to:
  !!~~~~~~~~~{.c}
  !! struct VESIN_API VesinNeighborList {}
  !!~~~~~~~~~
  !!
  type, bind(c) :: VesinNeighborList

     integer( c_size_t ) :: length = 0_c_size_t
     !< Number of pairs in this neighbor list
     !!
     !! C-declaration:
     !!~~~~~~~~~{.c}
     !! size_t length;
     !!~~~~~~~~~

     integer( c_int ) :: device = 0_c_int
     !< Device used for the data allocations
     !!
     !! C-declaration:
     !!~~~~~~~~~{.c}
     !! VesinDevice device;
     !!~~~~~~~~~

     type( c_ptr ) :: pairs = c_null_ptr
     !< Array of pairs (storing the indices of the first and second point in the
     !! pair), containing `length` elements.
     !!
     !! C-declaration:
     !!~~~~~~~~~{.c}
     !! size_t (*pairs)[2];
     !!~~~~~~~~~
     !! cast to fortran array as:
     !!~~~~~~~~~{.f90}
     !! integer( c_size_t ), pointer :: f_pairs(:) => null()
     !! if(c_associated(self%pairs)) call c_f_pointer( self%pairs, f_pairs, shape=[2, self%length] )
     !!~~~~~~~~~

     type( c_ptr ) :: shifts = c_null_ptr
     !< Array of box shifts, one for each `pair`. This is only set if
     !! `options.return_pairs` was `true` during the calculation.
     !!
     !! C-declaration:
     !!~~~~~~~~~{.c}
     !! int32_t (*shifts)[3];
     !!~~~~~~~~~
     !! cast to fortran array as:
     !!~~~~~~~~~{.f90}
     !! integer( c_int32_t ), pointer :: f_shifts(:,:) => null()
     !! if(c_associated(self%shifts)) call c_f_pointer( self%shifts, f_shifts, shape=[3, self%length] )
     !!~~~~~~~~~

     type( c_ptr ) :: distances = c_null_ptr
     !< Array of pair distance (i.e. distance between the two points), one for
     !! each pair. This is only set if `options.return_distances` was `true`
     !! during the calculation.
     !!
     !! C-declaration:
     !!~~~~~~~~~{.c}
     !! double *distances;
     !!~~~~~~~~~
     !! cast to fortran array as:
     !!~~~~~~~~~{.f90}
     !! real( c_double ), pointer :: f_distances(:) => null()
     !! if(c_associated(self%distances)) call c_f_pointer( self%distances, f_distances, shape=[self%length] )
     !!~~~~~~~~~

     type( c_ptr ) :: vectors = c_null_ptr
     !< Array of pair vector (i.e. vector between the two points), one for
     !! each pair. This is only set if `options.return_vector` was `true`
     !! during the calculation.
     !!
     !! C declaration:
     !!~~~~~~~~~{.c}
     !! double (*vectors)[3];
     !!~~~~~~~~~
     !! cast to fortran array as:
     !!~~~~~~~~~{.f90}
     !! real( c_double ), pointer :: f_distances(:,:) => null()
     !! if(c_associated(self%vectors)) call c_f_pointer( self%vectors, f_vectors, shape=[3, self%length] )
     !!~~~~~~~~~

  end type VesinNeighborList


  !> @brief
  !! Compute a neighbor list.
  !!
  !! @details
  !!
  !! The data is returned in a `VesinNeighborList`. For an initial call, the
  !! `VesinNeighborList` should be zero-initialized (or default-initalized in
  !! fortran). The `VesinNeighborList` can be re-used across calls to this functions
  !! to re-use memory allocations, and once it is no longer needed, users should
  !! call `vesin_free` to release the corresponding memory.
  !!
  !! @param points positions of all points in the system;
  !! @param n_points number of elements in the `points` array
  !! @param box bounding box for the system. If the system is non-periodic,
  !!     this is ignored. This should contain the three vectors of the bounding
  !!     box, one vector per row of the matrix.
  !! @param periodic is the system using periodic boundary conditions?
  !! @param device device where the `points` and `box` data is allocated.
  !! @param options options for the calculation
  !! @param neighbors a `type(VesinNeighborList)` instance, that will be used
  !!     to store the computed list of neighbors.
  !! @param error_message a `type(c_ptr)` to a null-terminated `char*` containing the error
  !!     message of this function. Cast to:
  !!~~~~~{.f90}
  !!character(len=1, kind=c_char), pointer :: c_string(:)
  !!~~~~~
  !!     then copy values to a fortran string.
  !! @returns status nonzero integer upon error; zero otherwise.
  !!
  !! C-header:
  !!
  !!~~~~~~~~~~~~~~~~~{.c}
  !! int VESIN_API vesin_neighbors(
  !!     const double (*points)[3],
  !!     size_t n_points,
  !!     const double box[3][3],
  !!     bool periodic,
  !!     VesinDevice device,
  !!     struct VesinOptions options,
  !!     struct VesinNeighborList* neighbors,
  !!     const char** error_message
  !! );
  !!~~~~~~~~~~~~~~~~~
  interface
     function vesin_neighbors( &
          points,        &
          n_points,      &
          box,           &
          periodic,      &
          device,        &
          options,       &
          neighbors,     &
          error_message ) result( status ) bind( c, name="vesin_neighbors" )
       import :: c_double, c_size_t, c_bool, c_ptr, c_int, VesinOptions, VesinNeighborList
       integer( c_size_t ), value   :: n_points
       real( c_double ), intent(in) :: points(3, n_points)
       real( c_double ), intent(in) :: box(3,3)
       logical( c_bool ),    value  :: periodic
       integer(c_int),       value  :: device
       type( VesinOptions ), value  :: options
       type( VesinNeighborList )    :: neighbors
       type( c_ptr ),    intent(in) :: error_message
       integer( c_int )             :: status
     end function vesin_neighbors
  end interface


  !> @brief
  !! Free all allocated memory inside a `VesinNeighborList`, according to it's
  !! `device`.
  !!
  !! @details
  !! C-header:
  !!
  !!~~~~~~~~~~~~{.c}
  !! void VESIN_API vesin_free(struct VesinNeighborList* neighbors);
  !!~~~~~~~~~~~~
  interface
     subroutine vesin_free( neighbors ) bind(C, name="vesin_free")
       import :: VesinNeighborList
       type( VesinNeighborList ) :: neighbors
     end subroutine vesin_free
  end interface

contains

end module vesin
