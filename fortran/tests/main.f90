program main

  use vesin
  use, intrinsic :: iso_c_binding
  implicit none

  type( VesinOptions ) :: options
  type( VesinNeighborList ) :: neigh
  integer(c_int) :: device
  integer( c_size_t ), parameter :: nat=2
  real(c_double) :: pos(3,nat)
  real(c_double) :: box(3,3)
  integer(c_int) :: ierr
  logical(c_bool) :: periodic
  type( c_ptr ) :: errmsg

  ! positions
  pos(:,1) = [ 0.0_c_double, 0.0_c_double, 0.0_c_double ]
  pos(:,2) = [ 0.0_c_double, 1.3_c_double, 1.3_c_double ]

  ! lattice
  box(:,1) = [ 3.2_c_double, 0.0_c_double, 0.0_c_double ]
  box(:,2) = [ 0.0_c_double, 3.2_c_double, 0.0_c_double ]
  box(:,3) = [ 0.0_c_double, 0.0_c_double, 3.2_c_double ]

  ! options
  options% return_vectors = .true.
  options% full = .true.

  ! periodic calc
  periodic = .true.

  ! compute neighbors
  options% cutoff = 4.2_c_double
  device=VesinCPU
  ierr = vesin_neighbors( pos, nat, box, periodic, device, options, neigh, errmsg )
  write(*,*) neigh% length
  if(ierr /= 0_c_int ) write(*,*) c2f_string( errmsg )


  ! change cutoff
  options% cutoff = 2.4_c_double

  ! compute neighbors
  ierr = vesin_neighbors( pos, nat, box, periodic, device, options, neigh, errmsg )
  write(*,*) neigh% length
  if(ierr /= 0_c_int ) write(*,*) c2f_string( errmsg )


  call vesin_free( neigh )


contains

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

end program main

