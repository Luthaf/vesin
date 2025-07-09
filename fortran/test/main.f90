program main
  use vesin, only: vesin_t
  use, intrinsic :: iso_c_binding
  implicit none
  type( vesin_t ), pointer :: neigh
  integer( c_size_t ), parameter :: nat=2
  real(c_double) :: pos(3,nat)
  real(c_double) :: box(3,3)
  integer(c_int) :: ierr

  ! positions
  pos(:,1) = [ 0.0_c_double, 0.0_c_double, 0.0_c_double ]
  pos(:,2) = [ 0.0_c_double, 1.3_c_double, 1.3_c_double ]

  ! lattice
  box(:,1) = [ 3.2_c_double, 0.0_c_double, 0.0_c_double ]
  box(:,2) = [ 0.0_c_double, 3.2_c_double, 0.0_c_double ]
  box(:,3) = [ 0.0_c_double, 0.0_c_double, 3.2_c_double ]

  ! create the instance, set options
  neigh => vesin_t( cutoff=4.2_c_double, full=.true., return_shifts=.true., return_distances=.true. )

  ! launch computation of neighbor list
  ierr = neigh% compute( nat, pos, box )
  if( ierr/= 0 ) then
     write(*,*) neigh% errmsg
     stop
  end if

  ! data is inside `neigh`:
  write(*,*) "got length:", neigh% length

  ! destroy data
  deallocate( neigh )

end program main

