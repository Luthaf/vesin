program main
  use f_vesin_wrapper, only: vesin_t, rp
  implicit none
  type( vesin_t ), pointer :: neigh
  integer, parameter :: nat=2
  real(rp) :: pos(3,nat)
  real(rp) :: box(3,3)
  integer :: ierr

  ! positions
  pos(:,1) = [ 0.0_rp, 0.0_rp, 0.0_rp ]
  pos(:,2) = [ 0.0_rp, 1.3_rp, 1.3_rp ]

  ! lattice
  box(:,1) = [ 3.2_rp, 0.0_rp, 0.0_rp ]
  box(:,2) = [ 0.0_rp, 3.2_rp, 0.0_rp ]
  box(:,3) = [ 0.0_rp, 0.0_rp, 3.2_rp ]

  ! create the instance, set options
  neigh => vesin_t( cutoff=4.2_rp, full=.true., return_shifts=.true., return_distances=.true. )

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

