program main
  use vesin_wrapper
  use vesin_wrapper, only: rp=>double, ip=>size_t
  implicit none
  integer(ip) :: nat
  real(rp), allocatable :: pos(:,:)
  real(rp) :: lat(3,3)
  integer :: i
  integer :: ierr
  character(:), allocatable :: errmsg
  type( NeighborList ) :: neigh

  ! number of atoms
  nat = 2_ip
  allocate( pos(1:3, 1:nat) )

  ! positions
  pos(:,1) = [ 0.0_rp, 0.0_rp, 0.0_rp ]
  pos(:,2) = [ 0.0_rp, 1.3_rp, 1.3_rp ]

  ! box
  lat(:,1) = [ 3.2_rp, 0.0_rp, 0.0_rp ]
  lat(:,2) = [ 0.0_rp, 3.2_rp, 0.0_rp ]
  lat(:,3) = [ 0.0_rp, 0.0_rp, 3.2_rp ]

  ! initialize neigh with some options
  neigh = NeighborList( cutoff=4.2_rp, full=.true., sorted=.true. )
  ! compute
  ierr = neigh% compute( nat, pos, lat, periodic=.true. )
  if( ierr /= 0 ) then
     write(*,*) neigh%errmsg
     stop
  end if
  write(*,*) "length:",neigh% length
  do i = 1, neigh% length
     write(*,*) i, ":", neigh% pairs(:,i)
  end do


  ! set a new cutoff to the same neigh allocation
  call neigh% options( cutoff=2.4_rp, return_distances=.true. )

  ! compute
  ierr = neigh% compute( nat, pos, lat, periodic=.true. )
  if( ierr /= 0 ) then
     write(*,*) neigh%errmsg
     stop
  end if
  write(*,*) "length:",neigh% length
  do i = 1, neigh% length
     write(*,*) i, ":", neigh%pairs(:,i), neigh% distances(i)
  end do

  call neigh% free()
  deallocate( pos )

end program main
