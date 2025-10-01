---
project: vesin
preprocess: false
project_github: https://github.com/luthaf/vesin
project_website: https://luthaf.fr/vesin/
css: docs.css
---

This is the API documentation for the Fortran bindings to vesin.

We provide two modules: [[vesin]] which is the high-level interface and
[[vesin_c]] which is a direct translation of the C API. For the high-level
interface, everything is handled by the [[NeighborList]] type, which holds
options, runs the calculations and holds the data afterward.

#### Example usage

```fortran
program main
    use vesin, only: NeighborList

    implicit none

    real, allocatable :: positions(:,:)
    real :: box(3,3)
    integer :: i, ierr
    type(NeighborList) :: neighbor_list

    ! we have 2000 points in this example
    allocate(positions(3, 2000))

    ! set the values for points `positions` and bounding `box` here
    ! positions = ...
    ! box = ...

    ! initialize `neighbor_list` with some options
    neighbor_list = NeighborList(cutoff=4.2, full=.true., sorted=.true.)

    ! run the calculation
    call neighbor_list%compute(positions, box, periodic=.true., status=ierr)
    if (ierr /= 0) then
        write(*, *) neighbor_list%errmsg
        stop
    end if

    write(*,*) "we got ", neighbor_list%length, "pairs"
    do i=1,neighbor_list%length
        write(*, *) " - ", i, ":", neighbor_list%pairs(:, i)
    end do

    ! release allocated memory
    call neighbor_list%free()
    deallocate(positions)
end program main
```
