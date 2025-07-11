add_library(vesin_fortran_wrapper vesin_wrapper.f90)
target_link_libraries(vesin_fortran_wrapper PUBLIC vesin_fortran)


add_executable(fortran-wrapper wrapper_test.f90)
target_link_libraries(fortran-wrapper vesin_fortran_wrapper)

add_test(fortran-wrapper fortran-wrapper)
