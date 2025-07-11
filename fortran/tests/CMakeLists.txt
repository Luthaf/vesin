add_executable(fortran-main main.f90)
target_link_libraries(fortran-main vesin_fortran)

add_test(fortran-main fortran-main)
