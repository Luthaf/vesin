add_executable(fortran-tests tests.f90)
target_link_libraries(fortran-tests vesin_fortran)

add_test(fortran-tests fortran-tests)
