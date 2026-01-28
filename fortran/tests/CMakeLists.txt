add_executable(fortran-tests tests.f90)
target_link_libraries(fortran-tests vesin_fortran)

set_target_properties(fortran-tests PROPERTIES LINKER_LANGUAGE Fortran)

add_test(fortran-tests fortran-tests)
