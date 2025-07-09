g++ -I../../vesin/include -J./ -c ../../vesin-single-build.cpp
gfortran -c ../f_vesin_wrapper.f90
gfortran -I./ -o main.x main.f90 vesin-single-build.o f_vesin_wrapper.o -lstdc++
