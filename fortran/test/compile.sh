if [ ! -e ../../vesin-single-build.cpp ]; then cd ../../ && python3 create-single-cpp.py && cd -; fi
g++ -I../../vesin/include -J./ -c ../../vesin-single-build.cpp
gfortran -g -c ../vesin.f90
gfortran -g -I./ -o main.x main.f90 vesin-single-build.o vesin.o -lstdc++
