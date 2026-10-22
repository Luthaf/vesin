#ifndef VESIN_CUDA_INTEGERS_CUH
#define VESIN_CUDA_INTEGERS_CUH

typedef short int16;
typedef unsigned short uint16;
static_assert(sizeof(int16) == 2);
static_assert(sizeof(uint16) == 2);

typedef int int32;
typedef unsigned int uint32;
static_assert(sizeof(int32) == 4);
static_assert(sizeof(uint32) == 4);

typedef long long int64;
typedef unsigned long long uint64;
static_assert(sizeof(int64) == 8);
static_assert(sizeof(uint64) == 8);

#endif // VESIN_CUDA_INTEGERS_CUH
