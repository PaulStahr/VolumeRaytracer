#ifndef TUPLE_DECLARATION_H
#define TUPLE_DECLARATION_H

#include "tuple_declaration.h"

template <typename T, uint8_t dim>void CUDAONLY(__host__ __device__) print(cuda_tuple<T, dim> const & tuple);
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<float, 1> const & tuple){printf("(%f)",tuple.x);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<float, 2> const & tuple){printf("(%f %f)",tuple.x, tuple.y);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<float, 3> const & tuple){printf("(%f %f %f)",tuple.x, tuple.y, tuple.z);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<float, 4> const & tuple){printf("(%f %f %f %f)",tuple.x, tuple.y, tuple.z, tuple.w);}

template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<int, 1> const & tuple){printf("(%d)",tuple.x);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<int, 2> const & tuple){printf("(%d %d)",tuple.x, tuple.y);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<int, 3> const & tuple){printf("(%d %d %d)",tuple.x, tuple.y,tuple.z);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<int, 4> const & tuple){printf("(%d %d %d %d)",tuple.x, tuple.y, tuple.z, tuple.w);}

template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<short, 1> const & tuple){printf("(%d)",tuple.x);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<short, 2> const & tuple){printf("(%d %d)",tuple.x, tuple.y);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<short, 3> const & tuple){printf("(%d %d %d)",tuple.x, tuple.y,tuple.z);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<short, 4> const & tuple){printf("(%d %d %d %d)",tuple.x, tuple.y, tuple.z, tuple.w);}

template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<unsigned short, 1> const & tuple){printf("(%u)",tuple.x);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<unsigned short, 2> const & tuple){printf("(%u %u)",tuple.x, tuple.y);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<unsigned short, 3> const & tuple){printf("(%u %u %u)",tuple.x, tuple.y, tuple.z);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<unsigned short, 4> const & tuple){printf("(%u %u %u %u)",tuple.x, tuple.y, tuple.z,tuple.w);}

template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<unsigned int, 1> const & tuple){printf("(%u)",tuple.x);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<unsigned int, 2> const & tuple){printf("(%u %u)",tuple.x, tuple.y);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<unsigned int, 3> const & tuple){printf("(%u %u %u)",tuple.x, tuple.y, tuple.z);}
template <>void CUDAONLY(__host__ __device__) print(cuda_tuple<unsigned int, 4> const & tuple){printf("(%u %u %u %u)",tuple.x, tuple.y, tuple.z, tuple.w);}

#endif
