#ifndef tuple_declaration
#define tuple_declaration

#ifdef NCUDA
#define __host__
#define __device__
#define __global__
#define CUDAONLY(a)
#define NCUDAONLY(a) (a)
#else
#define CUDAONLY(a) (a)
#define NCUDAONLY(a)
#endif

template<typename T, uint8_t n>struct cuda_tuple{};
template<typename T>struct cuda_tuple<T,1>{T x;
inline  T & operator[](uint8_t idx){if(idx == 0) return x;*(int*)0=0;return x;}
};
template<typename T>struct cuda_tuple<T,2>{T x,y;
inline T & operator[](uint8_t idx){if(idx == 0) return x; if (idx == 1) return y;*(int*)0=0;return x;}
};
template<typename T>struct cuda_tuple<T,3>{T x,y,z;
inline T & operator[](uint8_t idx){if(idx == 0) return x; if (idx == 1) return y; if (idx == 2) return z;*(int*)0=0;return x;}
};
template<typename T>struct cuda_tuple<T,4>{T x,y,z,w;
inline T & operator[](uint8_t idx){if(idx == 0) return x; if (idx == 1) return y; if (idx == 2) return z; if (idx == 3) return w;*(int*)0=0;return x;}
};




#endif
