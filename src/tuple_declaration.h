#ifndef tuple_declaration
#define tuple_declaration

template<typename T, uint8_t n>struct cuda_tuple{};
template<typename T>struct cuda_tuple<T,1>{T x;
inline __host__ __device__ T & operator[](uint8_t idx){if(idx == 0) return x;*(int*)0=0;return x;}
};
template<typename T>struct cuda_tuple<T,2>{T x,y;
inline __host__ __device__ T & operator[](uint8_t idx){if(idx == 0) return x; if (idx == 1) return y;*(int*)0=0;return x;}
};
template<typename T>struct cuda_tuple<T,3>{T x,y,z;
inline __host__ __device__ T & operator[](uint8_t idx){if(idx == 0) return x; if (idx == 1) return y; if (idx == 2) return z;*(int*)0=0;return x;}
};
template<typename T>struct cuda_tuple<T,4>{T x,y,z,w;
inline __host__ __device__ T & operator[](uint8_t idx){if(idx == 0) return x; if (idx == 1) return y; if (idx == 2) return z; if (idx == 3) return w;*(int*)0=0;return x;}
};




#endif
