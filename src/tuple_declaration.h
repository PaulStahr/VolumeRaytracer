#ifndef tuple_declaration
#define tuple_declaration

template<typename T, uint8_t n>struct cuda_tuple{};
template<typename T>struct cuda_tuple<T,1>{T x;};
template<typename T>struct cuda_tuple<T,2>{T x,y;};
template<typename T>struct cuda_tuple<T,3>{T x,y,z;};
template<typename T>struct cuda_tuple<T,4>{T x,y,z,w;};

#endif
