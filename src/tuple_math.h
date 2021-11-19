/*
Copyright (c) 2018 Paul Stahr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef tuple_math
#define tuple_math

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "tuple_declaration.h"

#ifdef NCUDA
#include <cmath>
#endif

/*#include <xmmintrin.h>

inline __host__ __device__  void add(float4 &a, float4 b)
{
    #ifdef __CUDA_ARCH__
    a += b;
    #else
    *reinterpret_cast<__m128*>(&a) = _mm_add_ps(*reinterpret_cast<__m128*>(&a),*reinterpret_cast<__m128*>(&b));
    a = *reinterpret_cast<float4*>(&erg);
    a+=b;
    #endif
}*/

template <typename T, uint8_t n>
struct make_struct
{
    make_struct(){}

    template <typename V, uint8_t m>
    inline __host__ __device__  cuda_tuple<T,n> operator()(cuda_tuple<V, m> a) const{}
};

template <typename T>
struct make_struct<T,1>
{
    template <typename V, uint8_t m>inline __host__ __device__  cuda_tuple<T,1> operator()(cuda_tuple<V, m> a) const{}
    template <typename V>inline __host__ __device__  cuda_tuple<T,1> operator()(cuda_tuple<V, 1> a) const{cuda_tuple<T,1> res;res.x = a.x;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,1> operator()(V x)                const{cuda_tuple<T,1> res;res.x = x;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,1> operator()(cuda_tuple<V, 2> a) const{cuda_tuple<T,1> res;res.x = a.x;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,1> operator()(cuda_tuple<V, 3> a) const{cuda_tuple<T,1> res;res.x = a.x;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,1> operator()(cuda_tuple<V, 4> a) const{cuda_tuple<T,1> res;res.x = a.x;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,1> operator()(V* a)               const{cuda_tuple<T,1> res;res.x = a[0];return res;}
};

template <typename T>
struct make_struct<T,2>
{
    template <typename V, uint8_t m>
    inline __host__ __device__  cuda_tuple<T,2> operator()(cuda_tuple<V, m> a) const{}

    template <typename V>inline __host__ __device__  cuda_tuple<T,2> operator()(cuda_tuple<V, 1> a) const{cuda_tuple<T,2> res;res.x = a.x;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,2> operator()(cuda_tuple<V, 2> a) const{cuda_tuple<T,2> res;res.x = a.x;res.y = a.y;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T, 2> operator()(V x, V y) const{cuda_tuple<T, 2> res;res.x = x;res.y = y;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,2> operator()(cuda_tuple<V, 3> a) const{cuda_tuple<T,2> res;res.x = a.x;res.y = a.y;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,2> operator()(cuda_tuple<V, 4> a) const{cuda_tuple<T,2> res;res.x = a.x;res.y = a.y;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,2> operator()(V* a) const{cuda_tuple<T,2> res;res.x = a[0];res.y = a[1];return res;}
};

template <typename T>
struct make_struct<T,3>
{
    template <typename V, uint8_t m>inline __host__ __device__  cuda_tuple<T,3> operator()(cuda_tuple<V, m> a) const{}
    template <typename V>inline __host__ __device__  cuda_tuple<T,3> operator()(cuda_tuple<V, 1> a) const{cuda_tuple<T,3> res;res.x = a.x;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,3> operator()(cuda_tuple<V, 2> a) const{cuda_tuple<T,3> res;res.x = a.x;res.y = a.y;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T, 3> operator()(V x, V y, V z) const{cuda_tuple<T, 3> res;res.x = x;res.y = y;res.z = z;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,3> operator()(cuda_tuple<V, 3> a) const{cuda_tuple<T,3> res;res.x = a.x;res.y = a.y;res.z = a.z;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,3> operator()(cuda_tuple<V, 4> a) const{cuda_tuple<T,3> res;res.x = a.x;res.y = a.y;res.z = a.z;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,3> operator()(V* a) const{cuda_tuple<T,3> res;res.x = a[0];res.y = a[1];res.z = a[2];return res;}
};


template <typename T>
struct make_struct<T,4>
{
    template <typename V, uint8_t m>inline __host__ __device__  cuda_tuple<T,4> operator()(cuda_tuple<V, m> a) const{}
    template <typename V>inline __host__ __device__  cuda_tuple<T,4> operator()(cuda_tuple<V, 1> a) const{cuda_tuple<T,4> res;res.x = a.x;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,4> operator()(cuda_tuple<V, 2> a) const{cuda_tuple<T,1> res;res.x = a.x;res.y = a.y;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,4> operator()(cuda_tuple<V, 3> a) const{cuda_tuple<T,4> res;res.x = a.x;res.y = a.y;res.z = a.z;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T, 4> operator()(V x, V y, V z, V w) const{cuda_tuple<T, 4> res;res.x = x;res.y = y;res.z = z;res.w = w;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,4> operator()(cuda_tuple<V, 4> a) const{cuda_tuple<T,4> res;res.x = a.x;res.y = a.y;res.z = a.z;res.w = a.w;return res;}
    template <typename V>inline __host__ __device__  cuda_tuple<T,4> operator()(V* a) const{cuda_tuple<T,4> res;res.x = a[0];res.y = a[1];res.z = a[2];res.w = a[3];return res;}
};

template <typename T, typename V>inline __host__ __device__  void operator/=(cuda_tuple<T, 1> &a, V s){a.x /= s;}
template <typename T, typename V>inline __host__ __device__  void operator/=(cuda_tuple<T, 2> &a, V s){a.x /= s;a.y /= s;}
template <typename T, typename V>inline __host__ __device__  void operator/=(cuda_tuple<T, 3> &a, V s){a.x /= s;a.y /= s;a.z /= s;}
template <typename T, typename V>inline __host__ __device__  void operator/=(cuda_tuple<T, 4> &a, V s){a.x /= s;a.y /= s;a.z /= s;a.w /= s;}

template <typename T, uint8_t dim>inline __host__ __device__  T & get(cuda_tuple<T, dim> &a, uint8_t index){return reinterpret_cast<T*>(&a)[index];}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 1> &a, cuda_tuple<V, 1> b){a.x *= b.x;}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 2> &a, cuda_tuple<V, 2> b){a.x *= b.x;a.y *= b.y;}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 3> &a, cuda_tuple<V, 2> b){a.x *= b.x;a.y *= b.y;}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 3> &a, cuda_tuple<V, 3> b){a.x *= b.x;a.y *= b.y;a.z *= b.z;}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 4> &a, cuda_tuple<V, 3> b){a.x *= b.x;a.y *= b.y;a.z *= b.z;}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 4> &a, cuda_tuple<V, 4> b){a.x *= b.x;a.y *= b.y;a.z *= b.z;a.w *= b.w;}

template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 1> &a, V s){a.x *= s;}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 2> &a, V s){a.x *= s;a.y *= s;}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 3> &a, V s){a.x *= s;a.y *= s;a.z *= s;}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<T, 4> &a, V s){a.x *= s;a.y *= s;a.z *= s;a.w *= s;}

template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 1> &a, cuda_tuple<V, 1> b){a.x += b.x;}
template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 2> &a, cuda_tuple<V, 2> b){a.x += b.x;a.y += b.y;}
template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 2> &a, cuda_tuple<V, 3> b){a.x += b.x;a.y += b.y;}
template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 3> &a, cuda_tuple<V, 3> b){a.x += b.x;a.y += b.y;a.z += b.z;}
template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 3> &a, cuda_tuple<V, 4> b){a.x += b.x;a.y += b.y;a.z += b.z;}
template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 4> &a, cuda_tuple<V, 4> b){a.x += b.x;a.y += b.y;a.z += b.z;a.w += b.w;}

#ifdef __AVX2__
__m128 & to_sse(cuda_tuple<float, 4> & a){return *reinterpret_cast<__m128*>(&a);}

template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<float, 4> &a, cuda_tuple<float, 4> b){to_sse(a) = _mm_add_ps (to_sse(a), to_sse(b));}
template <typename T, typename V>inline __host__ __device__  void operator-=(cuda_tuple<float, 4> &a, cuda_tuple<float, 4> b){to_sse(a) = _mm_sub_ps (to_sse(a), to_sse(b));}
template <typename T, typename V>inline __host__ __device__  void operator*=(cuda_tuple<float, 4> &a, cuda_tuple<float, 4> b){to_sse(a) = _mm_mul_ps (to_sse(a), to_sse(b));}
template <typename T, typename V>inline __host__ __device__  void operator/=(cuda_tuple<float, 4> &a, cuda_tuple<float, 4> b){to_sse(a) = _mm_div_ps (to_sse(a), to_sse(b));}
#endif

template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 1> &a, V s){a.x += s;}
template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 2> &a, V s){a.x += s;a.y += s;}
template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 3> &a, V s){a.x += s;a.y += s;a.z += s;}
template <typename T, typename V>inline __host__ __device__  void operator+=(cuda_tuple<T, 4> &a, V s){a.x += s;a.y += s;a.z += s;a.w += s;}

template <typename T, typename V>inline __host__ __device__  void operator>>=(cuda_tuple<T, 1> &a, V s){a.x >>= s;}
template <typename T, typename V>inline __host__ __device__  void operator>>=(cuda_tuple<T, 2> &a, V s){a.x >>= s;a.y >>= s;}
template <typename T, typename V>inline __host__ __device__  void operator>>=(cuda_tuple<T, 3> &a, V s){a.x >>= s;a.y >>= s;a.z >>= s;}
template <typename T, typename V>inline __host__ __device__  void operator>>=(cuda_tuple<T, 4> &a, V s){a.x >>= s;a.y >>= s;a.z >>= s;a.w >>= s;}

template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,1> operator>>(cuda_tuple<T, 1> a, V s){return make_struct<T,1>()(a.x >> s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,2> operator>>(cuda_tuple<T, 2> a, V s){return make_struct<T,2>()(a.x >> s, a.y >> s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,3> operator>>(cuda_tuple<T, 3> a, V s){return make_struct<T,3>()(a.x >> s, a.y >> s, a.z >> s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,4> operator>>(cuda_tuple<T, 4> a, V s){return make_struct<T,4>()(a.x >> s, a.y >> s, a.z >> s, a.w >> s);}

template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,1> operator-(cuda_tuple<T, 1> a, V s){return make_struct<T,1>()(a.x - s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,2> operator-(cuda_tuple<T, 2> a, V s){return make_struct<T,2>()(a.x - s, a.y - s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,3> operator-(cuda_tuple<T, 3> a, V s){return make_struct<T,3>()(a.x - s, a.y - s, a.z - s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,4> operator-(cuda_tuple<T, 4> a, V s){return make_struct<T,4>()(a.x - s, a.y - s, a.z - s, a.w - s);}

template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,1> operator*(cuda_tuple<T, 1> a, cuda_tuple<V, 1> b){return make_struct<T,1>()(a.x * b.x);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,2> operator*(cuda_tuple<T, 2> a, cuda_tuple<V, 2>  b){return make_struct<T,2>()(a.x * b.x, a.y * b.y);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,3> operator*(cuda_tuple<T, 3> a, cuda_tuple<V, 3>  b){return make_struct<T,3>()(a.x * b.x, a.y * b.y, a.z * b.z);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,4> operator*(cuda_tuple<T, 4> a, cuda_tuple<V, 4>  b){return make_struct<T,4>()(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);}

template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,1> operator*(cuda_tuple<T, 1> a, V s){return make_struct<T,1>()(a.x * s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,2> operator*(cuda_tuple<T, 2> a, V s){return make_struct<T,2>()(a.x * s, a.y * s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,3> operator*(cuda_tuple<T, 3> a, V s){return make_struct<T,3>()(a.x * s, a.y * s, a.z * s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,4> operator*(cuda_tuple<T, 4> a, V s){return make_struct<T,4>()(a.x * s, a.y * s, a.z * s, a.w * s);}

template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,1> operator+(cuda_tuple<T, 1> a, V s){return make_struct<T,1>()(a.x + s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,2> operator+(cuda_tuple<T, 2> a, V s){return make_struct<T,2>()(a.x + s, a.y + s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,3> operator+(cuda_tuple<T, 3> a, V s){return make_struct<T,3>()(a.x + s, a.y + s, a.z + s);}
template <typename T, typename V>inline __host__ __device__  cuda_tuple<T,4> operator+(cuda_tuple<T, 4> a, V s){return make_struct<T,4>()(a.x + s, a.y + s, a.z + s, a.w + s);}

template <typename T, typename V>inline __host__ __device__  bool operator<(cuda_tuple<T, 1> a, cuda_tuple<V,1> b){return a.x < b.x;}
template <typename T, typename V>inline __host__ __device__  bool operator<(cuda_tuple<T, 2> a, cuda_tuple<V,2> b){return a.x < b.x && a.y < b.y;}
template <typename T, typename V>inline __host__ __device__  bool operator<(cuda_tuple<T, 3> a, cuda_tuple<V,3> b){return a.x < b.x && a.y < b.y && a.z < b.z;}
template <typename T, typename V>inline __host__ __device__  bool operator<(cuda_tuple<T, 4> a, cuda_tuple<V,4> b){return a.x < b.x && a.y < b.y && a.z < b.z && a.w < b.w;}

template <typename T>inline __host__ __device__  T prod(cuda_tuple<T, 1> a){return a.x;}
template <typename T>inline __host__ __device__  T prod(cuda_tuple<T, 2> a){return a.x * a.y;}
template <typename T>inline __host__ __device__  T prod(cuda_tuple<T, 3> a){return a.x * a.y * a.z;}
template <typename T>inline __host__ __device__  T prod(cuda_tuple<T, 4> a){return a.x * a.y * a.z * a.w;}

template <typename T>inline __host__ __device__  T dot(cuda_tuple<T, 1> a, cuda_tuple<T,1> b){return a.x * b.x;}
template <typename T>inline __host__ __device__  T dot(cuda_tuple<T, 2> a, cuda_tuple<T,2> b){return a.x * b.x + a.y * b.y;}
template <typename T>inline __host__ __device__  T dot(cuda_tuple<T, 3> a, cuda_tuple<T,3> b){return a.x * b.x + a.y * b.y + a.z * b.z;}
template <typename T>inline __host__ __device__  T dot(cuda_tuple<T, 4> a, cuda_tuple<T,4> b){return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;}

inline __host__ int32_t __float2int_rn(float x){return static_cast<int32_t>(std::round(x));}

inline __host__ __device__  cuda_tuple<int, 4> __float2int_rn(cuda_tuple<float, 4> a)
{
    cuda_tuple<int,4> res;
    res.x = __float2int_rn(a.x);
    res.y = __float2int_rn(a.y);
    res.z = __float2int_rn(a.z);
    res.w = __float2int_rn(a.w);
    return res;
}

inline __host__ __device__  cuda_tuple<int, 3> __float2int_rn2(cuda_tuple<float, 3> a)
{
    cuda_tuple<int,3> res;
    res.x = __float2int_rn(a.x);
    res.y = __float2int_rn(a.y);
    res.z = __float2int_rn(a.z);
    return res;
}


inline __host__ __device__  cuda_tuple<int, 2> __float2int_rn2(cuda_tuple<float, 2> a)
{
    cuda_tuple<int,2> res;
    res.x = __float2int_rn(a.x);
    res.y = __float2int_rn(a.y);
    return res;
}

template <uint8_t n>
inline __host__ __device__  void add(cuda_tuple<float,n> & left, cuda_tuple<float, n> & right, float multl, float multr)
{
    left *= multl;
    right *= multr;
    left += right;
}

template <uint8_t n>
inline __host__ __device__  void add(cuda_tuple<float,n> & left, cuda_tuple<float, n> & right, uint16_t multl, uint16_t multr)
{
    left *= multl;
    right *= multr;
    left += right;
}

template <uint8_t n>
inline __host__ __device__  void add(cuda_tuple<float,n> & left, cuda_tuple<float, n> & right, float alpha)
{
    left *= alpha;
    alpha = 1 - alpha;
    right *= alpha;
    left += right;
}

inline __host__ int32_t __mulhi(int32_t x, int32_t y){return (static_cast<int64_t>(x) * static_cast<int64_t>(y)) >> 32;}
inline __host__ uint32_t __mulhi(uint32_t x, uint32_t y){return (static_cast<uint64_t>(x) * static_cast<uint64_t>(y)) >> 32;}

template <typename T, typename V>inline __host__ __device__  void __mulhi(cuda_tuple<T,1> & a, V b){a.x = __mulhi(a.x,b);}
template <typename T, typename V>inline __host__ __device__  void __mulhi(cuda_tuple<T,2> & a, V b){a.x = __mulhi(a.x,b);a.y = __mulhi(a.y,b);}
template <typename T, typename V>inline __host__ __device__  void __mulhi(cuda_tuple<T,3> & a, V b){a.x = __mulhi(a.x,b);a.y = __mulhi(a.y,b);a.z = __mulhi(a.z,b);}
template <typename T, typename V>inline __host__ __device__  void __mulhi(cuda_tuple<T,4> & a, V b){a.x = __mulhi(a.x,b);a.y = __mulhi(a.y,b);a.z = __mulhi(a.z,b);a.w = __mulhi(a.w,b);}

template <uint8_t n>
inline __host__ __device__  void add(cuda_tuple<int32_t, n> & left, cuda_tuple<int32_t, n> & right, int32_t multl, int32_t multr)
{
    __mulhi(left, multl);
    __mulhi(right, multr);
    left += right;
    left *= 0x100;
}

#endif
