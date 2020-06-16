/*
Copyright (c) 2019 Paul Stahr

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

#include <cinttypes>
#include <vector>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <omp.h>
#include <cstdio>
#include <thread>
#include "types.h"
#include "tuple_math.h"
#include "tuple_io.h"

#include "cuda_volume_raytracer.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << cudaGetErrorString( err ) << " in " << file << " at line " << line << " (" << err << ")";
        throw std::runtime_error(ss.str());
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static int inited = 0;

template <class T, T V>
struct template_constant
{
    template_constant(){};    
    constexpr operator T() const { return V; }
    template <typename W>
    constexpr T operator ()(W const &) const{return V;}
    template <typename W, typename X>
    constexpr T operator ()(W const &, X const &) const{return V;}
};

bool init()
{
    int count = 0;
    if (cudaSuccess != cudaGetDeviceCount(&count))
    {
        std::cerr << "Failed to get device count, no cuda availible" << std::endl;
        return false;
    }
    std::cout << "Device Count" << count << std::endl;
    if (!inited)
    {
        //HANDLE_ERROR(cudaSetDevice(1));
        //HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceBlockingSync));
        inited = count;
        return true;
    }
    return false;
}

template <uint8_t dim, typename DirType>
struct raydata_t{
    cuda_tuple<pos_t,dim> _position;
    cuda_tuple<DirType,dim> _direction;
    brightness_t _remaining_brightness;
    uint32_t _iterations;
};

__host__ __device__ cuda_tuple<int32_t, 2> interpolate(
    cuda_tuple<diff_t, 2> *diff_interleaved,
    uint32_t bounds1,
    cuda_tuple<uint32_t,2> pos)
{
    cuda_tuple<int32_t,2> values[4];
    diff_interleaved += (pos.x >> 16) * bounds1 + (pos.y >> 16);
    
    for (uint8_t i = 0; i < 4; ++i)
    {
        values[i] = make_struct<int32_t,2>()(diff_interleaved[((i >> 1) & 1) * bounds1 + (i & 1)]);
        values[i] *= 0x10000;
    }

    uint32_t multr = (pos.x & 0xFFFF) * 0x100;
    uint32_t multl = 0x1000000 - multr;

    for (uint8_t i = 0; i < 2; ++i)
    {
        add<2>(values[i],values[i+2],multl,multr);
    }
    multr = (pos.y & 0xFFFF) * 0x100;
    multl = 0x1000000 - multr;

    add<2>(values[0],values[1],multl,multr);
    return values[0];
 }
 
__host__ __device__ cuda_tuple<int32_t,3> interpolate(
    cuda_tuple<diff_t,4> *diff_interleaved,
    uint32_t bounds1,
    uint32_t bounds2,
    cuda_tuple<uint32_t,3> pos)
{
    diff_interleaved += ((pos.x >> 16) * bounds1 + (pos.y >> 16)) * bounds2 + (pos.z >> 16);
    cuda_tuple<int32_t,3> values[8];

    for (uint8_t i = 0; i < 8; ++i)
    {
        values[i] = make_struct<int32_t,3>()(diff_interleaved[(((i >> 2)) * bounds1 + ((i >> 1) & 1)) * bounds2 + (i & 1)]);
        values[i] *= 0x10000;
    }
    
    uint32_t multr = (pos.x & 0xFFFF) * 0x100;
    uint32_t multl = 0x1000000 - multr;
    for (uint8_t i = 0; i < 4; ++i)
    {
        add<3>(values[i],values[i+4],multl,multr);
    }
    multr = (pos.y & 0xFFFF) * 0x100;
    multl = 0x1000000 - multr;
    for (uint8_t i = 0; i < 2; ++i)
    {
        add<3>(values[i],values[i+2],multl,multr);
    }
    multr = (pos.z & 0xFFFF) * 0x100;
    multl = 0x1000000 - multr;
    add<3>(values[0],values[1],multl,multr);
    return values[0];
}

/*template <uint8_t dim, typename T>
inline __host__ __device__ cuda_tuple<float, dim> interpolatef(
    T *diff_interleaved,
    uint16_t bounds1,
    uint16_t bounds2,
    cuda_tuple<pos_t,3> pos)
{
    diff_interleaved += ((pos.x >> 16) * static_cast<uint32_t>(bounds1) + (pos.y >> 16)) * static_cast<uint32_t>(bounds2) + (pos.z >> 16);
    cuda_tuple<float,dim> values[8];
    
    for (uint8_t i = 0; i < 8; ++i)
    {
        values[i] = make_struct<float,dim>()(diff_interleaved[((i >> 2) * static_cast<uint32_t>(bounds1) + ((i >> 1) & 1)) * static_cast<uint32_t>(bounds2) + (i & 1)]);
    }

    float multr = pos.x & 0xFFFF;
    float multl = 0x10000 - (pos.x & 0xFFFF);
    for (uint8_t i = 0; i < 4; ++i)
    {
        add<dim>(values[i],values[i+4],multl,multr);
    }
    multr = pos.y & 0xFFFF;
    multl = 0x10000 - (pos.x & 0xFFFF);
    for (uint8_t i = 0; i < 2; ++i)
    {
        add<dim>(values[i],values[i+2],multl,multr);
    }
    multr = pos.z & 0xFFFF;
    multl = 0x10000 - (pos.x & 0xFFFF);
    add<dim>(values[0],values[1],multl,multr);
    values[0] /= 0x1000000000000p0f;
    return values[0];
}*/

template <uint8_t dim>
inline __host__ __device__ size_t get_index(cuda_tuple<uint16_t, dim> bounds, cuda_tuple<uint32_t, dim> pos)
{
    return 0;
}

template <>
inline __host__ __device__ size_t get_index(cuda_tuple<uint16_t, 2> bounds, cuda_tuple<uint32_t, 2> pos)
{
    return (pos.x >> 16) * static_cast<uint32_t>(bounds.y) + (pos.y >> 16);
}

template <>
inline __host__ __device__ size_t get_index(cuda_tuple<uint16_t, 3> bounds, cuda_tuple<uint32_t, 3> pos)
{
    return ((pos.x >> 16) * static_cast<uint32_t>(bounds.y) + (pos.y >> 16)) * static_cast<uint32_t>(bounds.z) + (pos.z >> 16);
}

template <uint8_t T>
struct type_uint8_t{};

template <uint8_t dim, uint8_t dimtuple, typename T>
inline __host__ __device__ cuda_tuple<float, dimtuple> interpolatef(
    cuda_tuple<T, dimtuple> *diff_interleaved,
    cuda_tuple<uint16_t,dim> bounds,
    cuda_tuple<pos_t,dim> pos,
    type_uint8_t<dimtuple> td);//{return cuda_tuple<float, dimtuple>();}  

template <uint8_t dimtuple, typename T>
inline __host__ __device__ cuda_tuple<float, dimtuple> interpolatef(
    cuda_tuple<T, dimtuple> *diff_interleaved,
    cuda_tuple<uint16_t,3> bounds,
    cuda_tuple<pos_t,3> pos,
    type_uint8_t<dimtuple> td)
{
                        
    diff_interleaved += get_index(bounds, pos);
    cuda_tuple<float,dimtuple> values[8];
    
    for (uint8_t i = 0; i < 8; ++i)
    {
        values[i] = make_struct<float,dimtuple>()(diff_interleaved[((i >> 2) * static_cast<uint32_t>(bounds.y) + ((i >> 1) & 1)) * static_cast<uint32_t>(bounds.z) + (i & 1)]);
    }
    float multr = pos.x & 0xFFFF;
    float multl = 0x10000 - multr;
    for (uint8_t i = 0; i < 4; ++i)
    {
        add<dimtuple>(values[i],values[i+4],multl,multr);
    }
    multr = pos.y & 0xFFFF;
    multl = 0x10000 - multr;
    for (uint8_t i = 0; i < 2; ++i)
    {
        add<dimtuple>(values[i],values[i+2],multl,multr);
    }
    multr = pos.z & 0xFFFF;
    multl = 0x10000 - multr;
    add<dimtuple>(values[0],values[1],multl,multr);
    values[0] /= 0x1000000000000p0f;
    return values[0];
}

template <uint8_t dimtuple, typename T>
inline __host__ __device__ cuda_tuple<float, dimtuple> interpolatef(
    cuda_tuple<T, dimtuple> *diff_interleaved,
    cuda_tuple<uint16_t,2> bounds,
    cuda_tuple<pos_t,2> pos,
    type_uint8_t<dimtuple> td)
{
    diff_interleaved += get_index(bounds, pos);
    cuda_tuple<float,dimtuple> values[4];
    
    for (uint8_t i = 0; i < 4; ++i)
    {
        values[i] = make_struct<float,dimtuple>()(diff_interleaved[((i >> 1) & 1) * static_cast<uint32_t>(bounds.y) + (i & 1)]);
    }

    float multr = pos.x & 0xFFFF;
    float multl = 0x10000 - multr;
    for (uint8_t i = 0; i < 2; ++i)
    {
        add<dimtuple>(values[i],values[i+2],multl,multr);
    }
    multr = pos.y & 0xFFFF;
    multl = 0x10000 - multr;
    add<dimtuple>(values[0],values[1],multl,multr);
    values[0] /= 0x100000000p0f;
    return values[0];
}

/*template <typename P, typename T, typename B>
__global__ void trace_ray(
    cuda_tuple<diff_t,2> *diff_interleaved,
    T translucency,
    cuda_tuple<uint32_t,2> bounds,
    raydata_t<2, dir_t> *raydata,
    P path,
    B minimum_brightness,
    uint32_t n)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        raydata += i;
        cuda_tuple<pos_t,2> pos = raydata->_position;
        cuda_tuple<int32_t,2> direction = make_struct<int32_t,2>()(raydata->_direction);
        uint32_t iterations = raydata->_iterations;
        path += iterations * i;
        direction *= 0x10000;
        
        B brightness = 0xFFFFFFFF;
        while (iterations -- > 0 && (pos >> 16) < bounds - 1)
        {
            if (translucency)
            {
                //printf("%u %u\n",pos.x>>16,pos.y>>16);
                brightness -= min(brightness, 0xFFFFFFFF-translucency[(pos.x >> 16) * bounds.y + (pos.y >> 16)]);
                if (brightness < minimum_brightness)
                {
                    break;
                }
            }
            cuda_tuple<int32_t,2> erg = interpolate(diff_interleaved, bounds.y, pos);
            erg >>= 12;
            direction += erg;
            pos += direction >> 16;
            path[iterations] = pos;
        }
        if (path)
        {
            ++iterations;
            while (iterations --> 0)
            {
                path[iterations] = pos;
            }
        }
        direction /= 0x10000;
        raydata->_position = pos;
        raydata->_direction = make_struct<int16_t,2>()(direction);
        raydata->_remaining_brightness = brightness;
        raydata->_iterations = iterations;
    }
}

template <typename P, typename T, typename B>
__global__ void trace_ray(
    cuda_tuple<diff_t,4> *diff_interleaved,
    T translucency,
    cuda_tuple<uint32_t,3> bounds,
    raydata_t<3, dir_t> *raydata,
    P path,
    B minimum_brightness,
    uint16_t n)
{
    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        raydata += i;
        path += iterations * 3 * i;
        cuda_tuple<uint32_t,3> pos = raydata->_position;
        cuda_tuple<int32_t,3> direction = make_struct<int32_t,3>()(raydata->_direction);
        uint32_t iterations = raydata._iterations;
        direction *= 0x10000;

        B brightness = 0xFFFFFFFF;
        while (iterations -- > 0 && (pos >> 16) < bounds - 1)
        {
            if (translucency)
            {
                brightness -= min(brightness, 0xFFFFFFFF-translucency[((pos.x >> 16) * bounds.y + (pos.y >> 16)) * bounds.z + (pos.z >> 16)]);
                if (brightness < minimum_brightness)
                {
                    break;
                }
            }
            cuda_tuple<int32_t,3> erg = interpolate(diff_interleaved, bounds.y, bounds.z, pos);
            erg >>= 12;
            direction += erg;
            pos += direction >> 16;
            path[iterations] = pos;
        }
        if (path)
        {
            ++iterations;
            while (iterations --> 0)
            {
                path[iterations] = pos;
            }
        }
        direction /= 0x10000;
        raydata->_position = pos;
        raydata->_direction = make_struct<dir_t,3>()(direction);
        raydata->_remaining_brightness = brightness;
        raydata->_iterations = iterations;
    }
}*/

class DummyObject
{
public:
    inline __host__ __device__ DummyObject(){}
    
    template <typename T>
    inline __host__ __device__ DummyObject(T t){}
    
    template <typename T>
    inline __host__ __device__ T& operator=(T&& other) noexcept{return other;}
    
    template <typename T>
    inline __host__ __device__ operator T() const{return T();}
    
    template <typename T>
    inline __host__ __device__ DummyObject operator-=(T value){return *this;}
    
    template <typename T>
    inline __host__ __device__ DummyObject operator<(T value){return false;}
    
    
};

template <typename T>
inline __host__ __device__ DummyObject operator -(T a, DummyObject b){return DummyObject();}

//template<class T>
//void operator=(T& value, const DummyObject& v)
//{}

class DummyArray{
    public:
    inline __host__ __device__ DummyObject const operator [](size_t index) const{return DummyObject();}

    inline __host__ __device__ DummyObject operator [](size_t index) {return DummyObject();}
    
    inline __host__ __device__ void operator +=(size_t index) {}

    inline __host__ __device__ DummyArray operator +(size_t index) {return DummyArray();}

    inline __host__ __device__ operator bool() const{return false;}
};

template <typename P, typename T, typename B, typename DiffType, typename DirType, uint8_t dim>
inline __host__ __device__ void trace_ray_function(
    cuda_tuple<DiffType,dim + 1>  *diff_interleaved,
    T translucency,/*translucency_t*/
    cuda_tuple<uint16_t,dim> bounds,
    cuda_tuple<float,dim> invscale,
    raydata_t<dim, DirType> *raydata,
    P path, /*cuda_tuple<pos_t,dim>*/    
    B minimum_brightness)
{
    cuda_tuple<uint32_t,dim> pos = raydata->_position;
    cuda_tuple<float,dim> direction = make_struct<float,dim>()(raydata->_direction);
    uint32_t iterations = raydata->_iterations;
    if(std::is_same<dir_t, DirType>::value)
    {
        direction *= 0x100;
    }
    else
    {
        direction *= 0x10000;
    }
    B brightness = 0xFFFFFFFF;
    path[--iterations] = pos;

    while (iterations -- > 0 && make_struct<uint16_t, dim>()(pos >> 16) < bounds - 1)
    {
        if (translucency)
        {
            brightness -= min(static_cast<brightness_t>(brightness), static_cast<translucency_t>(0xFFFFFFFF-translucency[get_index(bounds, pos)]));
            if (brightness < minimum_brightness)
            {
                //printf("b=%u<%u=mb", brightness, minimum_brightness);
                break;
            }
        }
        cuda_tuple<float,dim + 1> interpolation = interpolatef(diff_interleaved, bounds, pos, type_uint8_t<dim + 1>());
        if (get(interpolation, dim) > 0)
        {
            break;
        }
        interpolation *= invscale;//TODO can be precalculated
        direction += interpolation;
        float ilen = 0x40000000p0f / dot(direction, direction);
        pos += __float2int_rn2(direction * invscale * ilen);
        path[iterations] = pos;
    }
    if (!std::is_same<P,DummyArray>::value)
    {
        ++iterations;
        while (iterations --> 0)
        {
            path[iterations] = pos;
        }
    }
    if(std::is_same<dir_t, DirType>::value)
    {
        direction /= 0x100;
        raydata->_direction = make_struct<DirType,dim>()(__float2int_rn2(direction));
    }
    else
    {
        direction /= 0x10000;
        raydata->_direction = make_struct<DirType,dim>()(direction);
    }
    raydata->_position = pos;
    if (translucency)
    {
        raydata->_remaining_brightness = brightness;
    }
    raydata->_iterations = iterations;
}

template <typename P, typename T, typename B, typename DiffType, typename DirType, uint8_t dim>
void trace_rays_cpu(
    DiffType *diff_interleaved,
    T translucency,
    cuda_tuple<uint16_t,dim> bounds,
    cuda_tuple<float,dim> invscale,
    raydata_t<dim, DirType> *raydata,
    P path,
    uint32_t iterations,
    B minimum_brightness,
    size_t blocksize,
    size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) if(blocksize > 0x100)
    for (size_t i = 0; i < blocksize; ++i)
    {
        trace_ray_function(reinterpret_cast<cuda_tuple<DiffType,dim + 1>* >(diff_interleaved), translucency, bounds, invscale, raydata + i, path + iterations * i, minimum_brightness);
    }
}

template <typename P, typename T, typename B, typename DiffType, typename DirType, uint8_t dim>
__global__ void trace_rays_gpu(
    DiffType *diff_interleaved,
    T translucency,
    cuda_tuple<uint16_t,dim> bounds,
    cuda_tuple<float,dim> invscale,
    raydata_t<dim, DirType> *raydata,
    P path,
    uint32_t iterations,
    B minimum_brightness,
    uint16_t n)
{
    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        trace_ray_function(reinterpret_cast<cuda_tuple<DiffType,dim + 1>* >(diff_interleaved), translucency, bounds, invscale, raydata + i, path + iterations * i, minimum_brightness);
    }
}


template <typename U, typename T>
void interleave(U input, size_t num_rows, size_t num_cols, std::vector<T> & out)
{
    out.clear();
    std::cout << "reserve " << num_cols << '*' << num_rows << std::endl; 
    out.reserve(num_cols * num_rows);
    for (size_t i = 0; i < num_cols; ++i)
    {
        for (size_t j = 0; j < num_rows; ++j)
        {
            out.emplace_back(input[j][i]);
        }
    }
}

/*template <typename T>
void interleave(std::initializer_list<std::vector<T> > const & input, std::vector<T> & out)
{
    out.clear();
    size_t size = (*input.begin()).size();
    out.reserve(size * input.size());
    for (size_t i = 0; i < size; ++i)
    {
        for (std::vector<T> const & vec : input)
        {
            out.push_back(vec[i]);
        }
    }
}



template <typename T>
void interleave3(std::vector<std::vector<T> const * > input, std::vector<T> & out)
{
    out.clear();
    size_t size = (input[0])->size();
    out.reserve(size * input.size());
    for (size_t i = 0; i < size; ++i)
    {
        for (std::vector<T> const * vec : input)
        {
            out.emplace((*vec)[i]);
        }
    }
}

template <typename T, uint8_t dim>
void interleave2(std::array<std::vector<T> const *, dim > input, std::vector<T> & out)
{
    out.clear();
    size_t size = (*input.begin()).size();
    out.reserve(size * input.size());
    for (size_t i = 0; i < size; ++i)
    {
        for (std::vector<T> const & vec : *input)
        {
            out.emplace(vec[i]);
        }
    }
}
*/
template <int dim, typename DirType>
void fill_struct(
    std::vector<pos_t> const & start_position,
    std::vector<DirType> const & start_direction,
    size_t iterations,
    std::vector<raydata_t<dim, DirType> > & raydata)
{
    size_t num_rays = start_position.size() / dim;
    raydata.clear();
    raydata.reserve(num_rays);
    for (size_t i = 0; i < num_rays; ++i)
    {
        raydata.emplace_back();
        raydata_t<dim, DirType> & back = raydata.back();
        back._position = reinterpret_cast<cuda_tuple<pos_t, dim>const * >(start_position.data())[i];
        back._direction = reinterpret_cast<cuda_tuple<DirType, dim>const * >(start_direction.data())[i];
        
        back._remaining_brightness=std::numeric_limits<brightness_t>::max();
        back._iterations = iterations;
    } 
}

template <int dim, typename DirType>
void read_struct(
    std::vector<pos_t> & position,
    std::vector<DirType> & direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<raydata_t<dim, DirType> > const & raydata)
{
    bool warn = false;
    for (size_t i = 0; i < raydata.size(); ++i)
    {
        raydata_t<dim, DirType> const & current = raydata[i];
        
        reinterpret_cast<cuda_tuple<pos_t, dim> * >(position.data())[i] = current._position;
        reinterpret_cast<cuda_tuple<DirType, dim> * >(direction.data())[i] = current._direction;
        remaining_light[i] = current._remaining_brightness;
        if (current._iterations == 0 || current._iterations == std::numeric_limits<uint32_t>::max())
        {
            warn = true;
        }
    }
    if (warn)
    {
        std::cout << "Warning, maximum iterations hitted" << std::endl;
    }
}
    
template <typename T>
size_t inline sizeofvec(std::vector<T> const & vec)
{
    //return vec.size()      * sizeof(decltype(vec)::value_type);
    return vec.size()      * sizeof(T);
}

/*void trace_rays_cu(
    std::vector<size_t> const & output_sizes,
    std::vector<diff_t> const & diffx,
    std::vector<diff_t> const & diffy,
    std::vector<translucency_t> const & translucency_cropped,
    std::vector<pos_t> const & start_position,
    std::vector<dir_t> const & start_direction,
    std::vector<pos_t> & end_position,
    std::vector<dir_t> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    std::vector<float> const & invscale_vec,
    brightness_t minimum_brightness,
    uint32_t iterations,
    bool trace_paths,
    Options const & opt)
{    
    init();
    cuda_tuple<diff_t,2> *diff_interleaved_cuda;
    translucency_t *translucency_cuda;
    cuda_tuple<pos_t,2> *path_cuda = nullptr;
    raydata_t<2, dir_t> *raydata_cuda;
    size_t num_rays = start_position.size() / 2;
    if (trace_paths)
    {
        path.resize(iterations * 2 * num_rays);
    }
    
    std::vector<raydata_t<2, dir_t> > ray_data;
    fill_struct<2>(start_position, start_direction, iterations, ray_data);
    HANDLE_ERROR(cudaMalloc(&raydata_cuda,sizeofvec(ray_data)));
    HANDLE_ERROR(cudaMemcpyAsync(raydata_cuda,ray_data.data(),     sizeofvec(ray_data), cudaMemcpyHostToDevice));
    std::vector<diff_t> diff_interleaved;
    interleave({diffy, diffx}, diff_interleaved);
    HANDLE_ERROR(cudaMalloc(&diff_interleaved_cuda,diff_interleaved.size()      * sizeof(diff_t)));
    HANDLE_ERROR(cudaMalloc(&translucency_cuda,    translucency_cropped.size()  * sizeof(translucency_t)));
    if (trace_paths)
    {
        HANDLE_ERROR(cudaMalloc(&path_cuda,            path.size()                  * sizeof(pos_t)));
    }

    HANDLE_ERROR(cudaMemcpyAsync(diff_interleaved_cuda,           diff_interleaved.data(),                diff_interleaved.size()                 * sizeof(diff_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyAsync(translucency_cuda,    translucency_cropped.data(), translucency_cropped.size()  * sizeof(translucency_t), cudaMemcpyHostToDevice));
    float2 invscale = make_float2(invscale_vec[0],invscale_vec[1]);

    size_t maximum_rays_per_kernel = 32768;
    size_t threads_per_block = 128;
    //size_t maximum_rays_per_kernel = 64;
    //size_t threads_per_block = 32;
    
    for (size_t i = 0; i < num_rays; i += maximum_rays_per_kernel)
    {
        size_t kernel_rays = std::min(maximum_rays_per_kernel, num_rays - i * maximum_rays_per_kernel);
        size_t block_count = (kernel_rays + threads_per_block - 1)/threads_per_block;

        size_t shared_mem = 0;
        if (path_cuda == nullptr)
        {
        trace_ray<<<block_count, threads_per_block, shared_mem>>>(
            diff_interleaved_cuda,
            translucency_cuda,
            make_struct<uint32_t,2>()(output_sizes[0],output_sizes[1]),
            raydata_cuda + i * maximum_rays_per_kernel,
            DummyArray(),
            iterations,
            minimum_brightness,
            kernel_rays);
        }
        else
        {
        trace_ray<<<block_count, threads_per_block, shared_mem>>>(
            diff_interleaved_cuda,
            translucency_cuda,
            make_struct<uint32_t,2>()(output_sizes[0],output_sizes[1]),
            raydata_cuda + i * maximum_rays_per_kernel,
            path_cuda + i * iterations,
            iterations,
            minimum_brightness,
            kernel_rays);
        }
    }

    HANDLE_ERROR(cudaMemcpyAsync(ray_data.data(),    raydata_cuda,    sizeofvec(ray_data), cudaMemcpyDeviceToHost));
    read_struct<2>(end_position, end_direction, remaining_light, ray_data);
    if (trace_paths)
    {
        HANDLE_ERROR(cudaMemcpyAsync(path.data(),            path_cuda,            path.size()            * sizeof(pos_t),  cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaThreadSynchronize());

    HANDLE_ERROR(cudaFree(diff_interleaved_cuda));
    HANDLE_ERROR(cudaFree(raydata_cuda));
    HANDLE_ERROR(cudaFree(translucency_cuda));
    if (trace_paths)
    {
        HANDLE_ERROR(cudaFree(path_cuda));
    }
}*/

template <typename T>
std::vector<std::vector<T> const *> convert_to_references(std::vector<std::vector<T> > const & data)
{
    std::vector<std::vector<T> const *> res;
    res.reserve(data.size());
    for (std::vector<T> const & d : data)
    {
        res.push_back(&d);
    }
    return res;
}

template <typename DiffType>
TraceRaysCu<DiffType>::TraceRaysCu(
    std::vector<size_t> const & output_sizes_,
    std::vector<std::vector<DiffType> > const & diff_,
    std::vector<translucency_t> const & translucency_cropped_) : TraceRaysCu(output_sizes_, convert_to_references(diff_), translucency_cropped_)
{}

template <typename DiffType>
TraceRaysCu<DiffType>::TraceRaysCu(
        std::vector<size_t> const & bounds,
        std::vector<std::vector<DiffType> const * > const & diff,
        std::vector<translucency_t> const & translucency_cropped) : _translucency_cropped(translucency_cropped)
{
    init();
    uint8_t dim = bounds.size();
    _output_sizes.reserve(dim);
    std::copy(bounds.begin(), bounds.end(), std::back_inserter(_output_sizes));
    std::vector<DiffType> extra_component;
    extra_component.reserve(diff[0]->size());
    for (translucency_t tr : translucency_cropped)
    {
        extra_component.push_back((static_cast<int64_t>(0x7FFFFFFF)-(static_cast<int64_t>(tr)))/0x10000);
    }
    std::vector<typename std::vector<DiffType>::const_iterator> tmp;
    tmp.reserve(bounds.size() + 1);
    for (std::vector<DiffType> const * d : diff)
    {
        tmp.push_back(d->cbegin());
    }
    tmp.push_back(extra_component.cbegin()); 
    interleave(tmp, tmp.size(), diff[0]->size(), _diff_interleaved);
    _diff_interleaved_cuda.resize(inited);
    _translucency_cuda.resize(inited);
    for (size_t i = 0; i < inited; ++i)
    {
        HANDLE_ERROR(cudaSetDevice(i));
        HANDLE_ERROR(cudaMalloc(&_diff_interleaved_cuda[i],_diff_interleaved.size()      * sizeof(DiffType)));
        HANDLE_ERROR(cudaMalloc(&_translucency_cuda[i],    _translucency_cropped.size()  * sizeof(translucency_t)));

        HANDLE_ERROR(cudaMemcpyAsync(_diff_interleaved_cuda[i],_diff_interleaved.data(),     _diff_interleaved.size()      * sizeof(DiffType), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpyAsync(_translucency_cuda[i],    _translucency_cropped.data(), _translucency_cropped.size()  * sizeof(translucency_t), cudaMemcpyHostToDevice));
    }
}
    
template <typename DiffType>
template <typename DirType>
void TraceRaysCu<DiffType>::trace_rays_cu(
    std::vector<pos_t> const & start_position,
    std::vector<DirType> const & start_direction,
    std::vector<pos_t> & end_position,
    std::vector<DirType> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    std::vector<float> const & invscale_vec,
    brightness_t minimum_brightness,
    uint32_t iterations,
    bool trace_paths,
    Options const & opt)
    {
        if (_output_sizes.size() == 2)
        {
            trace_rays_cu_impl<DirType, 2>(start_position,
                start_direction,
                end_position,
                end_direction,
                remaining_light,
                path,
                invscale_vec,
                minimum_brightness,
                iterations,
                trace_paths,
                opt);
        }
        else if (_output_sizes.size() == 3)
        {
            trace_rays_cu_impl<DirType, 3>(start_position,
                start_direction,
                end_position,
                end_direction,
                remaining_light,
                path,
                invscale_vec,
                minimum_brightness,
                iterations,
                trace_paths,
                opt);
        }
        else
        {
            throw std::runtime_error("Illegal dimension");
        }
    }
    
template <typename DiffType>
template <typename DirType, uint8_t dim>
void TraceRaysCu<DiffType>::trace_rays_cu_impl(
    std::vector<pos_t>          const & start_position,
    std::vector<DirType>        const & start_direction,
    std::vector<pos_t> &        end_position,
    std::vector<DirType> &      end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> &        path,
    std::vector<float> const &  invscale_vec,
    brightness_t                minimum_brightness,
    uint32_t                    iterations,
    bool                        trace_paths,
    Options const &             opt)
{    
    size_t num_rays = start_position.size() / dim;
    if (trace_paths)
    {
        path.resize(iterations * dim * num_rays);
    }

    std::vector<raydata_t<dim, DirType> > ray_data;
    fill_struct<dim>(start_position, start_direction, iterations, ray_data);
    size_t maximum_rays_per_kernel = 0x8000;
    size_t threads_per_block = 0x80;
    //size_t maximum_rays_per_kernel = 64;
    //size_t threads_per_block = 32;
    size_t cuda_device_count = inited;
    if (num_rays <= 0x80)
    {
        cuda_device_count = 0;
    }
    cuda_device_count = std::min(cuda_device_count, (num_rays + maximum_rays_per_kernel - 1) / maximum_rays_per_kernel);
    std::vector<cuda_tuple<pos_t,dim> *> path_cuda(cuda_device_count);
    std::cout << "allocate " << cuda_device_count << std::endl;
    std::vector<raydata_t<dim, DirType>* > raydata_cuda(cuda_device_count);
    size_t cpu_device_count = omp_get_max_threads() - cuda_device_count;
    if (cuda_device_count != 0)
    {
        cpu_device_count = 0;
    }
    
    cuda_tuple<float,dim> invscale = make_struct<float,dim>()(invscale_vec.data());
    cuda_tuple<uint16_t, dim> output_sizes = make_struct<uint16_t, dim>()(_output_sizes.data());
    omp_set_nested(1);
    size_t count_cpu = 0;
    size_t count_gpu = 0;
    assert(diff_interleaved.size() == prod(output_sizes));
    size_t num_parallel = cuda_device_count + (cpu_device_count > 0);
    #pragma omp parallel for schedule(dynamic) num_threads(num_parallel) if (num_parallel > 1)
    for (size_t i = 0; i < num_rays; i += maximum_rays_per_kernel)
    {
        size_t num_kernel_rays = std::min(maximum_rays_per_kernel, num_rays - i);
        size_t block_count = (num_kernel_rays + threads_per_block - 1)/threads_per_block;
        if (opt._loglevel > 0)
        {
            std::cout << "iteration " << (i / maximum_rays_per_kernel) << " of " << (num_rays + maximum_rays_per_kernel - 1) / maximum_rays_per_kernel << std::endl;
        }
        size_t thread_num = omp_get_thread_num();
        if (thread_num < cuda_device_count)
        {
            if (raydata_cuda[thread_num] == nullptr)
            {
                HANDLE_ERROR(cudaSetDevice(thread_num));
                HANDLE_ERROR(cudaMalloc(&raydata_cuda[thread_num],std::min(ray_data.size(),maximum_rays_per_kernel)      * sizeof(raydata_t<dim, DirType>)));
                if (trace_paths)
                {
                    HANDLE_ERROR(cudaMalloc(&path_cuda[thread_num],            std::min(ray_data.size(), maximum_rays_per_kernel) * iterations * dim * sizeof(pos_t)));
                }
            }
            std::cout << thread_num << " gpu: " <<  num_kernel_rays << std::endl;
            HANDLE_ERROR(cudaMemcpyAsync(raydata_cuda[thread_num],ray_data.data() + i,     num_kernel_rays      * sizeof(raydata_t<dim, DirType>), cudaMemcpyHostToDevice));
            if (trace_paths)
            {
                trace_rays_gpu<<<block_count, threads_per_block>>>(
                    _diff_interleaved_cuda[thread_num],
                    _translucency_cuda[thread_num],
                    output_sizes,
                    invscale,
                    raydata_cuda[thread_num],
                    path_cuda[thread_num],
                    iterations,
                    minimum_brightness,
                    num_kernel_rays);
            }
            else
            {
                trace_rays_gpu<<<block_count, threads_per_block>>>(
                    _diff_interleaved_cuda[thread_num],
                    DummyArray(),//_translucency_cuda,
                    output_sizes,
                    invscale,
                    raydata_cuda[thread_num],
                    DummyArray(),
                    iterations,
                    DummyObject(),//minimum_brightness,
                    num_kernel_rays);
            }
            HANDLE_ERROR(cudaThreadSynchronize());
            HANDLE_ERROR(cudaMemcpyAsync(ray_data.data() + i,    raydata_cuda[thread_num],    num_kernel_rays   * sizeof(raydata_t<dim, DirType>), cudaMemcpyDeviceToHost));
            if (trace_paths)
            {
                HANDLE_ERROR(cudaMemcpyAsync(path.data() + i * dim * iterations,            path_cuda[thread_num],            path.size()            * sizeof(pos_t),  cudaMemcpyDeviceToHost));
            }
            ++count_gpu;
        }
        else
        {
            std::cout << thread_num << " cpu: " <<  num_kernel_rays << std::endl;
            if (trace_paths)
            {
                trace_rays_cpu(
                    _diff_interleaved.data(),
                    DummyArray(),//_translucency_cropped.data(),
                    output_sizes,
                    invscale,
                    ray_data.data() + i,
                    reinterpret_cast<cuda_tuple<pos_t,dim>*>(path.data()) + i * iterations,
                    iterations,
                    DummyObject(),//minimum_brightness,
                    num_kernel_rays,
                    cpu_device_count);
            }else{
                trace_rays_cpu(
                    _diff_interleaved.data(),
                    DummyArray(),//_translucency_cropped.data(),
                    output_sizes,
                    invscale,
                    ray_data.data() + i,
                    DummyArray(),
                    iterations,
                    DummyObject(),//minimum_brightness,
                    num_kernel_rays,
                    cpu_device_count);
            }
            ++count_cpu;
        }
        //std::cout << ray_data[i]._position[0] << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (opt._loglevel > 0)
    {
        std::cout << "cpu: " << count_cpu << " gpu: " << count_gpu << std::endl;
    }
    read_struct<dim>(end_position, end_direction, remaining_light, ray_data);
    for (size_t i = 0; i < cuda_device_count; ++i)
    {
        HANDLE_ERROR(cudaSetDevice(i));
        HANDLE_ERROR(cudaFree(raydata_cuda[i]));
        if (trace_paths)
        {
            HANDLE_ERROR(cudaFree(path_cuda[i]));
        }        
    }
    if (cuda_device_count != 0)
    {
        HANDLE_ERROR(cudaThreadSynchronize());
    }
}
    
template <typename DiffType>
TraceRaysCu<DiffType>::~TraceRaysCu()
{
    for (size_t i = 0; i < inited; ++i)
    {
        HANDLE_ERROR(cudaSetDevice(i));
        HANDLE_ERROR(cudaFree(_diff_interleaved_cuda[i]));
        HANDLE_ERROR(cudaFree(_translucency_cuda[i]));
    }
    if (inited)
    {
        HANDLE_ERROR(cudaThreadSynchronize());
    }
}


template class TraceRaysCu<diff_t>;
template class TraceRaysCu<float>;

template void TraceRaysCu<diff_t>::trace_rays_cu<dir_t>(
        std::vector<pos_t> const & start_position,
        std::vector<dir_t> const & start_direction,
        std::vector<pos_t> & end_position,
        std::vector<dir_t> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        std::vector<float> const & invscale_vec,
        brightness_t minimum_brightness,
        uint32_t iterations,
        bool trace_paths,
        Options const & opt);

template void TraceRaysCu<float>::trace_rays_cu<dir_t>(
        std::vector<pos_t> const & start_position,
        std::vector<dir_t> const & start_direction,
        std::vector<pos_t> & end_position,
        std::vector<dir_t> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        std::vector<float> const & invscale_vec,
        brightness_t minimum_brightness,
        uint32_t iterations,
        bool trace_paths,
        Options const & opt);
        
template void TraceRaysCu<diff_t>::trace_rays_cu<float>(
        std::vector<pos_t> const & start_position,
        std::vector<float> const & start_direction,
        std::vector<pos_t> & end_position,
        std::vector<float> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        std::vector<float> const & invscale_vec,
        brightness_t minimum_brightness,
        uint32_t iterations,
        bool trace_paths,
        Options const & opt);

template void TraceRaysCu<float>::trace_rays_cu<float>(
        std::vector<pos_t> const & start_position,
        std::vector<float> const & start_direction,
        std::vector<pos_t> & end_position,
        std::vector<float> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        std::vector<float> const & invscale_vec,
        brightness_t minimum_brightness,
        uint32_t iterations,
        bool trace_paths,
        Options const & opt);

