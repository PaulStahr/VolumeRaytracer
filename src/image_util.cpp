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

#include <numeric>
#include <algorithm>
#include <cassert>
#include <functional>
#include <experimental/filesystem>
#include "cuda_volume_raytracer.h"
#include "image_util.h"
#include "types.h"
#include "io_util.h"
#include "image_io.h"
#include "util.h"

namespace SERIALIZE{
template <typename IorType, typename DirType>
std::ostream & write_value(std::ostream & out, RaytraceInstanceRef<IorType, DirType> const & value)
{
    write_value(out, value._bound_vec);
    write_value(out, value._ior);
    write_value(out, value._translucency);
    write_value(out, value._start_position);
    write_value(out, value._start_direction);
    write_value(out, value._scale);
    write_value(out, value._minimum_brightness);
    write_value(out, value._iterations);
    write_value(out, value._trace_path);
    write_value(out, value._normalize_length);
    return out;
}

template <typename IorType, typename DirType>
std::ostream & write_value(std::ostream & out, RaytraceInstance<IorType, DirType> const & value)
{
    return write_value(out, RaytraceInstanceRef<IorType, DirType>(const_cast<RaytraceInstance<IorType, DirType> &>(value)));
}

template <typename IorType, typename DirType>
std::istream & read_value(std::istream & in, RaytraceInstance<IorType, DirType> & value)
{
    read_value(in, value._bound_vec);
    read_value(in, value._ior);
    read_value(in, value._translucency);
    read_value(in, value._start_position);
    read_value(in, value._start_direction);
    read_value(in, value._scale);
    read_value(in, value._minimum_brightness);
    read_value(in, value._iterations);
    read_value(in, value._trace_path);
    read_value(in, value._normalize_length);
    return in;
}

template std::istream & read_value(std::istream & in, RaytraceInstance<ior_t, dir_t> & value);

template <typename IorType>
std::ostream & write_value(std::ostream & out, RayTraceSceneInstanceRef<IorType> const & value)
{
    write_value(out, value._bound_vec);
    write_value(out, value._ior);
    write_value(out, value._translucency);
    return out;
}

template <typename IorType>
std::ostream & write_value(std::ostream & out, RayTraceSceneInstance<IorType> const & value)
{
    return write_value(out, RayTraceSceneInstanceRef<IorType>(const_cast<RayTraceSceneInstance<IorType> &>(value)));
}

template
std::ostream & write_value(std::ostream & out, RayTraceSceneInstance<float> const & value);

template
std::ostream & write_value(std::ostream & out, RayTraceSceneInstance<ior_t> const & value);

template <typename IorType>
std::istream & read_value(std::istream & in, RayTraceSceneInstance<IorType> & value)
{
    std::cout << "Read raytrace scene instance" << std::endl;
    read_value(in, value._bound_vec);
    read_value(in, value._ior);
    read_value(in, value._translucency);
    return in;
}

template std::istream & read_value(std::istream & in, RayTraceSceneInstance<ior_t> & value);
/*
template <typename DirType>
std::ostream & write_value(std::ostream & out, RayTraceRayInstanceRef<DirType> const & value)
{
    write_value(out, value._start_position);
    write_value(out, value._start_direction);
    write_value(out, value._scale);
    write_value(out, value._minimum_brightness);
    write_value(out, value._iterations);
    write_value(out, value._trace_path);
    write_value(out, value._normalize_length);
    return out;
}


template <typename DirType>
std::ostream & write_value(std::ostream & out, RayTraceRayInstance<DirType> const & value)
{
    return write_value(out, RayTraceRayInstanceRef<DirType>(const_cast<RayTraceRayInstance<DirType> &>(value)));
}*/

template <typename DirType>
std::istream & read_value(std::istream & in, RayTraceRayInstance<DirType> & value)
{
    std::cout << "Read raytrace ray instance" << std::endl;
    read_value(in, value._start_position);
    read_value(in, value._start_direction);
    read_value(in, value._scale);
    read_value(in, value._minimum_brightness);
    read_value(in, value._iterations);
    read_value(in, value._trace_path);
    read_value(in, value._normalize_length);
    return in;
}

template std::istream & read_value(std::istream & in, RayTraceRayInstance<dir_t> & value);
}

template <typename IorType>
RayTraceSceneInstanceRef<IorType>::RayTraceSceneInstanceRef(RayTraceSceneInstance<IorType> & ref) : _bound_vec(ref._bound_vec), _ior(ref._ior), _translucency(ref._translucency){}

template RayTraceSceneInstanceRef<ior_t>::RayTraceSceneInstanceRef(RayTraceSceneInstance<ior_t> & ref);

void get_position(size_t index, std::vector<size_t> const & bounds, std::vector<size_t> & pos)
{
    pos.resize(bounds.size(), 0);
    for (size_t i = bounds.size(); i --> 0;)
    {
        pos[i] = index % bounds[i];
        index /= bounds[i];
    }
}

template <typename T>
void permutate_dimensions(
    std::vector<T> const & input,
    std::vector<T> & output,
    std::vector<size_t> const & bounds,
    std::vector<size_t> const & permutation)
{
    assert(permutation.size() == bounds.size());
    output.resize(input.size());
    std::vector<size_t> input_position;
    std::vector<size_t> output_position;
    output_position.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        get_position(i, bounds, input_position);
        output_position.clear();
        UTIL::permutate_from_indice(permutation.begin(), permutation.end(), input_position.begin(), std::back_inserter(output_position));
        size_t out_index = get_index(bounds, output_position.begin());
        output[out_index] = input[i];
    }
}
   
template <typename T>
void swap_dimensions(
    std::vector<T> const & input,
    std::vector<T> & output,
    std::vector<size_t> const & bounds,
    size_t x, size_t y)
{
    std::vector<size_t> permutation;
    permutation.reserve(bounds.size());
    UTIL::iota_n(std::back_inserter(permutation), bounds.size(), 0);
    permutation[x] = y;
    permutation[y] = x;
    permutate_dimensions(input, output, bounds, permutation);
}


template <typename T,typename V, typename W>
struct convolution
{
    std::vector<T> const & _input;
    std::vector<size_t> const & _input_sizes;
    std::vector<size_t> const & _stencil_sizes;
    std::vector<size_t> _output_sizes;
    size_t _output_size;
    size_t _stencil_size;
    
    convolution(
        std::vector<T> const & input_,
        std::vector<size_t> const & input_sizes_,
        std::vector<size_t> const & stencil_sizes_): _input(input_), _input_sizes(input_sizes_), _stencil_sizes(stencil_sizes_)
    {
        _output_sizes.reserve(_input_sizes.size());
        std::transform(_input_sizes.begin(), _input_sizes.end(), _stencil_sizes.begin(), std::back_inserter(_output_sizes), std::minus<size_t>());
        std::transform(_output_sizes.begin(), _output_sizes.end(), _output_sizes.begin(), UTIL::plus<size_t>(1));
        _output_size = accumulate(_output_sizes.begin(), _output_sizes.end(), 1lu, std::multiplies<size_t>());
        _stencil_size = accumulate(_stencil_sizes.begin(), _stencil_sizes.end(), 1lu, std::multiplies<size_t>());
    }
        
    void operator ()(std::vector<V> const & stencil, V weight, std::vector<W> & output)
    {
        output.clear();
        output.reserve(_output_size);
        std::vector<size_t> position;
        std::vector<size_t> stencil_offsets;
        std::vector<V> reduced_stencil;
        stencil_offsets.reserve(_stencil_size);
        reduced_stencil.reserve(stencil.size());
        for (size_t j = 0; j < stencil.size(); ++j)
        {
            if (stencil[j] != 0)
            {
                get_position(j, _stencil_sizes, position);
                stencil_offsets.push_back(get_index(_input_sizes, position.begin()));
                reduced_stencil.push_back(stencil[j]);
            }
        }
        for (size_t i = 0; i < _output_size; ++i)
        {
            get_position(i, _output_sizes, position);
            auto input_iter = _input.begin() + get_index(_input_sizes, position.begin());
            
            V sum = 0;
            for (size_t j = 0; j < reduced_stencil.size(); ++j)
            {
                sum += reduced_stencil[j] * input_iter[stencil_offsets[j]];
            }
            if (std::is_same<V, float>::value)
            {
                sum /= weight;
            }
            else
            {
                sum = divRoundClosest(sum, weight);
            }
            output.push_back(sum);
            if (output.back() != sum)
            {
                throw std::runtime_error("differention overflow:" + std::to_string(output.back()) + "!=" + std::to_string(sum));
            }
        }
    }
};

template <typename T>
void crop_matrix(
    std::vector<T> const & input,
    std::vector<T> & output,
    std::vector<size_t> const & bounds,
    std::vector<size_t> const & lower_bound,
    std::vector<size_t> const & output_bounds)
{
    output.clear();
    size_t output_size = std::accumulate(output_bounds.begin(), output_bounds.end(), 1lu, std::multiplies<size_t>());
    output.reserve(output_size);
    std::vector<size_t> position;
    for (size_t i = 0; i < output_size; ++i)
    {
        get_position(i, output_bounds, position);
        std::transform(position.begin(), position.end(), lower_bound.begin(), position.begin(), std::plus<size_t>());
        size_t input_index = get_index(bounds, position.begin());
        output.push_back(input[input_index]);
    }
}

namespace fs = std::experimental::filesystem;

template <typename T>
void export_image_stack(std::string const & prefix, std::vector<T> const & image, std::vector<size_t> const & bounds)
{
    std::cout << "export " << prefix << std::endl;
    if (bounds.size() == 3)
    {
        IMG_IO::image_t img(bounds[1], bounds[2], 3);
        for (size_t i = 0; i < bounds[0]; ++i)
        {
            for (size_t j = 0; j < bounds[1] * bounds[2]; ++j)
            {
                std::fill(img._data.begin() + j * 3, img._data.begin() + j * 3 + 3, (image[i * bounds[1] * bounds[2] + j]) / 10 + 128);
            }
            //std::transform(image.begin() + i * bounds[1] * bounds[2], image.begin() + (i+1) * bounds[1] * bounds[2], img._data.begin(), UTIL::divide_by<T>(0x100));
            std::string tmp = prefix + std::to_string(i) + ".png";
            std::string dir = tmp.substr(0, tmp.find_last_of("/\\"));
            if (!fs::is_directory(dir) || !fs::exists(dir)) { // Check if src folder exists
                fs::create_directories(dir); // create src folder
            }
            write_png(tmp.c_str(), img);
        }
    }
    else if (bounds.size() == 2)
    {
        IMG_IO::image_t img(bounds[0], bounds[1], 3);
         for (size_t j = 0; j < bounds[0] * bounds[1]; ++j)
        {
            std::fill(img._data.begin() + j * 3, img._data.begin() + j * 3 + 3, (image[j]) / 10 + 128);
        }
        //std::transform(image.begin() + i * bounds[1] * bounds[2], image.begin() + (i+1) * bounds[1] * bounds[2], img._data.begin(), UTIL::divide_by<T>(0x100));
        std::string tmp = prefix + ".png";
        
        if (!fs::is_directory(tmp) || !fs::exists(tmp)) { // Check if src folder exists
            fs::create_directories(tmp); // create src folder
        }
        
        write_png(tmp.c_str(), img);
    }
    else if (bounds.size() == 1)
    {
        IMG_IO::image_t img(bounds[0], 1, 3);
         for (size_t j = 0; j < bounds[0]; ++j)
        {
            std::fill(img._data.begin() + j * 3, img._data.begin() + j * 3 + 3, (image[j]) / 10 + 128);
        }
        //std::transform(image.begin() + i * bounds[1] * bounds[2], image.begin() + (i+1) * bounds[1] * bounds[2], img._data.begin(), UTIL::divide_by<T>(0x100));
        std::string tmp = prefix + ".png";
        
        if (!fs::is_directory(tmp) || !fs::exists(tmp)) { // Check if src folder exists
            fs::create_directories(tmp); // create src folder
        }
        
        write_png(tmp.c_str(), img);
    }
}

template <typename T, typename V, typename W>
void calculate_differations(
    std::vector<T> const & ior,
    std::vector<size_t> const & bounds,
    std::vector<V> & diffx,
    std::vector<V> & diffy,
    std::vector<V> & diffz, 
    W div,
    std::vector<size_t> & output_sizes)
{
    /*std::vector<int32_t> stamp({
        -1,0,1,-2,0,2,-1,0,1,
        -2,0,2,-4,0,4,-2,0,2,
        -1,0,1,-2,0,2,-1,0,1
    });*/
    std::vector<W> stamp({
        -14,0,14,-47,0,47,-14,0,14,
        -47,0,47,-162,0,162,-47,0,47,
        -14,0,14,-47,0,47,-14,0,14
    });   
    auto transform = UTIL::transform_iter(stamp.begin(), UTIL::abs);
    W stamp_weight = std::accumulate(transform, transform + stamp.size(), W(0), std::plus<W>()) * div;
         
    std::vector<size_t> stencil_bounds(3,3);
#ifndef NDEBUG 
    export_image_stack("differentiations/orig", ior, bounds);
#endif
    std::exception_ptr eptr;
    convolution<T, W, V> conv(ior, bounds, stencil_bounds);
    #pragma omp parallel for num_threads(3)
    for (size_t i = 0; i < 3; ++i){
        try{
            std::vector<size_t> local_output_sizes;
            switch(i)
            {
                case 0:
                {
                    
                    conv(stamp, stamp_weight, diffx);
    #ifndef NDEBUG 
                    export_image_stack("differentiations/diffx", diffx, local_output_sizes);
    #endif
                    break;
                }
                case 1: 
                {
                    std::vector<W> stencil;
                    std::cout << local_output_sizes.data() << std::endl;
                    swap_dimensions(stamp, stencil, stencil_bounds,1,2);
                    conv( stencil, stamp_weight, diffy);
    #ifndef NDEBUG 
                    export_image_stack("differentiations/diffy", diffy, local_output_sizes);
    #endif
                    break;
                }
                case 2: 
                {
                    std::vector<W> stencil;
                    swap_dimensions(stamp, stencil, stencil_bounds,0,2);
                    conv(stencil, stamp_weight, diffz);
    #ifndef NDEBUG 
                    export_image_stack("differentiations/diffz", diffz, local_output_sizes);
    #endif
                    break;
                }
            }
        }catch(...) {
            eptr = std::current_exception(); // capture
        }
    }
    output_sizes = conv._output_sizes;
    if (eptr) {
        std::rethrow_exception(eptr);
    }
}

template <typename T, typename V, typename W>
void calculate_differations(
    std::vector<T> const & ior,
    std::vector<size_t> const & bounds,
    std::vector<V> & diffx,
    std::vector<V> & diffy,
    W div,
    std::vector<size_t> & output_sizes)
{
    std::vector<W> stencilx({-47,0,47,-162,0,162,-47,0,47});
    std::vector<size_t> stencil_bounds(2,3);
    auto transform = UTIL::transform_iter(stencilx.begin(), UTIL::abs);
    W stamp_weight = std::accumulate(transform, transform + stencilx.size(), W(0), std::plus<W>()) * div;
    convolution<T, W, V> conv(ior, bounds, stencil_bounds);
    std::exception_ptr eptr;
    #pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < 2; ++i){
        try{
            std::vector<size_t> local_output_sizes;
            switch(i)
            {
                case 0:{
                    conv(stencilx, stamp_weight, diffx);
                    break;
                }
                case 1:{
                    std::vector<W> stencily;
                    swap_dimensions(stencilx, stencily, stencil_bounds,0,1);
                    conv(stencily, stamp_weight, diffy);
                    break;
                }
            }
        }catch(...) {
            eptr = std::current_exception(); // capture
        }
    }
    output_sizes = conv._output_sizes;
    if (eptr) {
        std::rethrow_exception(eptr);
    }
}


template <typename IorType, typename IorLogType, typename DiffType>
RaytraceScene<IorType, IorLogType, DiffType>::RaytraceScene(RayTraceSceneInstanceRef<IorType> const & ref, Options const & opt): RaytraceScene(ref._bound_vec, ref._ior, ref._translucency, opt){}

template RaytraceScene<ior_t, iorlog_t, diff_t>::RaytraceScene(RayTraceSceneInstanceRef<ior_t> const & ref, Options const & opt);

template <>
RaytraceScene<ior_t, iorlog_t, diff_t>::RaytraceScene(
        std::vector<size_t> const & bound_vec,
        std::vector<ior_t> const & ior,
        std::vector<translucency_t> const & translucency,
        Options const & opt): _bound_vec(bound_vec), _ior(ior), _translucency(translucency)
{
    uint8_t dim = bound_vec.size();
    if (dim == 0)
    {
        throw std::runtime_error("dimension is zero");
    }
    if (std::accumulate(_bound_vec.begin(), _bound_vec.end(), 1lu, std::multiplies<size_t>()) != _ior.size() || _translucency.size() != _ior.size())
    {
        throw std::runtime_error("imagesizes doesn't match:" + std::to_string(std::accumulate(_bound_vec.begin(), _bound_vec.end(), 1lu, std::multiplies<size_t>())) + "=" + std::to_string(_ior.size()) + "=" +  std::to_string(_translucency.size()) + " (bounds ior translucency)");
    }
    std::vector<size_t> lower_bound(_bound_vec.size(), 1);
    std::vector<size_t> output_size(_bound_vec);
    std::transform(output_size.begin(), output_size.end(), lower_bound.begin(), output_size.begin(), std::minus<size_t>());
    std::transform(output_size.begin(), output_size.end(), lower_bound.begin(), output_size.begin(), std::minus<size_t>());
    crop_matrix(_translucency, _translucency_cropped, _bound_vec, lower_bound, output_size);
    std::exception_ptr eptr;
    _ior_log.resize(_ior.size(), 0);
    #pragma omp parallel for
    for (size_t i = 0; i < _ior.size(); ++i)
    {
        if (eptr)
        {
            continue;
        }
        try{
            double fior = static_cast<double>(_ior[i])/0x10000;
            double tmp = log(fior)* 0x420000;
             if (tmp > static_cast<double>(std::numeric_limits<iorlog_t>::max()))
            {
                throw std::runtime_error("refraction-index overflow");
            }
            if (tmp < static_cast<double>(std::numeric_limits<iorlog_t>::lowest()))
            {
                throw std::runtime_error("refraction-index underflow: " + std::to_string(tmp) + '<' + std::to_string(static_cast<double>(std::numeric_limits<iorlog_t>::min()))+ " ior was: " + std::to_string(fior));
            }
            _ior_log[i] = std::round(tmp);
        }catch(...) {
            std::cout << "catched exception" << std::endl;  
            #pragma omp critical
            {
                eptr = std::current_exception(); // capture
            }
        }
    }
    if (eptr) {
        std::rethrow_exception(eptr);
    }
    _diff.resize(dim);
    switch (dim)
    {
        case 2:
        {
            calculate_differations(_ior_log, bound_vec, _diff[0], _diff[1], int32_t(0x100), _diff_bound_vec);
            break;
        }
        case 3:
        {
            calculate_differations(_ior_log, bound_vec, _diff[0], _diff[1], _diff[2], int32_t(0x100), _diff_bound_vec);
            break;
        }
        default:
        {
            throw std::runtime_error("Illegal dimension: " + std::to_string(bound_vec.size()));
        }
    }
    _calculation_object = new TraceRaysCu<diff_t>(_diff_bound_vec, _diff, _translucency_cropped);
    if (opt._loglevel < -1)
    {
        auto tr_minmax = std::minmax_element(_translucency_cropped.begin(), _translucency_cropped.end());
        auto ior_minmax = std::minmax_element(_ior.begin(), _ior.end());
        auto iorl_minmax = std::minmax_element(_ior_log.begin(), _ior_log.end());
        for (size_t i = 0; i < _diff.size(); ++i)
        {
            auto diff_minmax = std::minmax_element(_diff[i].begin(), _diff[i].end());
            std::cout << "diff" << i << " (" << *diff_minmax.first << ' ' << *diff_minmax.second << ") ";
        }
        std::cout << "tr ("<< *tr_minmax.first << ' ' << *tr_minmax.second << ") ior (" << *ior_minmax.first << ' ' << *ior_minmax.second << ") iorl (" << *iorl_minmax.first << ' ' << *iorl_minmax.second << ") outputsize: " << _calculation_object->_output_sizes.size() << std::endl;
    }
}

template <>
RaytraceScene<float, float, float>::RaytraceScene(
        std::vector<size_t> const & bound_vec,
        std::vector<float> const & ior,
        std::vector<translucency_t> const & translucency,
        Options const & opt): _bound_vec(bound_vec), _ior(ior), _translucency(translucency), _calculation_object(nullptr)
{
    size_t dim = bound_vec.size();
    if (dim == 0)
    {
        throw std::runtime_error("dimension is zero");
    }
    if (std::accumulate(_bound_vec.begin(), _bound_vec.end(), 1lu, std::multiplies<size_t>()) != _ior.size() || _translucency.size() != _ior.size())
    {
        throw std::runtime_error("imagesizes doesn't match:" + std::to_string(std::accumulate(_bound_vec.begin(), _bound_vec.end(), 1lu, std::multiplies<size_t>())) + "=" + std::to_string(_ior.size()) + "=" +  std::to_string(_translucency.size()) + " (bounds ior translucency)");
    }
    std::vector<size_t> lower_bound(_bound_vec.size(), 1);
    std::vector<size_t> output_size(_bound_vec);
    std::transform(output_size.begin(), output_size.end(), lower_bound.begin(), output_size.begin(), std::minus<size_t>());
    std::transform(output_size.begin(), output_size.end(), lower_bound.begin(), output_size.begin(), std::minus<size_t>());
    crop_matrix(_translucency, _translucency_cropped, _bound_vec, lower_bound, output_size);
    std::exception_ptr eptr;
    _ior_log.resize(_ior.size(), 0);
    #pragma omp parallel for
    for (size_t i = 0; i < _ior.size(); ++i)
    {
        if (eptr)
        {
            continue;
        }
        try{
            if (_ior[i] <= 0)
            {
                throw std::runtime_error("refraction-index underflow: " +std::to_string(_ior[i])+ "<0");
            }
            _ior_log[i] = log(_ior[i])* 0x420000;
        }catch(...) {
            std::cout << "catched exception" << std::endl;  
            #pragma omp critical
            {
                eptr = std::current_exception(); // capture
            }
        }
    }
    if (eptr) {
        std::rethrow_exception(eptr);
    }
    _diff.resize(dim);
    switch (dim)
    {
        case 2:
        {
            calculate_differations(_ior_log, bound_vec, _diff[0], _diff[1], float(0x100), _diff_bound_vec);
            break;
        }
        case 3:
        {
            calculate_differations(_ior_log, bound_vec, _diff[0], _diff[1], _diff[2], float(0x100), _diff_bound_vec);
            break;
        }
        default:
        {
            throw std::runtime_error("Illegal dimension: " + std::to_string(bound_vec.size()));
        }
    }
    _calculation_object = new TraceRaysCu<float>(_diff_bound_vec, _diff, _translucency_cropped);
    if (opt._loglevel < -1)
    {
        auto tr_minmax = std::minmax_element(_translucency_cropped.begin(), _translucency_cropped.end());
        auto ior_minmax = std::minmax_element(_ior.begin(), _ior.end());
        auto iorl_minmax = std::minmax_element(_ior_log.begin(), _ior_log.end());
        for (size_t i = 0; i < _diff.size(); ++i)
        {
            auto diff_minmax = std::minmax_element(_diff[i].begin(), _diff[i].end());
            std::cout << "diff" << i << " (" << *diff_minmax.first << ' ' << *diff_minmax.second << ") ";
        }
        std::cout << "tr ("<< *tr_minmax.first << ' ' << *tr_minmax.second << ") ior (" << *ior_minmax.first << ' ' << *ior_minmax.second << ") iorl (" << *iorl_minmax.first << ' ' << *iorl_minmax.second << ") outputsize: " << _calculation_object->_output_sizes.size() << std::endl;
    }
}

template <typename IorType, typename IorLogType, typename DiffType>
template <typename DirType>
void RaytraceScene<IorType, IorLogType, DiffType>::trace_rays(
        std::vector<pos_t> start_position,
        std::vector<DirType> start_direction,
        std::vector<pos_t> & end_position,
        std::vector<DirType> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        std::vector<float> const & scale,
        brightness_t minimum_brightness,
        uint32_t iterations,
        bool trace_path,
        bool normalize_length,
        Options const & opt)
{
    std::cout << _calculation_object->_output_sizes.size() << std::endl;
    size_t dim = _bound_vec.size();
    
    size_t num_rays = start_position.size() / dim;
    if (num_rays * dim != start_position.size() || num_rays * dim != start_direction.size())
    {
        throw std::runtime_error("raycounts doesn't match, dimension is: " + std::to_string(dim) + " raysizes are start_position: " + std::to_string(start_position.size()) + " start_direction: " + std::to_string(start_direction.size()));
    }
    
    if (normalize_length || true)
    {
        //std::vector<ior_t> interpolated=interpolate(_ior, start_position, _bound_vec);
    std::exception_ptr eptr;
        
#pragma omp parallel if (num_rays > 0x100)
        {
            interpolator<IorType> interp(_ior, _bound_vec);
    #pragma omp for
            for (size_t i = 0; i < num_rays * dim; i += dim)
            {
                if (eptr)
                {
                    continue;
                }
                try{
                    if (UTIL::any_of(start_position.begin() + i, start_position.begin() + i + dim, _bound_vec.begin(), [](size_t lhs, size_t rhs){return lhs < 0x10000 || lhs + 1 >= rhs * 0x10000;}))
                    {
                        std::stringstream ss;
                        print_elements(print_elements(ss << "ray " << (i / dim) << ':', start_position.begin() + i, start_position.begin() + i + dim, ' ', [](std::ostream & out, size_t elem)->std::ostream&{return out << (elem / 0x10000);}) << " is not in 0 to ", _bound_vec.begin(),_bound_vec.end(), ' ');
                        throw std::runtime_error(ss.str());
                    }
                    std::transform(start_position.begin() + i, start_position.begin() + i + dim, start_position.begin() + i, UTIL::minus<pos_t>(0x8000));
                    IorType interpolated = interp(start_position.begin() + i);
                    for (auto iter = start_direction.begin() + i; iter != start_direction.begin() + i + dim; ++iter)
                    {    
                        if (std::is_same<IorType, float>::value)
                        {
                            *iter *= interpolated / 0x100;
                        }
                        else
                        {
                            int64_t tmp = divRoundClosest(static_cast<int64_t>(*iter)  * static_cast<int64_t>(interpolated), static_cast<int64_t>(0x10000));
                            if (tmp > std::numeric_limits<dir_t>::max() || tmp < std::numeric_limits<dir_t>::lowest())
                            {
                                throw std::runtime_error("Normalize length failed: " + std::to_string(std::numeric_limits<dir_t>::lowest()) + "<=" + std::to_string(tmp) + "<="+ std::to_string(std::numeric_limits<dir_t>::max()));
                            }
                            *iter = tmp;
                        }
                    }
                    std::transform(start_position.begin() + i, start_position.begin() + i + dim, start_position.begin() + i, UTIL::minus<pos_t>(0x8000));
                }catch(...) {
                    std::cout << "catched exception" << std::endl;  
                    #pragma omp critical
                    {
                        eptr = std::current_exception();
                    }
                }
            }
        }
        if (eptr) {
            std::rethrow_exception(eptr);
        }
    }
    else
    { 
        for (size_t i = 0; i < num_rays * dim; i += dim)
        {
            if (UTIL::any_of(start_position.begin() + i, start_position.begin() + i + dim, _bound_vec.begin(), [](size_t lhs, size_t rhs){return lhs < 0x10000 || lhs + 1 >= rhs * 0x10000;}))
            {
                std::stringstream ss;
                print_elements(print_elements(ss << "ray " << (i / dim) << ':', start_position.begin() + i, start_position.begin() + i + dim, ' ', [](std::ostream & out, size_t elem)->std::ostream&{return out << (elem / 0x10000);}) << " is not in 0 to ", _bound_vec.begin(),_bound_vec.end(), ' ');
                throw std::runtime_error(ss.str());
            }
        }
        std::transform(start_position.begin(), start_position.end(), start_position.begin(), UTIL::minus<pos_t>(0x10000));
    }

    end_position.resize(num_rays * dim);
    end_direction.resize(num_rays * dim);
    remaining_light.resize(num_rays);

#ifndef NDEBUG            
    export_image_stack("differentiations/translucency_orig", _translucency, _bound_vec);
    //export_image_stack("differentiations/translucency", _translucency_cropped, _output_size);
#endif
    if (opt._loglevel < -2)
    {
        print_elements(std::cout << "start_position: ", start_position.begin(), start_position.end(), ' ') << std::endl;
        print_elements(std::cout << "start_direction:", start_direction.begin(), start_direction.end(), ' ') << std::endl;
    }
     _calculation_object->trace_rays_cu(start_position, start_direction, end_position, end_direction, remaining_light, path, scale, minimum_brightness, iterations, trace_path, opt);
    if (opt._loglevel < -2)
    {
        print_elements(std::cout << "end_position: ", end_position.begin(), end_position.end(), ' ') << std::endl;
        print_elements(std::cout << "end_direction:", end_direction.begin(), end_direction.end(), ' ') << std::endl;
    }
    std::transform(path.begin(), path.end(), path.begin(), UTIL::plus<pos_t>(0x10000));
    std::transform(end_position.begin(), end_position.end(), end_position.begin(), UTIL::plus<pos_t>(0x10000));
}


template <typename IorType, typename IorLogType, typename DiffType, typename DirType>
void trace_rays(
    std::vector<size_t> const & bound_vec,
    std::vector<IorType> ior,
    std::vector<translucency_t> const & translucency,
    std::vector<pos_t> start_position,
    std::vector<DirType> start_direction,
    std::vector<pos_t> & end_position,
    std::vector<DirType> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    std::vector<float> const & scale,
    brightness_t minimum_brightness,
    uint32_t iterations,
    bool trace_path,
    bool normalize_length,
    Options const & opt)
{
    RaytraceScene<IorType, IorLogType, DiffType>(bound_vec, ior, translucency, opt).trace_rays(start_position, start_direction, end_position, end_direction, remaining_light, path, scale, minimum_brightness, iterations, trace_path, normalize_length, opt);
}

template <typename IorType, typename IorLogType, typename DiffType>
template <typename DirType>
void RaytraceScene<IorType, IorLogType, DiffType>::trace_rays(
        RayTraceRayInstanceRef<DirType> const & ref,
        std::vector<pos_t> & end_position,
        std::vector<DirType> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        Options const & opt)
{
    trace_rays(
        ref._start_position,
        ref._start_direction,
        end_position,
        end_direction,
        remaining_light,
        path,
        ref._scale,
        ref._minimum_brightness,
        ref._iterations,
        ref._trace_path,
        ref._normalize_length,
        opt);
}

template <typename IorType, typename IorLogType, typename DiffType>
template <typename DirType>
void RaytraceScene<IorType, IorLogType, DiffType>::trace_rays(
    RayTraceRayInstance<DirType> const & ref,
    std::vector<pos_t> & end_position,
    std::vector<DirType> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt)
{
    trace_rays(
        RayTraceRayInstanceRef<DirType>(const_cast<RayTraceRayInstance<DirType> &>(ref)),
        end_position,
        end_direction,
        remaining_light,
        path,
        opt);               
}


template <typename  IorType, typename IorLogType, typename DiffType, typename DirType>   
void trace_rays(
    RaytraceInstanceRef<IorType, DirType> const & inst,
    std::vector<pos_t> & end_position,
    std::vector<DirType> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt)
{
    trace_rays<IorType, IorLogType, DiffType, DirType>(
        inst._bound_vec,
        inst._ior,
        inst._translucency,
        inst._start_position,
        inst._start_direction,
        end_position,
        end_direction,
        remaining_light,
        path,
        inst._scale,
        inst._minimum_brightness,
        inst._iterations,
        inst._trace_path,
        inst._normalize_length,
        opt);
}

template <typename  IorType, typename IorLogType, typename DiffType, typename DirType>   
void trace_rays(
    RaytraceInstance<IorType, DirType> const & inst,
    std::vector<pos_t> & end_position,
    std::vector<DirType> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt)
{
    trace_rays<IorType, IorLogType, DiffType, DirType>(
        RaytraceInstanceRef<IorType, DirType>(const_cast<RaytraceInstance<IorType, DirType> & >(inst)),
        end_position,
        end_direction,
        remaining_light,
        path,
        opt);
}

template void trace_rays<ior_t, iorlog_t, diff_t, dir_t>(
    RaytraceInstance<ior_t, dir_t> const & inst,
    std::vector<pos_t> & end_position,
    std::vector<dir_t> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt);


template void trace_rays<float, float, float, float>(
    RaytraceInstance<float, float> const & inst,
    std::vector<pos_t> & end_position,
    std::vector<float> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt);

/*template void RaytraceScene<ior_t, iorlog_t, diff_t>::trace_rays(
    RayTraceRayInstance<dir_t> const & ref,
    std::vector<pos_t> & end_position,
    std::vector<dir_t> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt);*/

template void RaytraceScene<float, float, float>::trace_rays<float>(
    RayTraceRayInstance<float> const & ref,
    std::vector<pos_t> & end_position,
    std::vector<float> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt);

template void RaytraceScene<ior_t, iorlog_t, diff_t>::trace_rays<dir_t>(
    RayTraceRayInstanceRef<dir_t> const & ref,
    std::vector<pos_t> & end_position,
    std::vector<dir_t> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt);
