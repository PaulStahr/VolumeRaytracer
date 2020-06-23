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

#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H

#include <vector>
#include <cinttypes>
#include <numeric>
#include "types.h"
#include "util.h"
#include "io_util.h"
#include "serialize.h"


template <typename T>
T inline divRoundClosest(const T n, const T d)
{
  return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);
}

template <typename DiffType>
class TraceRaysCu;

template <typename IorType>
struct RayTraceSceneInstance;

template <typename IorType>
struct RayTraceSceneInstanceRef
{
    std::vector<size_t> const & _bound_vec;
    std::vector<IorType> const & _ior;
    std::vector<translucency_t> const& _translucency;
    RayTraceSceneInstanceRef(std::vector<size_t> const & bound_vec_, std::vector<ior_t> const & ior_, std::vector<translucency_t> const & translucency_) : _bound_vec(bound_vec_), _ior(ior_), _translucency(translucency_){}
    
    RayTraceSceneInstanceRef(RayTraceSceneInstance<IorType> & ref);
};

template <typename IorType>
struct RayTraceSceneInstance
{
    std::vector<size_t> _bound_vec;
    std::vector<IorType> _ior;
    std::vector<translucency_t> _translucency;
    
    RayTraceSceneInstance(std::vector<size_t> const & bound_vec_, std::vector<ior_t> const & ior_, std::vector<translucency_t> const & translucency_) : _bound_vec(bound_vec_), _ior(ior_), _translucency(translucency_){}
    
    RayTraceSceneInstance() {}

    RayTraceSceneInstance(RayTraceSceneInstanceRef<IorType> const & ref) : _bound_vec(ref._bound_vec), _ior(ref._ior), _translucency(ref._translucency){}
};

template <typename DirType>
struct RayTraceRayInstance
{
    std::vector<pos_t>     _start_position;
    std::vector<DirType>   _start_direction;
    std::vector<float>     _invscale;
    brightness_t           _minimum_brightness;
    uint32_t               _iterations;
    bool                   _trace_path;
    bool                   _normalize_length;
    
    RayTraceRayInstance(){}
};

template <typename DirType>
struct RayTraceRayInstanceRef
{
    std::vector<pos_t>      const & _start_position;
    std::vector<DirType>    const & _start_direction;
    std::vector<float>      const & _invscale;
    brightness_t            _minimum_brightness;
    uint32_t                _iterations;
    bool                    _trace_path;
    bool                    _normalize_length;
    
    RayTraceRayInstanceRef(
        std::vector<pos_t>     const & start_position_,
        std::vector<DirType>   const & start_direction_,
        std::vector<float>     const & invscale_,
        brightness_t             minimum_brightness_,
        uint32_t                 iterations_,
        bool                     trace_path_,
        bool                    normalize_length_
                       ) : 
        _start_position(start_position_),
        _start_direction(start_direction_),
        _invscale(invscale_),
        _minimum_brightness(minimum_brightness_),
        _iterations(iterations_),
        _trace_path(trace_path_),
        _normalize_length(normalize_length_){}
        
    RayTraceRayInstanceRef(RayTraceRayInstance<DirType> & other) : 
        _start_position(other._start_position),
        _start_direction(other._start_direction),
        _invscale(other._invscale),
        _minimum_brightness(other._minimum_brightness),
        _iterations(other._iterations),
        _trace_path(other._trace_path),
        _normalize_length(other._normalize_length){}
};

class RaytraceSceneBase{
public:
    void virtual dummy() = 0;
    
    virtual ~RaytraceSceneBase(){}
};

template <typename IorType, typename IorLogType, typename DiffType>
class RaytraceScene: public RaytraceSceneBase
{
public:
    private:
    std::vector<size_t> const _bound_vec;
    std::vector<IorType> const _ior;
    std::vector<translucency_t> const _translucency;
    std::vector<std::vector<DiffType> > _diff;
    std::vector<IorLogType> _ior_log;
    std::vector<uint32_t> _translucency_cropped;
    std::vector<size_t> _diff_bound_vec;
    TraceRaysCu<DiffType> *_calculation_object;
public:
    RaytraceScene(
        std::vector<size_t> const & bound_vec,
        std::vector<IorType> const & _ior,
        std::vector<translucency_t> const & _translucency,
        Options const & opt);
    
    RaytraceScene(
        RayTraceSceneInstanceRef<IorType> const & ref,
        Options const & opt);
    
    void dummy(){}
    
    template <typename DirType>
    void trace_rays(
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
        Options const & opt);
    
    template <typename DirType>
    void trace_rays(
        RayTraceRayInstanceRef<DirType> const & ref,
        std::vector<pos_t> & end_position,
        std::vector<DirType> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        Options const & opt);
    
    template <typename DirType>
    void trace_rays(
        RayTraceRayInstance<DirType> const & ref,
        std::vector<pos_t> & end_position,
        std::vector<DirType> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        Options const & opt);
};

struct RaytraceInstanceBase{};

template <typename IorType, typename DirType>
struct RaytraceInstance : RaytraceInstanceBase
{
    std::vector<size_t>     _bound_vec;
    std::vector<IorType>    _ior;
    std::vector<translucency_t>   _translucency;
    std::vector<pos_t>      _start_position;
    std::vector<DirType>    _start_direction;
    std::vector<float>      _invscale;
    brightness_t            _minimum_brightness;
    uint32_t                _iterations;
    bool                    _trace_path;
    bool                    _normalize_length;
    
    RaytraceInstance(){}
};

template <typename IorType, typename DirType>
struct RaytraceInstanceRef
{
    std::vector<size_t>    const & _bound_vec;
    std::vector<IorType>   const & _ior;
    std::vector<translucency_t>  const & _translucency;
    std::vector<pos_t>     const & _start_position;
    std::vector<DirType>   const & _start_direction;
    std::vector<float>     const & _invscale;
    brightness_t           _minimum_brightness;
    uint32_t               _iterations;
    bool                   _trace_path;
    bool                   _normalize_length;
    
    RaytraceInstanceRef(
        std::vector<size_t>    const & bound_vec_,
        std::vector<IorType>   const & ior_,
        std::vector<translucency_t>  const & translucency_,
        std::vector<pos_t>     const & start_position_,
        std::vector<DirType>   const & start_direction_,
        std::vector<float>     const & scale_,
        brightness_t             minimum_brightness_,
        uint32_t                 iterations_,
        bool                     trace_path_,
        bool                    normalize_length_
                       ) : 
        _bound_vec(bound_vec_),
        _ior(ior_),
        _translucency(translucency_),
        _start_position(start_position_),
        _start_direction(start_direction_),
        _invscale(scale_),
        _minimum_brightness(minimum_brightness_),
        _iterations(iterations_),
        _trace_path(trace_path_),
        _normalize_length(normalize_length_){}
        
        
    
    RaytraceInstanceRef(RaytraceInstance<IorType, DirType> & other) : 
        _bound_vec(other._bound_vec),
        _ior(other._ior),
        _translucency(other._translucency),
        _start_position(other._start_position),
        _start_direction(other._start_direction),
        _invscale(other._invscale),
        _minimum_brightness(other._minimum_brightness),
        _iterations(other._iterations),
        _trace_path(other._trace_path),
        _normalize_length(other._normalize_length){}
};

namespace SERIALIZE{
template <typename IorType, typename DirType>
std::ostream & write_value(std::ostream & out, RaytraceInstanceRef<IorType, IorType> const & value);

template <typename IorType, typename DirType>
std::ostream & write_value(std::ostream & out, RaytraceInstance<IorType, DirType> const & value);

template <typename IorType, typename DirType>
std::istream & read_value(std::istream & in, RaytraceInstance<IorType, DirType> & value);

template <typename IorType>
std::ostream & write_value(std::ostream & out, RayTraceSceneInstanceRef<IorType> const & value);

template <typename IorType>
std::ostream & write_value(std::ostream & out, RayTraceSceneInstance<IorType> const & value);

template <typename IorType>
std::istream & read_value(std::istream & in, RayTraceSceneInstance<IorType> & value);

/*template <typename DirType>
std::ostream & write_value(std::ostream & out, RayTraceRayInstanceRef<DirType> const & value);

template <typename DirType>
std::ostream & write_value(std::ostream & out, RayTraceRayInstance<DirType> const & value);
*/

template <typename DirType>
std::ostream & write_value(std::ostream & out, RayTraceRayInstanceRef<DirType> const & value)
{
    write_value(out, value._start_position);
    write_value(out, value._start_direction);
    write_value(out, value._invscale);
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
}

template <typename DirType>
std::istream & read_value(std::istream & in, RayTraceRayInstance<DirType> & value);
}

template <typename BoundIter, typename PosIter>
size_t get_index(BoundIter bound_begin, BoundIter bound_end, PosIter pos)
{
    if (bound_begin == bound_end)
    {
        return 0;
    }
    size_t index = *pos;
    ++pos;
    ++bound_begin;
    while (bound_begin != bound_end)
    {
        index *= *bound_begin;
        index += *pos;
        ++bound_begin;
        ++pos;
    }
    return index;
}

class interpolator_base
{
public:
    size_t _dim;
    std::vector<size_t> _indices;
    interpolator_base(size_t dim_) : _dim(dim_){
        _indices.reserve(1lu << _dim);
    }
};

template <typename T>
class interpolator : interpolator_base
{
private:
    std::vector<T> const & _image;
    std::vector<size_t> const & _bound_vec;
    std::vector<T> _image_values;
public:    
    interpolator(std::vector<T> const & image_, std::vector<size_t> const & bound_vec_) : interpolator_base(bound_vec_.size()), _image(image_), _bound_vec(bound_vec_)
    {
        if (image_.size() != std::accumulate(bound_vec_.begin(), bound_vec_.end(), size_t(1), std::multiplies<size_t>()))
        {
            throw std::runtime_error("image_size doensn't match bounds:" + std::to_string(image_.size()) + "!=" + std::to_string(std::accumulate(bound_vec_.begin(), bound_vec_.end(), size_t(1), std::multiplies<size_t>())));
        }
        _image_values.reserve(1lu << _dim);
    }
    
    template <typename PositionIterator>
    T operator()(PositionIterator positions)
    {
        _image_values.clear();
        _indices.clear();
        _indices.emplace_back(get_index(_bound_vec.begin(), _bound_vec.end(), UTIL::transform_iter(positions, UTIL::shift_right(16))));
        for (size_t d = _dim,offset=1; d --> 0;offset *= _bound_vec[d])
        {
            std::transform(_indices.begin(), _indices.end(), std::back_inserter(_indices), UTIL::plus(offset));
        }
        UTIL::permutate_from_indice(_indices.begin(), _indices.end(), _image.begin(), std::back_inserter(_image_values));
        for (size_t d = 0; d < _dim; ++d)
        {
            size_t multr = positions[d] % 0x10000;
            size_t multl = 0x10000 - multr;
            for (size_t i = 0; i < _image_values.size()/2; ++i)
            {
                _image_values[i] = divRoundClosest(static_cast<uint64_t>(_image_values[i]) * multl + static_cast<uint64_t>(_image_values[i + _image_values.size() / 2]) * multr,static_cast<uint64_t>(0x10000));
            }
            _image_values.erase(_image_values.begin() + _image_values.size() / 2, _image_values.end());
        }
        return _image_values.front();
    }
};

template <>
class interpolator<float> : interpolator_base
{
private:
    std::vector<float> const & _image;
    std::vector<size_t> const & _bound_vec;
    std::vector<float> _image_values;
    float _end_multiplier;
public:    
    interpolator(std::vector<float> const & image_, std::vector<size_t> const & bound_vec_) : interpolator_base(bound_vec_.size()), _image(image_), _bound_vec(bound_vec_), _end_multiplier(1./pow(0x10000,bound_vec_.size()))
    {
        if (image_.size() != std::accumulate(bound_vec_.begin(), bound_vec_.end(), size_t(1), std::multiplies<size_t>()))
        {
            throw std::runtime_error("image_size doensn't match bounds:" + std::to_string(image_.size()) + "!=" + std::to_string(std::accumulate(bound_vec_.begin(), bound_vec_.end(), size_t(1), std::multiplies<size_t>())));
        }
        _image_values.reserve(1lu << _dim);
    }
    
    template <typename PositionIterator>
    float operator()(PositionIterator positions)
    {
        _image_values.clear();
        _indices.clear();
        _indices.emplace_back(get_index(_bound_vec.begin(), _bound_vec.end(), UTIL::transform_iter(positions, UTIL::shift_right(16))));
        for (size_t d = _dim,offset=1; d --> 0;offset *= _bound_vec[d])
        {
            std::transform(_indices.begin(), _indices.end(), std::back_inserter(_indices), UTIL::plus(offset));
        }
        UTIL::permutate_from_indice(_indices.begin(), _indices.end(), _image.begin(), std::back_inserter(_image_values));
        for (size_t d = 0; d < _dim; ++d)
        {
            size_t multr = positions[d] % 0x10000;
            size_t multl = 0x10000 - multr;
            for (size_t i = 0; i < _image_values.size()/2; ++i)
            {
                _image_values[i] = _image_values[i] * multl + _image_values[i + _image_values.size() / 2] * multr;
            }
            _image_values.erase(_image_values.begin() + _image_values.size() / 2, _image_values.end());
        }
        return _image_values.front() * _end_multiplier;
    }
};



void get_position(size_t index, std::vector<size_t> const & bounds, std::vector<size_t> & pos);

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
    Options const & opt);

template <typename IorType, typename IorLogType, typename DiffType, typename DirType>
void trace_rays(
    RaytraceInstanceRef<IorType, DirType> const & inst,
    std::vector<pos_t> & end_position,
    std::vector<DirType> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt);

template <typename IorType, typename IorLogType, typename DiffType, typename DirType>
void trace_rays(
    RaytraceInstance<IorType, DirType> const & inst,
    std::vector<pos_t> & end_position,
    std::vector<DirType> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    Options const & opt);
#endif
