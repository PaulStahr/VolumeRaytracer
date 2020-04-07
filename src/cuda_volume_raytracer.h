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

#ifndef CUDA_VOLUME_RAYTRACER_H
#define CUDA_VOLUME_RAYTRACER_H

#include <cinttypes>
#include <vector>
#include "types.h"
#include "tuple_declaration.h"

void trace_rays_cu(
    std::vector<size_t> const & output_sizes,
    std::vector<diff_t> const & diffx,
    std::vector<diff_t> const & diffy,
    std::vector<translucency_t> const & transculency_cropped,
    std::vector<pos_t> const & start_position,
    std::vector<dir_t> const & start_direction,
    std::vector<pos_t> & end_position,
    std::vector<dir_t> & end_direction,
    std::vector<brightness_t> & remaining_light,
    std::vector<pos_t> & path,
    std::vector<float> const & scale,
    brightness_t minimum_brightness,
    uint32_t iterations,
    bool trace_path,
    Options const & opt);

//class TraceRaysCuBase
//{};

template <typename DiffType>
class TraceRaysCu
{
    private:
    std::vector<translucency_t> const & _translucency_cropped;
    std::vector<uint32_t *> _translucency_cuda;
    std::vector<DiffType> _diff_interleaved;
    std::vector<DiffType *>_diff_interleaved_cuda;
    
    public:
    std::vector<uint16_t> _output_sizes;
    TraceRaysCu(
        std::vector<size_t> const & output_sizes_,
        std::vector<std::vector<DiffType> const *> const & diff_,
        std::vector<translucency_t> const & translucency_cropped_);

    TraceRaysCu(
        std::vector<size_t> const & output_sizes_,
        std::vector<std::vector<DiffType> > const & diff_,
        std::vector<translucency_t> const & translucency_cropped_);

    template <typename DirType>
    void trace_rays_cu(
        std::vector<pos_t> const & start_position,
        std::vector<DirType> const & start_direction,
        std::vector<pos_t> & end_position,
        std::vector<DirType> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        std::vector<float> const & scale_vec,
        brightness_t minimum_brightness,
        uint32_t iterations,
        bool trace_paths,
        Options const & opt);
    
    template <typename DirType, uint8_t dim>
    void trace_rays_cu_impl(
        std::vector<pos_t> const & start_position,
        std::vector<DirType> const & start_direction,
        std::vector<pos_t> & end_position,
        std::vector<DirType> & end_direction,
        std::vector<brightness_t> & remaining_light,
        std::vector<pos_t> & path,
        std::vector<float> const & scale_vec,
        brightness_t minimum_brightness,
        uint32_t iterations,
        bool trace_paths,
        Options const & opt);
    
    ~TraceRaysCu();
};

#endif
