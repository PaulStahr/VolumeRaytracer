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

#include <inttypes.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>
#include "io_util.h"
#include "image_util.h"
#include "types.h"
#include "serialize.h"

template <uint64_t divisor>
struct print_div_struct
{
    print_div_struct(){}
    
    template <typename T>
    std::ostream & operator ()(std::ostream & out, T elem) const
    {
        return out << static_cast<double>(elem) / divisor;
    }
};

template <uint64_t divisor>
static const print_div_struct<divisor> print_div;

void interpolation_test()
{
    std::vector<size_t> bounds({5,5,5});
    std::vector<iorlog_t> values(std::accumulate(bounds.begin(), bounds.end(), size_t(1), std::multiplies<size_t>()));
    std::vector<pos_t> pos({
        0x10000, 0x10000, 0x10000,
        0x18000, 0x10000, 0x10000,
        0x10000, 0x18000, 0x10000,
        0x10000, 0x10000, 0x18000,
        0x18000, 0x18000, 0x18000,
        0x20000, 0x10000, 0x10000,
        0x10000, 0x20000, 0x10000,
        0x10000, 0x10000, 0x20000,
        0x20000, 0x20000, 0x20000});
    
    for (size_t div = 1; div < 125; div *= 5)
    {
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = 100 * ((i / div) % 5);
        }
        interpolator<iorlog_t> interp(values, bounds);
        std::vector<iorlog_t> interpolation;
        for (size_t i = 0; i < pos.size(); i += 3)
        {
            interpolation.push_back(interp(pos.begin() + i));
        }
        print_elements(std::cout << "interpolation: ", interpolation.begin(), interpolation.end(), ' ') << std::endl;
    }
}

void scaling_test2()
{
    std::cout << "scaling_test" << std::endl;
    RaytraceInstance<float, float> inst;
    inst._bound_vec = std::vector<size_t>({1000,10,10});
    size_t num_pixel = std::accumulate(inst._bound_vec.begin(), inst._bound_vec.end(), size_t(1), std::multiplies<size_t>());
    inst._ior = std::vector<float>(num_pixel, 0);
    inst._translucency = std::vector<uint32_t>(num_pixel, 0xFFFFFFFF);
    inst._start_position = std::vector<pos_t>({0x10000,0x10000,0x10000});
    inst._start_direction = std::vector<float>({0x1800,0,0});
    inst._scale = std::vector<float>({10,10,10});
    inst._iterations = 100000;
    inst._trace_path = true;
    inst._minimum_brightness = 0;
    size_t num_layer_pixel = inst._bound_vec[1] * inst._bound_vec[2];
    for (size_t i = 0; i < inst._bound_vec[0]; ++i)
    {
        std::fill(inst._ior.begin() + i * num_layer_pixel, inst._ior.begin() + (i + 1) * num_layer_pixel, 0x10000 + (i * 0x10000 / (inst._bound_vec[0] - 1)));
    }
    std::vector<pos_t> end_position;
    std::vector<float> end_direction;
    std::vector<uint32_t> remaining_light;
    std::vector<pos_t> path;    
    print_elements(std::cout << " beginposition ", inst._start_position.begin(), inst._start_position.end(), ' ', print_div<0x10000>) << std::endl;
    print_elements(std::cout << " begindirection ", inst._start_direction.begin(), inst._start_direction.end(), ' ', print_div<0x100>) << std::endl;
    trace_rays<float, float, float, float>(inst, end_position, end_direction, remaining_light, path, Options());
    std::cout << static_cast<double>(end_direction[0]) / inst._start_direction[0] << std::endl;
    print_elements(std::cout << " endposition ", end_position.begin(), end_position.end(), ' ', print_div<0x10000>) << std::endl;
    print_elements(std::cout << " enddirection ", end_direction.begin(), end_direction.end(), ' ', print_div<0x100>) << std::endl;
    std::cout << "path " << std::endl;
    for (size_t i = 0; i < path.size(); i += 9000)
    {
        print_elements(std::cout, path.rbegin() + i, path.rbegin() + i + 3, ' ', print_div<0x10000>) << std::endl;
    }
}

void scaling_test()
{
    std::cout << "scaling_test" << std::endl;
    RaytraceInstance<ior_t, dir_t> inst;
    inst._bound_vec = std::vector<size_t>({1000,10,10});
    size_t num_pixel = std::accumulate(inst._bound_vec.begin(), inst._bound_vec.end(), size_t(1), std::multiplies<size_t>());
    inst._ior = std::vector<ior_t>(num_pixel, 0);
    inst._translucency = std::vector<uint32_t>(num_pixel, 0xFFFFFFFF);
    inst._start_position = std::vector<pos_t>({0x10000,0x10000,0x10000});
    inst._start_direction = std::vector<dir_t>({0x1800,0,0});
    inst._scale = std::vector<float>({10,10,10});
    inst._iterations = 100000;
    inst._trace_path = true;
    inst._minimum_brightness = 0;
    size_t num_layer_pixel = inst._bound_vec[1] * inst._bound_vec[2];
    for (size_t i = 0; i < inst._bound_vec[0]; ++i)
    {
        std::fill(inst._ior.begin() + i * num_layer_pixel, inst._ior.begin() + (i + 1) * num_layer_pixel, 0x10000 + (i * 0x10000 / (inst._bound_vec[0] - 1)));
    }
    std::vector<pos_t> end_position;
    std::vector<dir_t> end_direction;
    std::vector<uint32_t> remaining_light;
    std::vector<pos_t> path;    
    print_elements(std::cout << " beginposition ", inst._start_position.begin(), inst._start_position.end(), ' ', print_div<0x10000>) << std::endl;
    print_elements(std::cout << " begindirection ", inst._start_direction.begin(), inst._start_direction.end(), ' ', print_div<0x100>) << std::endl;
    trace_rays<ior_t, iorlog_t, diff_t, dir_t>(inst, end_position, end_direction, remaining_light, path, Options());
    std::cout << static_cast<double>(end_direction[0]) / inst._start_direction[0] << std::endl;
    print_elements(std::cout << " endposition ", end_position.begin(), end_position.end(), ' ', print_div<0x10000>) << std::endl;
    print_elements(std::cout << " enddirection ", end_direction.begin(), end_direction.end(), ' ', print_div<0x100>) << std::endl;
    std::cout << "path " << std::endl;
    for (size_t i = 0; i < path.size(); i += 9000)
    {
        print_elements(std::cout, path.rbegin() + i, path.rbegin() + i + 3, ' ', print_div<0x10000>) << std::endl;
    }
}

int main(int argc, char* argv[])
{
    RaytraceInstance<ior_t, dir_t> inst;
    if (argc >= 3)
    {
        RayTraceSceneInstance<ior_t> scene_inst;
        RayTraceRayInstance<dir_t> ray_inst;
    
        {
            std::cout << "scene" << std::endl;
            std::ifstream stream(argv[1]);
            SERIALIZE::read_value(stream, scene_inst);
            stream.close();
        }

        {
            std::cout << "ray" << std::endl;
            std::ifstream stream(argv[2]);
            SERIALIZE::read_value(stream, ray_inst);
            stream.close();
        }
        std::cout << ray_inst._scale.size() << std::endl;
        std::cout << ray_inst._scale[0] << std::endl;
        Options opt;
        opt._loglevel = -4;
        RaytraceScene<ior_t, iorlog_t, diff_t> scene(scene_inst, opt);
        
        std::vector<pos_t> end_position;
        std::vector<dir_t> end_direction;
        std::vector<uint32_t> remaining_light;
        std::vector<pos_t> path;
        
        scene.trace_rays(RayTraceRayInstanceRef<dir_t>(ray_inst), end_position, end_direction, remaining_light, path, opt);
        
        print_elements(std::cout << "begin ", ray_inst._start_position.begin(), ray_inst._start_position.end(), ' ') << std::endl;
        print_elements(std::cout << "end ", end_position.begin(), end_position.end(), ' ') << std::endl;
        return 0;
    }
    if (argc >= 2)
    {
        if (std::string("#s") == argv[1])
        {
            scaling_test();
            return 0;
        }
        else if (std::string("#i") == argv[1])
        {
            interpolation_test();
            return 0;
        }
        
        std::ifstream stream(argv[1]);
        SERIALIZE::read_value(stream, inst);
    }
    else
    {
        inst._bound_vec = std::vector<size_t>({100,100,100});
        inst._ior = std::vector<ior_t>(1000000);
        inst._translucency = std::vector<uint32_t>(1000000);
        inst._start_position = std::vector<pos_t>(10000);
        inst._start_direction = std::vector<dir_t>({1,0,0});
        inst._iterations = 100;
        inst._trace_path = true;
        inst._minimum_brightness = 100;
        
    }
    size_t dim = inst._bound_vec.size();
    std::vector<pos_t> end_position;
    std::vector<dir_t> end_direction;
    std::vector<uint32_t> remaining_light;
    std::vector<pos_t> path;
    print_elements(std::cout << "bounds: ", inst._bound_vec.begin(), inst._bound_vec.end(), ' ') << std::endl;
    trace_rays<ior_t, iorlog_t, diff_t, dir_t>(inst, end_position, end_direction, remaining_light, path, Options());
    if (end_position.size() < 1000)
    {
        print_elements(std::cout << "end position:", end_position.begin(), end_position.end(),' ') << std::endl;
        print_elements(std::cout << "end direction:", end_direction.begin(), end_direction.end(),' ') << std::endl;
    }
    std::cout << "path" << std::endl;
    if (inst._trace_path)
    {
        for (size_t i = 0; i < inst._iterations; ++i)
        {
            print_elements(std::cout << i << ':', path.begin() + i * dim, path.begin() + (i + 1) * dim, ' ');
            print_elements(std::cout << " <-> ", path.begin() + i * dim, path.begin() + (i + 1) * dim, ' ', print_div<0x10000>) << std::endl;
        }
    }
    
    return 0;
}
