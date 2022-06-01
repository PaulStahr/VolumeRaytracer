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
        std::cout << ray_inst._invscale.size() << std::endl;
        std::cout << ray_inst._invscale[0] << std::endl;
        Options opt;
        opt._loglevel = -4;
        opt._minimum_gpu = 0;
        RaytraceScene<ior_t, iorlog_t, diff_t> scene(scene_inst, opt);
        
        std::vector<pos_t> end_position;
        std::vector<dir_t> end_direction;
        std::vector<uint32_t> end_iteration;
        std::vector<uint32_t> remaining_light;
        std::vector<pos_t> path;
        
        scene.trace_rays(RayTraceRayInstanceRef<dir_t>(ray_inst), end_position, end_direction, end_iteration, remaining_light, path, opt);
        
        print_elements(std::cout << "begin ", ray_inst._start_position.begin(), ray_inst._start_position.end(), ' ') << std::endl;
        print_elements(std::cout << "end ", end_position.begin(), end_position.end(), ' ') << std::endl;
        return 0;
    }
    if (argc >= 2)
    {
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
    std::vector<uint32_t> end_iteration;
    std::vector<uint32_t> remaining_light;
    std::vector<pos_t> path;
    print_elements(std::cout << "bounds: ", inst._bound_vec.begin(), inst._bound_vec.end(), ' ') << std::endl;
    trace_rays<ior_t, iorlog_t, diff_t, dir_t>(inst, end_position, end_direction, end_iteration, remaining_light, path, Options());
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
