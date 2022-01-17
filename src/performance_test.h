#include <chrono>
#include "image_util.h"
#include "io_util.h"
#include "types.h"

#ifndef PERFORMANCE_TEST_H
#define PERFORMANCE_TEST_H

template<typename IOR_TYPE, typename IORLOG_TYPE, typename DIR_TYPE, typename DIFF_TYPE>
void run_performance_test()
{
    Options opt;
    opt._max_cpu = 1;
    opt._loglevel = -3;
    RayTraceSceneInstance<IOR_TYPE> inst;
    RayTraceRayInstance<DIR_TYPE> ray_instance;
    inst._bound_vec = std::vector<size_t>({1000,10,10});
    size_t num_pixel = std::accumulate(inst._bound_vec.begin(), inst._bound_vec.end(), size_t(1), std::multiplies<size_t>());
    inst._ior = std::vector<IOR_TYPE>(num_pixel, 0);
    inst._translucency = std::vector<uint32_t>(num_pixel, 0xFFFFFFFF);
    DIR_TYPE xdir = 0x10 * dir_typeinfo<DIR_TYPE>.unit_value;
    for (size_t i = 0; i < 1000; ++i)
    {
        ray_instance._start_position.push_back(0x10000);
        ray_instance._start_position.push_back(0x40000);
        ray_instance._start_position.push_back(0x40000);
        ray_instance._start_position.push_back(static_cast<pos_t>(0x10000*inst._bound_vec[0] - 0x30000));
        ray_instance._start_position.push_back(0x40000);
        ray_instance._start_position.push_back(0x40000);
        ray_instance._start_direction.push_back(xdir);
        ray_instance._start_direction.push_back(0);
        ray_instance._start_direction.push_back(0);
        ray_instance._start_direction.push_back(-xdir);
        ray_instance._start_direction.push_back(0);
        ray_instance._start_direction.push_back(0);
    }
    ray_instance._invscale = std::vector<float>(3,2);
    ray_instance._iterations = 1000000;
    ray_instance._trace_path = false;
    ray_instance._minimum_brightness = 0;
    opt._loglevel = 100;
    size_t num_layer_pixel = inst._bound_vec[1] * inst._bound_vec[2];
    std::fill(inst._ior.begin(), inst._ior.begin() + 10 * num_layer_pixel, ior_typeinfo<IOR_TYPE>.unit_value);
    std::fill(inst._ior.end() - 10 * num_layer_pixel, inst._ior.end(), ior_typeinfo<IOR_TYPE>.unit_value * 2);
    for (size_t i = 10; i < inst._bound_vec[0] - 10; ++i)
    {
         std::fill(inst._ior.begin() + i * num_layer_pixel, inst._ior.begin() + (i + 1) * num_layer_pixel, ior_typeinfo<IOR_TYPE>.unit_value + ior_typeinfo<IOR_TYPE>.unit_value * static_cast<IOR_TYPE>(i) / (inst._bound_vec[0] - 21));
    }

    RaytraceScene<IOR_TYPE, IORLOG_TYPE, DIFF_TYPE> scene(inst, opt);
    
    std::vector<pos_t> end_position;
    std::vector<DIR_TYPE> end_direction;
    std::vector<uint32_t> end_iteration;
    std::vector<uint32_t> remaining_light;
    std::vector<pos_t> path;
    
    size_t iterations = 1;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i)
    {
        end_position.clear();
        end_direction.clear();
        end_iteration.clear();
        remaining_light.clear();
        scene.trace_rays(
            RayTraceRayInstanceRef<DIR_TYPE>(ray_instance),
            end_position,
            end_direction,
            end_iteration,
            remaining_light,
            path,
            opt);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Rays per time = " << (iterations * ray_instance._start_position.size() / 3) * 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[R/s]" << std::endl;    
}

BOOST_AUTO_TEST_SUITE( performance )

BOOST_AUTO_TEST_CASE( performance_test )
{
    run_performance_test<ior_t, iorlog_t, dir_t, diff_t>();
    run_performance_test<float, float, float, float>();
}
BOOST_AUTO_TEST_SUITE_END()
#endif
