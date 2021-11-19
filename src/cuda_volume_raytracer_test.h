#include "image_util.h"
#include "io_util.h"

template<typename IOR_TYPE, typename IORLOG_TYPE, typename DIR_TYPE, typename DIFF_TYPE>
void run_scaling_test()
{
    Options opt;
    opt._loglevel = -3;
    std::cout << "scaling_test" << std::endl;
    RaytraceInstance<IOR_TYPE, DIR_TYPE> inst;
    inst._bound_vec = std::vector<size_t>({1000,10,10});
    size_t num_pixel = std::accumulate(inst._bound_vec.begin(), inst._bound_vec.end(), size_t(1), std::multiplies<size_t>());
    inst._ior = std::vector<IOR_TYPE>(num_pixel, 0);
    inst._translucency = std::vector<uint32_t>(num_pixel, 0xFFFFFFFF);
    inst._start_position = std::vector<pos_t>({0x10000,0x40000,0x40000,static_cast<pos_t>(0x10000*inst._bound_vec[0] - 0x30000),0x40000,0x40000});
    DIR_TYPE xdir = 0x10 * dir_typeinfo<DIR_TYPE>.unit_value;
    inst._start_direction = std::vector<DIR_TYPE>({xdir,0,0, static_cast<DIR_TYPE>(-xdir), 0, 0});
    inst._invscale = std::vector<float>(3,2);
    inst._iterations = 1000000;
    inst._trace_path = true;
    inst._minimum_brightness = 0;
    size_t num_layer_pixel = inst._bound_vec[1] * inst._bound_vec[2];
    std::fill(inst._ior.begin(), inst._ior.begin() + 10 * num_layer_pixel, ior_typeinfo<IOR_TYPE>.unit_value);
    std::fill(inst._ior.end() - 10 * num_layer_pixel, inst._ior.end(), ior_typeinfo<IOR_TYPE>.unit_value * 2);
    for (size_t i = 10; i < inst._bound_vec[0] - 10; ++i)
    {
         std::fill(inst._ior.begin() + i * num_layer_pixel, inst._ior.begin() + (i + 1) * num_layer_pixel, ior_typeinfo<IOR_TYPE>.unit_value + ior_typeinfo<IOR_TYPE>.unit_value * static_cast<IOR_TYPE>(i) / (inst._bound_vec[0] - 21));
    }
    
    std::vector<pos_t> end_position;
    std::vector<DIR_TYPE> end_direction;
    std::vector<uint32_t> end_iteration;
    std::vector<uint32_t> remaining_light;
    std::vector<pos_t> path;    
    print_elements(std::cout << " beginposition ", inst._start_position.begin(), inst._start_position.end(), ' ', print_div<0x10000>) << std::endl;
    print_elements(std::cout << " begindirection ", inst._start_direction.begin(), inst._start_direction.end(), ' ', print_convert(dir_typeinfo<DIR_TYPE>.to_double)) << std::endl;
    trace_rays<IOR_TYPE, IORLOG_TYPE, DIFF_TYPE, DIR_TYPE>(
        inst,
        end_position,
        end_direction,
        end_iteration,
        remaining_light,
        path,
        opt);
    std::cout << "scaling " << static_cast<double>(end_direction[0]) / inst._start_direction[0] << ' ' << static_cast<double>(end_direction[3]) / inst._start_direction[3] << ' ' << (static_cast<double>(end_direction[0]) / (inst._start_direction[0] * 2) + inst._start_direction[3] / static_cast<double>(end_direction[3])) * 0.5 << std::endl;
    
    interpolator<IOR_TYPE> interp(inst._ior, inst._bound_vec);
    BOOST_TEST(static_cast<double>(end_direction[0]) / inst._start_direction[0] == interp(end_position.begin()) / interp(inst._start_position.begin()),boost::test_tools::tolerance(0.00001 + dir_typeinfo<DIR_TYPE>.tolerance));
    BOOST_TEST(static_cast<float>(end_iteration[0]) == 46718, boost::test_tools::tolerance(static_cast<float>(100))); //TODO give formular for number of iterations
    BOOST_TEST(static_cast<float>(end_iteration[1]) == 46718, boost::test_tools::tolerance(static_cast<float>(100))); //TODO give formular for number of iterations
    print_elements(std::cout << " endposition ", end_position.begin(), end_position.end(), ' ', print_div<0x10000>) << std::endl;
    print_elements(std::cout << " enddirection ", end_direction.begin(), end_direction.end(), ' ', print_convert(dir_typeinfo<DIR_TYPE>.to_double)) << std::endl;
    print_elements(std::cout << " enditeration ", end_iteration.begin(), end_iteration.end(), ' ') << std::endl;
    for (size_t i = 0; i < 2; ++i)
    {
        std::cout << "path " << ' ';
        for (size_t j = 0; j < 50; ++j)
        {
            auto elem = path.rbegin() + ((1 - i) * inst._iterations + (j * end_iteration[i] / 50)) * 3;
            print_elements(std::cout, elem, elem + 3, ' ', print_div<0x10000>) << ' ';
        }
        std::cout << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE(cuda_volume_raytracer_test)

BOOST_AUTO_TEST_CASE( scaling_test )
{
    run_scaling_test<ior_t, iorlog_t, dir_t, diff_t>();
    run_scaling_test<float, float, float, float>();
}
BOOST_AUTO_TEST_SUITE_END()
