#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <inttypes.h>
#include <vector>
#include "image_util.h"
#include "types.h"
#include "fstream"
#include "serialize.h"

std::tuple<std::vector<pos_t>,std::vector<dir_t>, std::vector<brightness_t>, std::vector<pos_t> > py_trace_rays(
    std::vector<size_t> const & bound_vec,
    std::vector<ior_t> const & ior,
    std::vector<uint32_t> const & transculency,
    std::vector<pos_t> const & start_position,
    std::vector<dir_t> const & start_direction,
    std::vector<float> const & scale,
    brightness_t minimum_brightness,
    uint32_t iterations,
    bool trace_paths)
{
    std::ofstream debug_out("debug_raytrace_instance");
    RaytraceInstanceRef<ior_t, dir_t> inst(
        bound_vec,
        ior,
        transculency,
        start_position,
        start_direction,
        scale,
        minimum_brightness,
        iterations,
        trace_paths,
        true);
    SERIALIZE::write_value(debug_out, inst);
    debug_out.close();
    
    std::tuple<std::vector<pos_t>,std::vector<dir_t>, std::vector<brightness_t>, std::vector<pos_t> > res;
    std::vector<pos_t> path;
    trace_rays<ior_t, iorlog_t, diff_t, dir_t>(
        inst,
        std::get<0>(res),
        std::get<1>(res),
        std::get<2>(res),
        std::get<3>(res),
        path,
        Options());
    return res;
}

PYBIND11_MODULE(cuda_raytrace, m) {
    m.doc() = "A function which casts rays with cuda";
    m.def("cuda_raytrace", &py_trace_rays, "A function which casts rays with cuda");
}
