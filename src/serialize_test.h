#include "serialize.h"
#include "image_util.h"
BOOST_AUTO_TEST_SUITE(serialize)

BOOST_AUTO_TEST_CASE( constructors_test )
{
    
}

BOOST_AUTO_TEST_CASE( serialize_test )
{
    std::stringstream s;
    RayTraceSceneInstance<ior_t> inst;
    inst._bound_vec = {1,1,1};
    inst._ior = {1};
    inst._translucency = {1};
    SERIALIZE::write_value(s, inst);
    RayTraceSceneInstance<ior_t> inst2;
    SERIALIZE::read_value(s, inst2);
    BOOST_CHECK( inst == inst2 );
}

BOOST_AUTO_TEST_SUITE_END()
