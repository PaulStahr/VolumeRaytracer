#include "image_util.h"
BOOST_AUTO_TEST_SUITE(image_util)

BOOST_AUTO_TEST_CASE( interpolation_test )
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
    
    /*The tests creates a gradient inside a 3d-volume, from minor to major axis*/
    for (size_t div = 1, axis = 0; div < 125; div *= 5, ++axis)
    {
        /*Create gradient on each axis, with increments of 100*/
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = 100 * ((i / div) % 5);
        }
        interpolator<iorlog_t> interp(values, bounds);
        /*check if interpolation into given direction matches
         Please note, that minor index has the highest index*/
        for (size_t i = 0; i < pos.size(); i += 3)
        {
            BOOST_TEST(interp(pos.begin() + i) == pos[i + 2 - axis] * 100 / 0x10000);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
