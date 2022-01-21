#ifndef TYPES_H
#define TYPES_H

#include <cinttypes>
typedef uint32_t pos_t;
typedef int16_t dir_t;
typedef int16_t diff_t;
typedef int32_t iorlog_t;
typedef uint32_t ior_t;
typedef uint32_t brightness_t;
typedef uint32_t translucency_t;

template <typename IOR_TYPE>
struct ior_typeinfo_struct{};

template<>
struct ior_typeinfo_struct<ior_t>
{
    static const ior_t unit_value = 0x10000;

    static constexpr float tolerance = static_cast<float>(1) / 0x10000;

    static double constexpr to_double (ior_t ior){return static_cast<double>(ior) / static_cast<double>(unit_value);}
    static double constexpr to_float  (ior_t ior){return static_cast<float>(ior) / static_cast<float>(unit_value);}
};

template<>
struct ior_typeinfo_struct<float>
{
    static constexpr float unit_value = static_cast<float>(1);

    static constexpr float tolerance = 0;

    static double constexpr to_double (float ior){return static_cast<double>(ior) / static_cast<double>(unit_value);}
    static double constexpr to_float  (float ior){return static_cast<float>(ior) / static_cast<float>(unit_value);}
};

template <typename DIR_TYPE>
struct dir_typeinfo_struct{};

template<>
struct dir_typeinfo_struct<dir_t>
{
    static const dir_t unit_value = 0x100;

    static constexpr float tolerance = static_cast<float>(1) / 0x100;

    static double constexpr to_double (dir_t dir){return static_cast<double>(dir) / static_cast<double>(unit_value);}
    static double constexpr to_float  (dir_t dir){return static_cast<float>(dir) / static_cast<float>(unit_value);}
};

template<>
struct dir_typeinfo_struct<float>
{
    static constexpr float unit_value = static_cast<float>(1);

    static constexpr float tolerance = 0;

    static double constexpr to_double(float dir){return static_cast<double>(dir);}
    static double constexpr to_float (float dir){return dir;}
};

template <typename POS_TYPE>
struct pos_typeinfo_struct{};

template<>
struct pos_typeinfo_struct<float>
{
    static constexpr float unit_value = static_cast<float>(1);

    static constexpr float tolerance = 0;

    static double constexpr to_double(pos_t pos){return static_cast<double>(pos);}
    static double constexpr to_float (pos_t pos){return pos;}
};

#ifndef __CUDACC__
template <typename IOR_TYPE> static const ior_typeinfo_struct<IOR_TYPE> ior_typeinfo;
template <typename DIR_TYPE> static const dir_typeinfo_struct<DIR_TYPE> dir_typeinfo;
template <typename POS_TYPE> static const pos_typeinfo_struct<POS_TYPE> pos_typeinfo;
#endif

struct Options
{
    int _loglevel;
    size_t _target;
    bool _write_instance;
    size_t _max_cpu;
    Options(size_t loglevel_, size_t target_, bool write_instance_) : _loglevel(loglevel_), _target(target_), _write_instance(write_instance_), _max_cpu(256){}
    Options() : _loglevel(0), _target(3), _write_instance(false), _max_cpu(256){}
};
#endif
