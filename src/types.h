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

struct Options
{
    size_t _loglevel;
    size_t _target;
    bool _write_instance;
    Options(size_t loglevel_, size_t target_, bool write_instance_) : _loglevel(loglevel_), _target(target_), _write_instance(write_instance_){}
    Options() : _loglevel(0), _target(3), _write_instance(false){}
};
#endif
