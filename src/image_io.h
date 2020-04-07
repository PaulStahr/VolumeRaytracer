#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstdint>
#include <vector>
namespace IMG_IO{
struct image_t{
    size_t _width;
    size_t _height;
    size_t _channels;
    std::vector<uint8_t> _data;
    image_t(size_t width_, size_t height_, size_t channels_) : _width(width_), _height(height_), _channels(channels_), _data(width_ * height_ * channels_) {}
    image_t(){}
};

image_t load_jpeg(char* FileName, bool Fast = true);

void write_jpeg(const char* filename, image_t const & img);

image_t read_png(const char*filename);
}
#endif
