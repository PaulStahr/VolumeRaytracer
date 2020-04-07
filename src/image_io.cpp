#include <cstdlib>
#include <cstdio>
#include <jpeglib.h>
#include <png.h>
#include <jerror.h>
#include <stdexcept>
#include "image_io.h"
//if the jpeglib stuff isnt after I think stdlib then it wont work just put it at the end

namespace IMG_IO{  
image_t load_jpeg(char* FileName, bool Fast)
{
  FILE* file = fopen(FileName, "rb");  //open the file
  struct jpeg_decompress_struct info;  //the jpeg decompress info
  struct jpeg_error_mgr err;           //the error handler
 
 image_t erg;
  info.err = jpeg_std_error(&err);     //tell the jpeg decompression handler to send the errors to err
  jpeg_create_decompress(&info);       //sets info to all the default stuff
 
  //if the jpeg file didnt load exit
  if(!file)
  {
    throw std::runtime_error("Error reading JPEG file %s!!!" );
  }
 
  jpeg_stdio_src(&info, file);    //tell the jpeg lib the file we'er reading
 
  jpeg_read_header(&info, TRUE);   //tell it to start reading it
 
  //if it wants to be read fast or not
  if(Fast)
  {
    info.do_fancy_upsampling = FALSE;
  }
 
  jpeg_start_decompress(&info);    //decompress the file
 
  //set the x and y
  erg._width = info.output_width;
  erg._height = info.output_height;
  erg._channels = info.num_components;
 
  erg._data.resize(erg._channels * erg._width * erg._height);
 
 
  uint8_t* p2 = erg._data.data();
 
  while(info.output_scanline < info.output_height)
  {
    int numlines = jpeg_read_scanlines(&info, &p2, 1);
    p2 += numlines * 3 * info.output_width;
  }
 
  jpeg_finish_decompress(&info);   //finish decompressing this file
 
  fclose(file);                    //close the file
 
  return erg;
}

void write_jpeg(const char* filename, image_t const & img)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  /* More stuff */
  FILE * outfile;               /* target file */
  cinfo.err = jpeg_std_error(&jerr);
  /* Now we can initialize the JPEG compression object. */
  jpeg_create_compress(&cinfo);

  if ((outfile = fopen(filename, "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, outfile);
  cinfo.image_width = img._width;      /* image width and height, in pixels */
  cinfo.image_height = img._height;
  cinfo.input_components = img._channels;           /* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB;       /* colorspace of input image */
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, TRUE /* limit to baseline-JPEG values */);
    jpeg_start_compress(&cinfo, TRUE);
  int row_stride = img._width * img._channels; /* JSAMPLEs per row in image_buffer */

  while (cinfo.next_scanline < cinfo.image_height) {
    /* jpeg_write_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could pass
     * more than one scanline at a time if that's more convenient.
     */
    JSAMPROW row_pointer = const_cast<JSAMPROW>( img._data.data() + cinfo.next_scanline * row_stride);
    (void) jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  /* Step 6: Finish compression */

  jpeg_finish_compress(&cinfo);
  /* After finish_compress, we can close the output file. */
  fclose(outfile);

  /* Step 7: release JPEG compression object */

  /* This is an important step since it will release a good deal of memory. */
  jpeg_destroy_compress(&cinfo);

  /* And we're done! */
}

image_t read_png(const char *filename) {
    png_byte color_type;
    png_byte bit_depth;
  FILE *fp = fopen(filename, "rb");
    image_t erg;
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png) abort();

  png_infop info = png_create_info_struct(png);
  if(!info) abort();

  if(setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  png_read_info(png, info);

  erg._width      = png_get_image_width(png, info);
  erg._height     = png_get_image_height(png, info);
  color_type = png_get_color_type(png, info);
  bit_depth  = png_get_bit_depth(png, info);

  // Read any color_type into 8bit depth, RGBA format.
  // See http://www.libpng.org/pub/png/libpng-manual.txt

  if(bit_depth == 16)
    png_set_strip_16(png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  // These color_type don't have an alpha channel then fill it with 0xff.
  if(color_type == PNG_COLOR_TYPE_RGB ||
     color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

  if(color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);

  png_read_update_info(png, info);
  erg._channels = png_get_channels(png, info);
  erg._data.resize(erg._height * erg._width * erg._channels);
  png_bytep* row_pointers = new png_bytep[erg._height];
  for(size_t y = 0; y < erg._height; y++) {
        row_pointers[y] =erg._data.data() + y * erg._width * erg._channels;
  }

  png_read_image(png, row_pointers);

  delete[] row_pointers;
  png_destroy_read_struct(&png, &info, NULL);
  png=NULL;
  info=NULL;
  fclose(fp);
  return erg;
}
}
