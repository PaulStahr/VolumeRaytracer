import cupy as cp
import logging
import numpy as np
import itertools
import inspect
import logging

logger = logging.getLogger(__name__)

#inline __device__ __host__ __device__  void trace_ray_function(
#    cuda_tuple<DiffType,dim + 1>  *diff_interleaved,
#    T translucency,/*translucency_t*/
#    cuda_tuple<uint16_t,dim> bounds,
#    cuda_tuple<float,dim> invscale,
#    raydata_t<dim, DirType> *raydata,
#    P path, /*cuda_tuple<pos_t,dim>*/
#    B minimum_brightness)
#{
#    cuda_tuple<uint32_t,dim> pos = raydata->_position;
#    cuda_tuple<float,dim> direction = make_struct<float,dim>()(raydata->_direction);
#    uint32_t iterations = raydata->_iterations;
#    if(std::is_same<dir_t, DirType>::value) {direction *= 0x100;}
#    else                                    {direction *= 0x10000;}
#    B brightness = 0xFFFFFFFF;
#    path[--iterations] = pos;
#
#    while (iterations -- > 0 && make_struct<uint16_t, dim>()(pos >> 16) < bounds - 1)
#    {
#        if (translucency)
#        {
#            brightness -= min(static_cast<brightness_t>(brightness), static_cast<translucency_t>(0xFFFFFFFF-translucency[get_index(bounds, pos)]));
#            if (brightness < minimum_brightness){break;}
#        }
#        cuda_tuple<float,dim + 1> interpolation = interpolatef(diff_interleaved, bounds, pos, type_uint8_t<dim + 1>());
#        if (get(interpolation, dim) > 0){break;}
    #        interpolation *= invscale;//TODO can be precalculated
#        direction += interpolation;
#        float ilen = 0x42000000p0f / dot(direction, direction);
#        pos += __float2int_rn2(direction * invscale * ilen);
#        path[iterations] = pos;
#    }
#    ++iterations;
#    raydata->_iterations = iterations;
#    if (!std::is_same<P,DummyArray>::value)
#    {
#        while (iterations --> 0)
#        {
#            path[iterations] = pos;
#        }
#    }
#    if(std::is_same<dir_t, DirType>::value)
#    {
#        direction /= 0x100;
#        raydata->_direction = make_struct<DirType,dim>()(__float2int_rn2(direction));
#    }
#    else
#    {
#        direction /= 0x10000;
#        raydata->_direction = make_struct<DirType,dim>()(direction);
#    }
#    raydata->_position = pos;
#    if (translucency)
#    {
#        raydata->_remaining_brightness = brightness;
#    }
#
source_cupy_texture_lookup = r'''
template <class T, T V>
struct template_constant
{
    template_constant(){};    
    constexpr operator T() const { return V; }
    template <typename W>
    constexpr T operator ()(W const &) const{return V;}
    template <typename W, typename X>
    constexpr T operator ()(W const &, X const &) const{return V;}
};

struct DummyObject
{
    inline __host__ __device__  DummyObject(){}
    
    template <typename T> inline __host__ __device__  DummyObject(T /*t*/){}
    template <typename T> inline __host__ __device__  T& operator=(T&& other) noexcept{return other;}
    template <typename T> inline __host__ __device__  operator T() const{return T();}
    template <typename T> inline __host__ __device__  DummyObject operator-=(T /*value*/){return *this;}
    template <typename T> inline __host__ __device__  DummyObject operator<(T /*value*/){return false;}  
};

template <typename T>
inline __host__ __device__  DummyObject operator -(T /*a*/, DummyObject /*b*/){return DummyObject();}

struct DummyArray{
    inline __host__ __device__  DummyObject const   operator [](size_t /*index*/) const{return DummyObject();}
    inline __host__ __device__  DummyObject         operator [](size_t /*index*/) {return DummyObject();}
    inline __host__ __device__  void                operator +=(size_t /*index*/) {}
    inline __host__ __device__  DummyArray          operator + (size_t /*index*/) {return DummyArray();}
    inline __host__ __device__  operator bool() const{return false;}
};

inline __device__ bool operator < (float2 a, float2 b)    {return a.x < b.x && a.y < b.y;}
inline __device__ bool operator < (float3 a, float3 b)    {return a.x < b.x && a.y < b.y && a.z < b.z;}
inline __device__ bool operator < (float2 a, float b)     {return a.x < b && a.y < b;}
inline __device__ bool operator < (float3 a, float b)     {return a.x < b && a.y < b && a.z < b;}
inline __device__ bool operator > (float2 a, float b)     {return a.x > b && a.y > b;}
inline __device__ bool operator > (float3 a, float b)     {return a.x > b && a.y > b && a.z > b;}


inline __device__ float2 operator + (float2 a, float2 b){return make_float2(a.x + b.x, a.y + b.y);}
inline __device__ float3 operator + (float3 a, float3 b){return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);}
inline __device__ float4 operator + (float4 a, float4 b){return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);}

inline __device__ float2 & operator += (float2 &a, float2 b){a.x += b.x; a.y += b.y; return a;}
inline __device__ float3 & operator += (float3 &a, float3 b){a.x += b.x; a.y += b.y; a.z += b.z; return a;}
inline __device__ float4 & operator += (float4 &a, float4 b){a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a;}

inline __device__ float2 operator * (float2 a, float b){return make_float2(a.x * b, a.y * b);}
inline __device__ float3 operator * (float3 a, float b){return make_float3(a.x * b, a.y * b, a.z * b);}
inline __device__ float4 operator * (float4 a, float b){return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);}

inline __device__ float2 operator * (float2 a, float2 b){return make_float2(a.x * b.x, a.y * b.y);}
inline __device__ float3 operator * (float3 a, float3 b){return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);}

inline __device__ float lastvar(float a){return a;}
inline __device__ float lastvar(float2 a){return a.y;}
inline __device__ float lastvar(float3 a){return a.z;}
inline __device__ float lastvar(float4 a){return a.w;}
inline __device__ float make_float1(float2 a){return a.x;}
inline __device__ float2 make_float2(float2 a){return make_float2(a.x, a.y);}
inline __device__ float2 make_float2(float3 a){return make_float2(a.x, a.y);}
inline __device__ float2 make_float2(float4 a){return make_float2(a.x, a.y);}
inline __device__ float3 make_float3(float3 a){return make_float3(a.x, a.y, a.z);}
inline __device__ float3 make_float3(float4 a){return make_float3(a.x, a.y, a.z);}
inline __device__ float4 make_float4(float4 a){return make_float4(a.x, a.y, a.z, a.w);}

inline __device__ float dot(float2 a){return a.x * a.x + a.y * a.y;}
inline __device__ float dot(float3 a){return a.x * a.x + a.y * a.y + a.z * a.z;}
inline __device__ float dot(float4 a){return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;}

inline __device__ float tex2D1Tuple(cudaTextureObject_t texObj, float2 pos){return tex2D<float>(texObj, pos.x, pos.y);}
inline __device__ float tex3D1Tuple(cudaTextureObject_t texObj, float3 pos){return tex3D<float>(texObj, pos.x, pos.y, pos.z);}
inline __device__ float4 tex2D4Tuple(cudaTextureObject_t texObj, float2 pos){return tex2D<float4>(texObj, pos.x, pos.y);}
inline __device__ float4 tex3D4Tuple(cudaTextureObject_t texObj, float3 pos){return tex3D<float4>(texObj, pos.x, pos.y, pos.z);}


extern "C"{
__global__ void raytracing_kernel(
    float{dim}* position,
    float{dim}* direction,
    unsigned int* iterations,
    float{dim}* bd,
    cudaTextureObject_t gradients,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float{dim} pos = position[idx];
        float{dim} dir = direction[idx];
        unsigned int iter = iterations[idx];
        float{dim} bound = make_float{dim}(*bd);
        while (iter-- > 0 && pos > 0 && pos < bound)
        {
            float{dim+1} interpolation = make_float{dim+1}(tex{dim}D4Tuple(gradients, pos));
            if (lastvar(interpolation) < 0.0f) {
                break;
            }
            dir += make_float{dim}(interpolation);
            pos += dir * (1 / dot(dir));
        }
        iterations[idx] = iter + 1;
        position[idx] = pos;
        direction[idx] = dir;
    }
}

__global__ void evaluate_ior_kernel(
    float{dim}* position,
    float* result,
    cudaTextureObject_t ior_texture,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n)
    {
        result[idx] = tex{dim}D1Tuple(ior_texture, position[idx]);
    }
}
}
'''

ior_evaluation_kernel = [None] * 4
raytracing_kernel = [None] * 4


standard_stamp = [None] * 4
standard_stamp[2] = cp.asarray([47,162,47], dtype=np.float32)
standard_stamp[3] = cp.asarray([[14,47,14],[47,162,47],[14,47,14]], dtype=np.float32)
for i in range(2, 4):
    standard_stamp[i] /= cp.sum(standard_stamp[i])


def create_cuda_texture(img, dtype=None):
    if dtype is None:
        dtype = cp.float32
    img = img.astype(dtype)
    if img.shape[-1] > 4:
        img = img[...,np.newaxis]
    num_channels = img.shape[-1]
    ex = None
    if img.ndim == 3:
        grad_stacked = img.reshape([img.shape[0], img.shape[1] * num_channels])
    elif img.ndim == 4:
        grad_stacked = img.reshape([img.shape[0], img.shape[1], img.shape[2] * num_channels])
    else:
        raise ValueError("Image must be 3D or 4D (HWC or NHWC format)")
    for _ in range(3):
        try:
            nbits = 8 * img.itemsize
            ch = cp.cuda.texture.ChannelFormatDescriptor(nbits,
                                                         nbits if num_channels > 1 else 0,
                                                         nbits if num_channels > 2 else 0,
                                                         nbits if num_channels > 3 else 0,
                                                         cp.cuda.runtime.cudaChannelFormatKindFloat)
            if img.ndim == 3:
                arr2 = cp.cuda.texture.CUDAarray(ch, img.shape[1], img.shape[0], flags=cp.cuda.runtime.cudaArrayDefault)
            elif img.ndim == 4:
                arr2 = cp.cuda.texture.CUDAarray(ch, img.shape[2], img.shape[1], img.shape[0],
                                                 flags=cp.cuda.runtime.cudaArrayDefault)
            else:
                raise ValueError("Image must be 3D or 4D (HWC or NHWC format)")
            res = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arr2)
            tex = cp.cuda.texture.TextureDescriptor(
                (cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp),
                cp.cuda.runtime.cudaFilterModePoint,
                cp.cuda.runtime.cudaReadModeElementType)
            arr2.copy_from(grad_stacked)
            return cp.cuda.texture.TextureObject(res, tex)
        except Exception as e:
            ex = e
            pass
    raise ex

class OpticalVolume:
    def __init__(self, ior:cp.ndarray|np.ndarray, transculency:cp.ndarray|np.ndarray, scale:cp.ndarray|np.ndarray, xp=cp):
        self.ior = ior
        self.translucency = transculency
        self.gradient = None
        self.texture = None
        self.ior_texture = None
        self.scale = scale
        self.ndim = ior.ndim
        self.shape = ior.shape
        self.xp = xp
        self.update()

    def get_ior(self, position):
        assert position.dtype == cp.float32, "Position must be of type float32"
        current_kernel = ior_evaluation_kernel[self.ndim]
        if current_kernel is None:
            code = source_cupy_texture_lookup.replace('{dim}', str(self.ndim)).replace('{dim+1}', str(self.ndim + 1))
            current_kernel = cp.RawKernel(code, 'evaluate_ior_kernel')
            ior_evaluation_kernel[self.ndim] = current_kernel
        block_size = 256
        num_blocks = (position.shape[0] + block_size - 1) // block_size
        result = cp.empty(position.shape[0], dtype=cp.float32)
        current_kernel((num_blocks,), (block_size,), (position, result, self.ior_texture, position.shape[0]))
        return result

    def update(self):
        # calculate gradients in all directions by building 2d or 3d convolution
        grad = self.xp.gradient(self.xp.log(OpticalVolume.convert(self.ior, self.xp)))
        gradient = []
        for axis in range(self.ndim):
            grad[axis] *= self.scale[axis]
            current = self.xp.zeros(shape=np.asarray(self.ior.shape) + 2, dtype=self.xp.float32)
            padded = self.xp.pad(grad[axis], [(1, 1)] * self.ndim, mode='edge')
            mask = np.ones(shape=self.ndim, dtype=bool)
            mask[axis] = False
            for shp in itertools.product(*[range(s) for s in standard_stamp[self.ndim].shape]):
                shift = np.zeros(shape=self.ndim, dtype=int)
                shift[mask] = np.asarray(shp) - 1
                current += self.xp.roll(padded, shift=tuple(shift)) * standard_stamp[self.ndim][*shp]
            gradient.append(current[tuple([slice(1,-1)] * self.ndim)])
        gradient = self.xp.stack(gradient, axis=-1)
        translucency = OpticalVolume.convert(self.translucency, self.xp)
        gradient = [gradient, translucency[..., self.xp.newaxis]]
        if self.ndim == 2:
            gradient.append(translucency[..., self.xp.newaxis])
        self.gradient = self.xp.concatenate(gradient, axis=-1)
        self.texture = create_cuda_texture(np.swapaxes(self.gradient, 0, -2), dtype=np.float32)
        self.ior_texture = create_cuda_texture(np.swapaxes(self.ior, 0, -1), dtype=np.float32)

    def trace_rays(self, positions:cp.ndarray|np.ndarray, directions:cp.ndarray|np.ndarray, iterations:cp.ndarray, bounds:cp.ndarray):
        if isinstance(positions, np.ndarray):
            positions = cp.asarray(positions, dtype=cp.float32)
        if isinstance(directions, np.ndarray):
            directions = cp.asarray(directions, dtype=cp.float32)
        if isinstance(iterations, np.ndarray):
            iterations = cp.asarray(iterations, dtype=cp.uint32)
        assert positions.dtype == cp.float32, "Positions must be of type float32"
        assert directions.dtype == cp.float32, "Directions must be of type float32"
        assert iterations.dtype == cp.uint32, "Iterations must be of type uint32"
        if not isinstance(bounds, cp.ndarray):
            bounds = cp.asarray(bounds, dtype=cp.float32)
        if bounds.dtype != cp.float32:
            logger.log(logging.WARNING, f"Converting bounds from {bounds.dtype} to float32")
            bounds = cp.asarray(bounds, dtype=cp.float32)
        current_kernel = raytracing_kernel[self.ndim]
        if current_kernel is None:
            code = source_cupy_texture_lookup.replace('{dim}', str(self.ndim)).replace('{dim+1}', str(self.ndim + 1))
            current_kernel = cp.RawKernel(code, 'raytracing_kernel')
            raytracing_kernel[self.ndim] = current_kernel
        #call current_kernel
        block_size = 256
        num_blocks = (positions.shape[0] + block_size - 1) // block_size
        current_kernel((num_blocks,), (block_size,), (positions, directions, iterations, bounds, self.texture, positions.shape[0]))
        return positions, directions, iterations


    @staticmethod
    def convert(img, module):
        if isinstance(img, str):
            return img
        if module == None:
            return img
        t = type(img)
        if inspect.getmodule(t) == module:
            return img
        if logging.DEBUG >= logging.root.level:
            finfo = inspect.getouterframes(inspect.currentframe())[1]
            logger.log(logging.DEBUG,
                       F'convert {t.__module__} to {module.__name__} by {finfo.filename} line {finfo.lineno}')
        if t.__module__ == 'cupy':
            return module.array(img.get(), copy=False)
        return module.array(img, copy=False)

