/*
Copyright (c) 2018 Paul Stahr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <jni.h>
//#include "/usr/lib/jvm/java-8-openjdk-amd64/include/jni.h"
#include <inttypes.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include "util.h"
#include "image_util.h"
#include "types.h"
#include "fstream"
#include "io_util.h"
#include "serialize.h"
#include "java_binding.h"

/* jbyte* data = env->GetByteArrayElements(array, NULL);
    if (data != NULL) {
        memcpy(buffer, data, len);
        env->ReleaseByteArrayElements(array, data, JNI_ABORT);
    }*/

template <typename T, typename JavaFormat, typename Transformation>
void jni_to_vec(JNIEnv &env, jobject obj, std::vector<T> & vec, Transformation tr, JavaFormat)
{
    int size = env.GetDirectBufferCapacity(obj);
    JavaFormat *buff = static_cast<JavaFormat*>(env.GetDirectBufferAddress(obj));
    std::transform(buff, buff + size, std::back_inserter(vec), tr);
}

template <typename T, typename JavaFormat, typename Transformation>
void vec_to_jni(JNIEnv &env, jobject obj, std::vector<T> const & vec, Transformation tr, JavaFormat)
{
    //int size = env.GetDirectBufferCapacity(obj);
    JavaFormat *buff = static_cast<JavaFormat*>(env.GetDirectBufferAddress(obj));
    std::transform(vec.begin(), vec.end(), buff, tr);
}



template <typename T, typename V>
bool delete_if_instance_of(T *ptr)
{
    V* tmp = dynamic_cast<V*>(ptr);
    if (tmp){delete tmp;return true;}
    return false;
}

template<typename T>
struct javatype_struct{typedef void _type;};
template <> struct javatype_struct<uint64_t>{typedef jlong _type;};
template <> struct javatype_struct<uint32_t>{typedef jint _type;};
template <> struct javatype_struct<uint16_t>{typedef jshort _type;};
template <> struct javatype_struct<uint8_t> {typedef jbyte _type;};
template <> struct javatype_struct<int64_t> {typedef jlong _type;};
template <> struct javatype_struct<int32_t> {typedef jint _type;};
template <> struct javatype_struct<int16_t> {typedef jshort _type;};
template <> struct javatype_struct<int8_t>  {typedef jbyte _type;};
template <> struct javatype_struct<float>   {typedef jfloat _type;};
template <> struct javatype_struct<double>  {typedef jdouble _type;};
template <typename T>
using javatype = typename javatype_struct<T>::_type;

template <typename DirType>
void java_trace_ray_impl(
    JNIEnv *env,
    jlong pointer,
    jobject start_position,
    jobject start_direction,
    jobject end_iteration,
    jobject scale,
    jfloat minimum_brightness,
    jint iterations,
    jboolean trace_paths,
    jobject trace,
    jlong opt_pt
)
{
    try
    {
        RaytraceSceneBase *sceneb = reinterpret_cast<RaytraceSceneBase*>(pointer);
        
        RayTraceRayInstance<DirType> ray_instance;
        std::vector<pos_t> end_position_vec;
        std::vector<DirType> end_direction_vec;
        std::vector<brightness_t> remaining_light_vec;
        std::vector<pos_t> trace_vec;
        std::vector<pos_t> end_iteration_vec;
        
        jni_to_vec(*env, start_position, ray_instance._start_position, UTIL::identity_function, (jint)0);
        jni_to_vec(*env, start_direction, ray_instance._start_direction, UTIL::identity_function, javatype<DirType>(0));
        jni_to_vec(*env, scale, ray_instance._invscale, UTIL::identity_function, (jfloat)0);
        ray_instance._minimum_brightness = minimum_brightness;
        ray_instance._iterations = iterations;
        ray_instance._trace_path = trace_paths;
        ray_instance._normalize_length = true;
        
        Options &opt = *reinterpret_cast<Options *>(opt_pt);
        if (opt._write_instance)
        {
            std::ofstream debug_out("debug_ray_instance");
            SERIALIZE::write_value(debug_out, ray_instance);
            debug_out.close();
        }
        
        {
            RaytraceScene<ior_t, iorlog_t, diff_t> *scene = dynamic_cast<RaytraceScene<ior_t, iorlog_t, diff_t>*>(sceneb);
            if (scene != nullptr)
            {
                scene->trace_rays(
                    RayTraceRayInstanceRef<DirType>(ray_instance),
                    end_position_vec,
                    end_direction_vec,
                    end_iteration_vec,
                    remaining_light_vec,
                    trace_vec,
                    opt);
                goto success;
            }
        }
        {
            RaytraceScene<float, float, float> *scene = dynamic_cast<RaytraceScene<float, float, float>*>(sceneb);
            if (scene != nullptr)
            {
                scene->trace_rays(
                    RayTraceRayInstanceRef<DirType>(ray_instance),
                    end_position_vec,
                    end_direction_vec,
                    end_iteration_vec,
                    remaining_light_vec,
                    trace_vec,
                    opt);
                goto success;
            }
        }
        std::cout << "Warning, can't parse input Scene" << std::endl;
        success:
        vec_to_jni(*env, start_position, end_position_vec, UTIL::identity_function, (jint)0);
        vec_to_jni(*env, start_direction, end_direction_vec, UTIL::identity_function, javatype<DirType>(0));
        vec_to_jni(*env, end_iteration, end_iteration_vec, UTIL::identity_function, (jint)0);
        if (trace_paths)
        {
            vec_to_jni(*env, trace, trace_vec, UTIL::identity_function, (jint)0);
        }
    }catch(std::exception const & e)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
    }
}

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    new_instance
 * Signature: (Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_data_raytrace_OpticalVolumeObject_new_1instance__Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2J
  (JNIEnv *env, jclass, jobject bounds, jobject ior, jobject translucency, jlong opt_pt){
    try
    {
        RayTraceSceneInstance<ior_t> inst;
        jni_to_vec(*env, bounds, inst._bound_vec, UTIL::identity_function, (jint)0);
        jni_to_vec(*env, ior, inst._ior, UTIL::identity_function, (jint)0);
        jni_to_vec(*env, translucency, inst._translucency, UTIL::identity_function, (jint)0);
               
        Options const & opt = *reinterpret_cast<Options*>(opt_pt);
        if (opt._write_instance)
        {
            std::ofstream debug_out("debug_scene_instance");
            SERIALIZE::write_value(debug_out, inst);
            debug_out.close();
        }
        std::cout << "bound_size:" << inst._bound_vec.size() << std::endl;
        RaytraceScene<ior_t, iorlog_t, diff_t> *scene = new RaytraceScene<ior_t, iorlog_t, diff_t>(inst, opt);
        return reinterpret_cast<jlong>(scene);
    }catch(std::exception const & e)
    {
        std::cout << e.what() << std::endl;
        return env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
    }
}

JNIEXPORT jlong JNICALL Java_data_raytrace_OpticalVolumeObject_new_1options
  (JNIEnv *, jclass)
  {
      Options *opt = new Options();
      return reinterpret_cast<jlong>(opt);
  }

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    new_instance
 * Signature: (Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;J)J
 */
JNIEXPORT jlong JNICALL Java_data_raytrace_OpticalVolumeObject_new_1instance__Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2J
  (JNIEnv *env, jclass, jobject bounds, jobject ior, jobject translucency, jlong opt_pt){
    try
    {
        RayTraceSceneInstance<float> inst;
        jni_to_vec(*env, bounds, inst._bound_vec, UTIL::identity_function, (jint)0);
        jni_to_vec(*env, ior, inst._ior, UTIL::identity_function, (jfloat)0);
        jni_to_vec(*env, translucency, inst._translucency, UTIL::identity_function, (jint)0);
        
        Options const & opt = *reinterpret_cast<Options*>(opt_pt);
        if (opt._write_instance)
        {
            std::ofstream debug_out("debug_scene_instance");
            SERIALIZE::write_value(debug_out, inst);
            debug_out.close();
        }
        std::cout << "bound_size:" << inst._bound_vec.size() << std::endl;
        RaytraceScene<float, float, float> *scene = new RaytraceScene<float, float, float>(inst, opt);
        return reinterpret_cast<jlong>(scene);
    }catch(std::exception const & e)
    {
        std::cout << e.what() << std::endl;
        return env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
    }
}

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    trace_rays
 * Signature: (JLjava/nio/IntBuffer;Ljava/nio/ShortBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;FIZLjava/nio/IntBuffer;J)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_trace_1rays__JLjava_nio_IntBuffer_2Ljava_nio_ShortBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2FIZLjava_nio_IntBuffer_2J
(
    JNIEnv *env,
    jclass ,
    jlong pointer,
    jobject start_position,
    jobject start_direction,
    jobject end_iteration,
    jobject scale,
    jfloat minimum_brightness,
    jint iterations,
    jboolean trace_paths,
    jobject trace,
    jlong opt_pt)
{
    java_trace_ray_impl<dir_t>(env, pointer, start_position, start_direction, end_iteration, scale, minimum_brightness, iterations, trace_paths, trace, opt_pt);
}

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    trace_rays
 * Signature: (JLjava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;FIZLjava/nio/IntBuffer;J)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_trace_1rays__JLjava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2FIZLjava_nio_IntBuffer_2J
(
    JNIEnv *env,
    jclass ,
    jlong pointer,
    jobject start_position,
    jobject start_direction,
    jobject end_iteration,
    jobject scale,
    jfloat minimum_brightness,
    jint iterations,
    jboolean trace_paths,
    jobject trace,
    jlong opt_pt)
{
    java_trace_ray_impl<float>(env, pointer, start_position, start_direction, end_iteration, scale, minimum_brightness, iterations, trace_paths, trace, opt_pt);
}

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    delete_instance
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_delete_1instance (JNIEnv *env, jclass, jlong pointer)
{
    RaytraceSceneBase *sceneb = reinterpret_cast<RaytraceSceneBase*>(pointer);
    try{
        delete_if_instance_of<RaytraceSceneBase, RaytraceScene<ior_t, iorlog_t, diff_t> >(sceneb);
        delete_if_instance_of<RaytraceSceneBase, RaytraceScene<float, float, float> >(sceneb);
    }catch(std::exception const & e)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
    }
}

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    delete_options
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_delete_1options
  (JNIEnv *, jclass, jlong opt_pt)
  {
      delete reinterpret_cast<Options*>(opt_pt);
  }
  
/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    get_option_valuei
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_data_raytrace_OpticalVolumeObject_get_1option_1valuei
  (JNIEnv *, jclass, jlong pt, jlong key)
  {
      Options &opt = *reinterpret_cast<Options*>(pt);
      switch(key)
      {
          case 0:return opt._loglevel;
          case 2:return opt._minimum_gpu;
          default: throw std::runtime_error("Illegal Argument " + std::to_string(key));
      }
  }

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    set_option_valuei
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_set_1option_1valuei
  (JNIEnv *, jclass, jlong pt, jlong key, jint value)
  {
      Options &opt = *reinterpret_cast<Options*>(pt);
      switch(key)
      {
          case 0: opt._loglevel = value;break;
          case 2: opt._minimum_gpu = value;break;
          default: throw std::runtime_error("Illegal Argument " + std::to_string(key));
      }
  }

  /*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    get_option_valueb
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_data_raytrace_OpticalVolumeObject_get_1option_1valueb
  (JNIEnv *, jclass, jlong pt, jlong key)
  {
      Options &opt = *reinterpret_cast<Options*>(pt);
      switch(key)
      {
          case 1:return opt._write_instance;
          default: throw std::runtime_error("Illegal Argument " + std::to_string(key));
      }
  }

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    set_option_valueb
 * Signature: (JJZ)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_set_1option_1valueb
  (JNIEnv *, jclass, jlong pt, jlong key, jboolean value)
  {
      Options &opt = *reinterpret_cast<Options*>(pt);
      switch(key)
      {
          case 1: opt._write_instance = value;break;
          default: throw std::runtime_error("Illegal Argument " + std::to_string(key));
      }
  }

#ifdef __cplusplus
}
#endif


