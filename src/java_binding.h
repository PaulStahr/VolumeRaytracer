#ifndef JAVA_BINDING_H
#define JAVA_BINDING_H
/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class data_raytrace_OpticalVolumeObject */

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    new_instance
 * Signature: (Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;J)J
 */
JNIEXPORT jlong JNICALL Java_data_raytrace_OpticalVolumeObject_new_1instance__Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2J
  (JNIEnv *, jclass, jobject, jobject, jobject, jlong);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    new_options
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_data_raytrace_OpticalVolumeObject_new_1options
  (JNIEnv *, jclass);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    new_instance
 * Signature: (Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;J)J
 */
JNIEXPORT jlong JNICALL Java_data_raytrace_OpticalVolumeObject_new_1instance__Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2J
  (JNIEnv *, jclass, jobject, jobject, jobject, jlong);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    delete_instance
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_delete_1instance
  (JNIEnv *, jclass, jlong);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    delete_options
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_delete_1options
  (JNIEnv *, jclass, jlong);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    get_option_valuei
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_data_raytrace_OpticalVolumeObject_get_1option_1valuei
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    set_option_valuei
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_set_1option_1valuei
  (JNIEnv *, jclass, jlong, jlong, jint);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    get_option_valueb
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_data_raytrace_OpticalVolumeObject_get_1option_1valueb
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    set_option_valueb
 * Signature: (JJZ)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_set_1option_1valueb
  (JNIEnv *, jclass, jlong, jlong, jboolean);
  
/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    trace_rays
 * Signature: (JLjava/nio/IntBuffer;Ljava/nio/ShortBuffer;Ljava/nio/FloatBuffer;FIZJ)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_trace_1rays__JLjava_nio_IntBuffer_2Ljava_nio_ShortBuffer_2Ljava_nio_FloatBuffer_2FIZJ
  (JNIEnv *, jclass, jlong, jobject, jobject, jobject, jfloat, jint, jboolean, jlong);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    trace_rays
 * Signature: (JLjava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;FIZJ)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_trace_1rays__JLjava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2FIZJ
  (JNIEnv *, jclass, jlong, jobject, jobject, jobject, jfloat, jint, jboolean, jlong);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    trace_rays
 * Signature: (Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;Ljava/nio/ShortBuffer;Ljava/nio/FloatBuffer;FIZ)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_trace_1rays__Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_ShortBuffer_2Ljava_nio_FloatBuffer_2FIZ
  (JNIEnv *, jclass, jobject, jobject, jobject, jobject, jobject, jobject, jfloat, jint, jboolean);

/*
 * Class:     data_raytrace_OpticalVolumeObject
 * Method:    trace_rays
 * Signature: (Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;FIZ)V
 */
JNIEXPORT void JNICALL Java_data_raytrace_OpticalVolumeObject_trace_1rays__Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2FIZ
  (JNIEnv *, jclass, jobject, jobject, jobject, jobject, jobject, jobject, jfloat, jint, jboolean);

#ifdef __cplusplus
}
#endif
#endif
