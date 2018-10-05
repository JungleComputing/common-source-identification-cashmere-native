/*
 * JOCL - Java bindings for OpenCL
 *
 * Copyright (c) 2009-2016 Marco Hutter - http://www.jocl.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef JOCL_CL_JNI_UTILS_HPP
#define JOCL_CL_JNI_UTILS_HPP

#include "JOCLCommon.hpp"
#include <map>

/**
 * A structure containing the information about the arguments that have
 * been passed to establish a callback method. A pointer to this structure
 * will be passed as the *user_data to the respective function. The
 * function will then use the data from the given structure
 * to call the Java callback method.
 */
typedef struct
{
    /**
     * A global reference to the user_data that was given
     */
    jobject globalUser_data;

    /**
     * A global reference to the pfn_notify that was given
     */
    jobject globalPfn_notify;

} CallbackInfo;


/**
 * Typedef for function pointers that may be passed to the
 * clCreateContext* functions
 */
typedef void(CL_CALLBACK *CreateContextFunctionPointer)(const char *errinfo, const void *private_info, size_t cb, void *user_data);

/**
 * Typedef for function pointers that may be passed to the
 * clBuildProgram function
 */
typedef void(CL_CALLBACK *BuildProgramFunctionPointer)(cl_program program, void *user_data);

/**
 * Typedef for function pointers that may be passed to the
 * clEnqueueNativeKernelFunction function
 */
typedef void(CL_CALLBACK *EnqueueNativeKernelFunctionPointer)(void *user_data);

/**
 * Typedef for function pointers that may be passed to the
 * clSetMemObjectDestructorCallback function
 */
typedef void(CL_CALLBACK *MemObjectDestructorCallbackFunctionPointer)(cl_mem memobj, void *user_data);

/**
 * Typedef for function pointers that may be passed to the
 * clSetEventCallback function
 */
typedef void(CL_CALLBACK *EventCallbackFunctionPointer)(cl_event event, cl_int event_command_exec_status, void *user_data);

/**
 * Typedef for function pointers that may be passed to the
 * clSetPrintfCallback function
 */
typedef void(CL_CALLBACK *PrintfCallbackFunctionPointer)(cl_context context, cl_uint printf_data_len, char* printf_data_ptr, void *user_data);

/**
 * Typedef for function pointers that may be passed to the
 * cnEnqueueSVMFree function
 */
typedef void (CL_CALLBACK *SVMFreeCallbackFunctionPointer)(cl_command_queue, cl_uint, void **, void *);


// The JVM, used for attaching the calling thread in
// callback functions
extern JavaVM *globalJvm;

// Field IDs for the cl_image_format class
extern jfieldID cl_image_format_image_channel_order; // cl_channel_order (cl_uint)
extern jfieldID cl_image_format_image_channel_data_type; // cl_channel_type (cl_uint)

// Field IDs for the cl_buffer_region class
extern jfieldID cl_buffer_region_origin; // size_t
extern jfieldID cl_buffer_region_size; // size_t

// Field IDs for the cl_image_desc class
extern jfieldID cl_image_desc_image_type; // cl_mem_object_type (cl_uint)
extern jfieldID cl_image_desc_image_width; // size_t
extern jfieldID cl_image_desc_image_height; // size_t
extern jfieldID cl_image_desc_image_depth; // size_t
extern jfieldID cl_image_desc_image_array_size; // size_t
extern jfieldID cl_image_desc_image_row_pitch; // size_t
extern jfieldID cl_image_desc_image_slice_pitch; // size_t
extern jfieldID cl_image_desc_num_mip_levels; // cl_uint
extern jfieldID cl_image_desc_num_samples; // cl_uint
extern jfieldID cl_image_desc_buffer; // cl_mem

// Class and method ID for cl_platform_id and its constructor
extern jclass cl_platform_id_Class;
extern jmethodID cl_platform_id_Constructor;

// Class and method ID for cl_device_id and its constructor
extern jclass cl_device_id_Class;
extern jmethodID cl_device_id_Constructor;

// Class and method ID for cl_context and its constructor
extern jclass cl_context_Class;
extern jmethodID cl_context_Constructor;

// Class and method ID for cl_command_queue and its constructor
extern jclass cl_command_queue_Class;
extern jmethodID cl_command_queue_Constructor;

// Class and method ID for cl_mem and its constructor
extern jclass cl_mem_Class;
extern jmethodID cl_mem_Constructor;

// Class and method ID for cl_image_format and its constructor
extern jclass cl_image_format_Class;
extern jmethodID cl_image_format_Constructor;

// Class and method ID for cl_sampler and its constructor
extern jclass cl_sampler_Class;
extern jmethodID cl_sampler_Constructor;

// Class and method ID for cl_program and its constructor
extern jclass cl_program_Class;
extern jmethodID cl_program_Constructor;

// Class and method ID for cl_kernel and its constructor
extern jclass cl_kernel_Class;
extern jmethodID cl_kernel_Constructor;

// Class and method ID for cl_event and its constructor
extern jclass cl_event_Class;
extern jmethodID cl_event_Constructor;

/**
 * The CallbackInfo structures of all contexts that have
 * been created so far and not released yet
 */
extern std::map<cl_context, CallbackInfo*> contextCallbackMap;


int initCLJNIUtils(JNIEnv *env);


cl_context_properties* createContextPropertiesArray(JNIEnv *env, jobject properties);
// cl_queue_properties* createQueuePropertiesArray(JNIEnv *env, jobject properties);
// cl_pipe_properties* createPipePropertiesArray(JNIEnv *env, jobject properties);
// cl_sampler_properties* createSamplerPropertiesArray(JNIEnv *env, jobject properties);

void getCl_image_format(JNIEnv *env, jobject image_format, cl_image_format &nativeImage_format);
void setCl_image_format(JNIEnv *env, jobject image_format, cl_image_format &nativeImage_format);

void getCl_image_desc(JNIEnv *env, jobject image_desc, cl_image_desc &nativeImage_desc);

void getCl_buffer_region(JNIEnv *env, jobject buffer_region, cl_buffer_region &nativeBuffer_region);

cl_device_partition_property* getCl_device_partition_property (JNIEnv *env, jobject properties);

cl_event* createEventList(JNIEnv *env, jobjectArray event_list, cl_uint num_events);
cl_device_id* createDeviceList(JNIEnv *env, jobjectArray device_list, cl_uint num_devices);
cl_mem* createMemList(JNIEnv *env, jobjectArray mem_list, cl_uint num_mems);
cl_program* createProgramList(JNIEnv *env, jobjectArray program_list, cl_uint num_programs);
void** createSvmPointers(JNIEnv *env, jobjectArray svm_pointers, cl_uint num_svm_pointers);

CallbackInfo* initCallbackInfo(JNIEnv *env, jobject pfn_notify, jobject user_data);
void deleteCallbackInfo(JNIEnv *env, CallbackInfo* &callbackInfo);
void destroyCallbackInfo(JNIEnv *env, cl_context context);
void finishCallback(JNIEnv *env);

#endif // JOCL_CL_JNI_UTILS_HPP
