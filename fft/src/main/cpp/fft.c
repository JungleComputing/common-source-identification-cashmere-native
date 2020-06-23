/*
 * Copyright 2018 Vrije Universiteit Amsterdam, The Netherlands
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>

#include <CL/opencl.h>
#include <jni.h>

#include <vector>

#include <CLJNIUtils.hpp>
#include <PointerUtils.hpp>


#include <fft.h>

/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>


/* We need a planHandle per device
   For that we need:
     - the nativeQueue
     - the nativeContext
     - height, width
*/

/* Create a list of plan handles, and matching queues. Note: will not work if passed NULL queue.
 */
std::vector<clfftPlanHandle> planHandles;
std::vector<cl_command_queue> nativeQueues;


JNIEXPORT jint JNICALL Java_nl_junglecomputing_common_1source_1identification_mc_opencl_FFT_initializeFFT
   (JNIEnv *env, jclass c, jobject context, jobject queue, jint height, 
    jint width) {

  cl_int err = 0;
  
  cl_context nativeContext = NULL;
  cl_command_queue nativeQueue = NULL;
    
  if (context != NULL) {
    nativeContext = (cl_context) env->GetLongField(context, NativePointerObject_nativePointer);
  }
  if (queue != NULL) {
    nativeQueue = (cl_command_queue) env->GetLongField(queue, NativePointerObject_nativePointer);
  }

  clfftDim dim = CLFFT_2D;
  size_t clLengths[2] = {(size_t) width, (size_t) height};

  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  if (err) {
    printf("%s, err: %d\n", "clfftInitSetupData", err);
    fflush(stdout);
    return err;
  }
  err = clfftSetup(&fftSetup);
  if (err) {
    printf("%s, err: %d\n", "clfftSetup", err);
    fflush(stdout);
    return err;
  }

  clfftPlanHandle planHandle;

  // Create a default plan for a complex FFT.
  err = clfftCreateDefaultPlan(&planHandle, nativeContext, dim, clLengths);
  if (err) {
    printf("%s, err: %d\n", "clfftCreateDefaultPlan", err);
    fflush(stdout);
    return err;
  }

  err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
  if (err) {
    printf("%s, err: %d\n", "clfftSetPlanPrecision", err);
    fflush(stdout);
    return err;
  }

  err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
  if (err) {
    printf("%s, err: %d\n", "clfftSetLayout", err);
    fflush(stdout);
    return err;
  }

  err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
  if (err) {
    printf("%s, err: %d\n", "clfftSetResultLocation", err);
    fflush(stdout);
    return err;
  }

  // Bake the plan.
  err = clfftBakePlan(planHandle, 1, &nativeQueue, NULL, NULL);
  if (err) {
    printf("%s, err: %d\n", "clfftBakePlan", err);
    fflush(stdout);
    return err;
  }

  planHandles.push_back(planHandle);
  nativeQueues.push_back(nativeQueue);

  return err;
}



JNIEXPORT jint JNICALL Java_nl_junglecomputing_common_1source_1identification_mc_opencl_FFT_deinitializeFFT_1native
        (JNIEnv *env, jclass c) {

    cl_int err = 0;

    clfftPlanHandle planHandle = planHandles.back();
    planHandles.pop_back();
    nativeQueues.pop_back();

    err = clfftDestroyPlan( &planHandle );

    if (err) return err;

    if (planHandles.size() == 0) {
	clfftTeardown();
    }

    return err;
}


JNIEXPORT jint JNICALL Java_nl_junglecomputing_common_1source_1identification_mc_opencl_FFT_doFFT
    (JNIEnv *env, jclass c, jobject queue, jint h, jint w,
    jobject bufferPtr, jobject tempPtr, jboolean forward, jint num_events_in_wait_list, 
    jobjectArray event_wait_list, jobject event) {
    
  cl_command_queue nativeQueue = NULL;

  cl_mem *nativeBufferPtr = NULL;
  cl_mem *nativeTempPtr = NULL;

  cl_uint nativeNum_events_in_wait_list = 0;
  cl_event *nativeEvent_wait_list = NULL;

  cl_event nativeEvent = NULL;
  cl_event *nativeEventPointer = NULL;

  cl_int err = 0;

  int nativeForward = forward != JNI_FALSE;
  
  if (queue != NULL) {
    nativeQueue = (cl_command_queue) env->GetLongField(queue, NativePointerObject_nativePointer);
  }

  int index = 0;
  for (unsigned i = 0; i < nativeQueues.size(); i++) {
    if (nativeQueues[i] == nativeQueue) {
	index = i;
	break;
    }
  }

  clfftPlanHandle planHandle = planHandles[index];

  PointerData *bufferPtrData = initPointerData(env, bufferPtr);
  if (bufferPtrData == NULL) {
      return CL_INVALID_HOST_PTR;
  }
  PointerData *tempPtrData = initPointerData(env, tempPtr);
  if (tempPtrData == NULL) {
      return CL_INVALID_HOST_PTR;
  }
  nativeBufferPtr = (cl_mem *) bufferPtrData->pointer;
  nativeTempPtr = (cl_mem *) tempPtrData->pointer;

  nativeNum_events_in_wait_list = (cl_uint) num_events_in_wait_list;
  if (event_wait_list != NULL) {
      nativeEvent_wait_list = createEventList(env, event_wait_list, num_events_in_wait_list);
      if (nativeEvent_wait_list == NULL) {
	  return CL_OUT_OF_HOST_MEMORY;
      }
  }
  if (event != NULL) {
      nativeEventPointer = &nativeEvent;
  }

  // Execute the plan.
  err = clfftEnqueueTransform(planHandle, 
			      nativeForward ? CLFFT_FORWARD : CLFFT_BACKWARD, 
			      1, &nativeQueue, nativeNum_events_in_wait_list, 
			      nativeEvent_wait_list, nativeEventPointer, 
			      nativeBufferPtr, NULL, *nativeTempPtr);
  //err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &nativeQueue, 0, NULL, NULL, &nativeBuffer, NULL, NULL);
  //err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &nativeQueue, nativeNum_events_in_wait_list, nativeEvent_wait_list, NULL, &nativeBuffer, NULL, NULL);

  if (err) return err;

  delete[] nativeEvent_wait_list;
  setNativePointer(env, event, (jlong) nativeEvent);

  return err;
}

