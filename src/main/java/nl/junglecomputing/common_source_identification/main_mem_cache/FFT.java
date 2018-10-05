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

package nl.junglecomputing.common_source_identification.main_mem_cache;

import org.jocl.cl_event;
import org.jocl.cl_context;
import org.jocl.cl_command_queue;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.CLException;

import org.jocl.LibUtils;

// helper class to make calling FFTs a bit easier.
class FFT {

    static {
	LoadNative.loadNativeLibrary("clFFT");
	LoadNative.loadNativeLibrary("fft");
    }

    // native functions to initialize the FFT library
    static native int initializeFFT(cl_context context, cl_command_queue queue, int height, int width);
    static native int deinitializeFFT();

    native static int doFFT(cl_command_queue queue, int h, int w,
            Pointer buffer, Pointer temp, boolean forward,
            int num_events_in_wait_list, cl_event[] event_wait_list,
            cl_event event);

    static void performFFT(cl_command_queue queue, int h, int w,
	    Pointer buffer, Pointer temp, boolean forward,
	    int num_events_in_wait_list, cl_event[] event_wait_list,
	    cl_event event) {
	int err = doFFT(queue, h, w, buffer, temp, forward, num_events_in_wait_list,
		event_wait_list, event);
	if (err != 0) {
	    throw new CLException(CL.stringFor_errorCode(err), err);
	}
    }
}
