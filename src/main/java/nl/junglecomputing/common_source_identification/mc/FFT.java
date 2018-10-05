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

package nl.junglecomputing.common_source_identification.mc;

import org.jocl.cl_event;
import org.jocl.cl_context;
import org.jocl.cl_command_queue;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.CLException;

import org.jocl.LibUtils;

import nl.junglecomputing.common_source_identification.LoadNative;

// helper class to make calling FFTs a bit easier.
public class FFT {

    static {
	LoadNative.loadNativeLibrary("clFFT");
	LoadNative.loadNativeLibrary("fft");
    }

    // native functions to initialize the FFT library
    public static native int initializeFFT(cl_context context, cl_command_queue queue, int height, int width);
    public static native int deinitializeFFT();

    private native static int doFFT(cl_command_queue queue, int h, int w,
            Pointer buffer, Pointer temp, boolean forward,
            int num_events_in_wait_list, cl_event[] event_wait_list,
            cl_event event);

    public static void performFFT(cl_command_queue queue, int h, int w,
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
