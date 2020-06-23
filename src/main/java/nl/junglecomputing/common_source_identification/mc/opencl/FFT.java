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

package nl.junglecomputing.common_source_identification.mc.opencl;

import org.jocl.cl_event;
import org.jocl.cl_context;
import org.jocl.cl_command_queue;
import org.jocl.CL;
import org.jocl.CLException;

import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.DeviceEvent;
import ibis.cashmere.constellation.CommandStream;
import ibis.cashmere.constellation.Pointer;

import ibis.cashmere.constellation.deviceImpl.jocl.OpenCLDevice;
import ibis.cashmere.constellation.deviceImpl.jocl.OpenCLEvent;
import ibis.cashmere.constellation.deviceImpl.jocl.OpenCLCommandStream;
import ibis.cashmere.constellation.deviceImpl.jocl.OpenCLPointer;

import org.jocl.LibUtils;

import nl.junglecomputing.common_source_identification.LoadNative;

public class FFT implements nl.junglecomputing.common_source_identification.mc.FFT_Interface {

    static {
	LoadNative.loadNativeLibrary("clFFT");
	LoadNative.loadNativeLibrary("fft");
    }

    // native functions to initialize the FFT library
    private static native int initializeFFT(cl_context context, cl_command_queue queue, int height, int width);
    private static native int deinitializeFFT_native();

    public void deinitializeFFT() {
        deinitializeFFT_native();
    }

    public void initializeFFT(Device d, CommandStream q, int height, int width) {
        initializeFFT(((OpenCLDevice) d).getContext(), ((OpenCLCommandStream) q).getQueue(), height, width);
    }

    private native static int doFFT(cl_command_queue queue, int h, int w,
            org.jocl.Pointer buffer, org.jocl.Pointer temp, boolean forward,
            int num_events_in_wait_list, cl_event[] event_wait_list,
            cl_event event);

    private static void performFFT(cl_command_queue queue, int h, int w,
	    org.jocl.Pointer buffer, org.jocl.Pointer temp, boolean forward,
	    int num_events_in_wait_list, cl_event[] event_wait_list,
	    cl_event event) {
	int err = doFFT(queue, h, w, buffer, temp, forward, num_events_in_wait_list,
		event_wait_list, event);
	if (err != 0) {
	    throw new CLException(CL.stringFor_errorCode(err), err);
	}
    }

    public DeviceEvent performFFT(CommandStream queue, int h, int w,
	    Pointer buffer, Pointer temp, boolean forward,
	    DeviceEvent[] event_wait_list) {
        cl_event evnt = new cl_event();
        int len = event_wait_list == null ? 0 : event_wait_list.length;
        cl_event[] wait_list = null;
        if (len > 0) {
            wait_list = new cl_event[len];
            for (int i = 0; i < len; i++) {
                wait_list[i] = ((OpenCLEvent) event_wait_list[i]).getCLEvent();
            }
        }
	int err = doFFT(((OpenCLCommandStream) queue).getQueue(), h, w,
                ((OpenCLPointer)buffer).getPointer(),
                ((OpenCLPointer)temp).getPointer(),
                forward, len, wait_list, evnt);
	if (err != 0) {
	    throw new CLException(CL.stringFor_errorCode(err), err);
	}
        return new OpenCLEvent(evnt);
    }
}
