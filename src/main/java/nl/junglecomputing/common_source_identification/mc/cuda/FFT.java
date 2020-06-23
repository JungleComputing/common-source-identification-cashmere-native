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

package nl.junglecomputing.common_source_identification.mc.cuda;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.Argument;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.cashmere.constellation.DeviceEvent;
import ibis.cashmere.constellation.CommandStream;
import ibis.cashmere.constellation.Pointer;
import ibis.cashmere.constellation.deviceImpl.jcuda.CudaPointer;
import ibis.cashmere.constellation.deviceImpl.jcuda.CudaDevice;
import ibis.cashmere.constellation.deviceImpl.jcuda.CudaEvent;
import ibis.cashmere.constellation.deviceImpl.jcuda.CudaCommandStream;
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuStreamWaitEvent;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;

import java.util.Map;
import java.util.HashMap;

import jcuda.runtime.cudaStream_t;
import jcuda.jcufft.*;
import jcuda.driver.*;

// helper class to make calling FFTs a bit easier.
public class FFT implements nl.junglecomputing.common_source_identification.mc.FFT_Interface {

    private Map<CUstream, cufftHandle> plans = new HashMap<>();
    private Map<CUstream, Device> devices = new HashMap<>();

    boolean initialized = false;

    public synchronized void initializeFFT(Device d, CommandStream q, int height, int width) {
        //initialize CUFFT
        if (! initialized) {
            JCufft.initialize();
            JCufft.setExceptionsEnabled(true);
            initialized = true;
        }
        cufftHandle plan = new cufftHandle();
        CUstream stream = ((CudaCommandStream) q).getQueue();

        CUcontext ctxt = ((CudaDevice) d).getContext();
        cuCtxSetCurrent(ctxt);

        //create CUFFT plan and associate with stream
        int res = JCufft.cufftPlan2d(plan, height, width, cufftType.CUFFT_C2C);
        if (res != cufftResult.CUFFT_SUCCESS) {
            throw new Error("Could not create CUFFT plan");
        }
        res = JCufft.cufftSetStream(plan, new cudaStream_t(stream));
        if (res != cufftResult.CUFFT_SUCCESS) {
            throw new Error("Error while associating plan with stream");
        }
        plans.put(stream, plan);
        devices.put(stream, d);
    }

    public DeviceEvent performFFT(CommandStream queue, int h, int w,
	    Pointer buffer, Pointer temp, boolean forward,
	    DeviceEvent[] event_wait_list) {
        //insert waits for the event wait list.
        CUstream str = ((CudaCommandStream) queue).getQueue();
        cufftHandle plan = plans.get(str);
        if (plan == null) {
            throw new Error("no plan associated with the specified stream");
        }
        if (event_wait_list != null) {
            for (DeviceEvent evnt : event_wait_list) {
                cuStreamWaitEvent(str, ((CudaEvent) evnt).getEvent(), 0);
            }
        }

        //apply complex to complex Fourier transform
        //Note, cufftExecC2C only allows input and output parameters, no temp.
        //synchronized on plan to make sure that it is used at most once at a time.
        synchronized(plan) {
            JCufft.cufftExecC2C(plan,
                ((CudaPointer) buffer).getPtr(),
                ((CudaPointer) buffer).getPtr(),
                forward ? JCufft.CUFFT_FORWARD : JCufft.CUFFT_INVERSE);
            cuStreamSynchronize(str);
        }

        if (! forward) {
            // normalize result, since cufft does not do it.
            /*
             * Should we add an event to wait upon here, before we normalize? I don't think so,
             * because we normalize on the same queue.
            CUevent execEvent = new CUevent();
            cuEventCreate(execEvent, jcuda.driver.CUevent_flags.CU_EVENT_BLOCKING_SYNC);
            cuEventRecord(execEvent, str);
            cuStreamWaitEvent(str, execEvent, 0);
            */
            int n = h * w;
            Device device = devices.get(str);
            Kernel normalizeKernel;
            try {
                normalizeKernel = Cashmere.getKernel("normalizeKernel", device);
            } catch(Exception e) {
                throw new Error("Could not get kernel", e);
            }
            KernelLaunch nkl = normalizeKernel.createLaunch();
            nkl.setArgument(n, Argument.Direction.IN);
            nkl.setArgumentNoCopy(buffer, Argument.Direction.INOUT);
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                                    : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            nkl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        }
        
        // create an event that we can synchronize on.
        CUevent execEvent = new CUevent();
        cuEventCreate(execEvent, jcuda.driver.CUevent_flags.CU_EVENT_BLOCKING_SYNC);
        cuEventRecord(execEvent, str);
        return new CudaEvent(execEvent);
    }

    public synchronized void deinitializeFFT() {
        if (initialized) {
            initialized = false;
            for (Map.Entry<CUstream, cufftHandle> entry : plans.entrySet()) {
                JCufft.cufftDestroy(entry.getValue());
            }
        }
    }
}
