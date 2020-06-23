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

import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.DeviceEvent;
import ibis.cashmere.constellation.CommandStream;
import ibis.cashmere.constellation.Pointer;

// helper class to make calling FFTs a bit easier.
public class FFT {

    public static FFT_Interface fft;

    public static synchronized void initializeFFT(Device d, CommandStream q, int height, int width) {
        if (fft == null) {
            String env = System.getProperty("cashmere.platform", "opencl");
            String name = FFT.class.getPackage().getName() + "." + env + ".FFT";
            Class<?> cl;
            try {
                cl = Class.forName(name);
            } catch(ClassNotFoundException e) {
                try {
                    cl = Thread.currentThread().getContextClassLoader().loadClass(name);
                } catch(ClassNotFoundException e1) {
                    throw new Error("Could not load class " + name, e1);
                }
            }
            try {
                fft = (FFT_Interface) cl.newInstance();
            } catch(Throwable e) {
                throw new Error("Could not instantiate class " + name, e);
            }
            fft.initializeFFT(d, q, height, width);
        }
    }

    public static DeviceEvent performFFT(CommandStream queue, int h, int w,
	    Pointer buffer, Pointer temp, boolean forward,
	    DeviceEvent[] event_wait_list) {
        return fft.performFFT(queue, h, w, buffer, temp, forward, event_wait_list);
    }

    public static synchronized void deinitializeFFT() {
        if (fft != null) {
            fft.deinitializeFFT();
            fft = null;
        }
    }
}
