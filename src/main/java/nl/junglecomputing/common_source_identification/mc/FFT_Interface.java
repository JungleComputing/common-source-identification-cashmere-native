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

import org.jocl.CLException;

import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.DeviceEvent;
import ibis.cashmere.constellation.CommandStream;
import ibis.cashmere.constellation.Pointer;

public interface FFT_Interface {

    public void initializeFFT(Device d, CommandStream q, int height, int width);

    public DeviceEvent performFFT(CommandStream queue, int h, int w,
	    Pointer buffer, Pointer temp, boolean forward,
	    DeviceEvent[] event_wait_list);

    public void deinitializeFFT();
}
