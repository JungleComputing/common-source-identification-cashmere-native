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


import java.io.IOException;

import java.nio.ByteBuffer;

import ibis.cashmere.constellation.Buffer;

// helper class to read in JPG files since the java AWT version is not multithreaded...
public class ReadJPG {

    static native int readJPG(ByteBuffer output, String fileName);

    public static void readJPG(Buffer output, String fileName) 
	throws IOException {

	int result = readJPG(output.getByteBuffer(), fileName);
	if (result != 0) {
	    throw new IOException("readJPG error code: " + result);
	}
    }

}
