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

package nl.junglecomputing.common_source_identification;

import java.util.Set;
import java.util.HashSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.jocl.LibUtils;

public class LoadNative {

    private static final Set<String> nativeLibrariesLoaded = new HashSet<String>();
    private static Logger logger = LoggerFactory.getLogger("LoadNative");
    
    public static void loadNativeLibrary(String name) {
	if (logger.isDebugEnabled()) {
	    logger.debug("Request to load native library: {}", name);
	}
	if (!nativeLibrariesLoaded.contains(name)) {
	    nativeLibrariesLoaded.add(name);
	    String fullName = name + "-csicn-1.2-SNAPSHOT";
	    if (logger.isDebugEnabled()) {
		logger.debug("Loading {}", fullName);
	    }
	    LibUtils.loadLibrary(fullName);
	}
    }
}
