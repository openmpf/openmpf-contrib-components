# Overview

This repository contains source code for Open Media Processing Framework (OpenMPF) components 
licensed under a copyleft license, such as GPL, a restrictive license, or a license that is not 
compatible with Apache 2.0. 

# Building the C++ Components
* In order to build the C++ components you must first install the 
  [OpenMPF C++ Component SDK](https://github.com/openmpf/openmpf-cpp-component-sdk).
* cd into the `openmpf-contrib-components/cpp` directory.
* Run the following commands:
```
mkdir build
cd build
cmake3 ..
make install
```
* The built plugin packages will be created in `openmpf-contrib-components/cpp/build/plugin-packages`.

### Building Individual C++ Components
If you would like to only build a single component, you can cd into that component's directory and run the
build commands listed above.
