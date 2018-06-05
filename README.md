# OpenMPF Contrib Components

Welcome to the Open Media Processing Framework (OpenMPF) Contrib Component Project!

## What is the OpenMPF?

OpenMPF provides a platform to perform content detection and extraction on bulk multimedia, enabling users to analyze, search, and share information through the extraction of objects, keywords, thumbnails, and other contextual data.

OpenMPF enables users to build configurable media processing pipelines, enabling the rapid development and deployment of analytic algorithms and large-scale media processing applications.

### Search and Share

Simplify large-scale media processing and enable the extraction of meaningful content

### Open API

Apply cutting-edge algorithms such as face detection and object classification

### Flexible Architecture

Integrate into your existing environment or use OpenMPF as a standalone application

## Overview

This repository contains source code for Open Media Processing Framework (OpenMPF) components licensed under a copyleft license, such as GPL, a restrictive license, or a license that is not compatible with Apache 2.0.

Included in this repository are the following C++ algorithms:
- OpenCV-based person
- OpenCV-based motion detection

## Where Am I?

- [Parent OpenMPF Project](https://github.com/openmpf/openmpf-projects)
- [OpenMPF Core](https://github.com/openmpf/openmpf)
- Components
    * [OpenMPF Standard Components](https://github.com/openmpf/openmpf-components)
    * [OpenMPF Contributed Components](https://github.com/openmpf/openmpf-contrib-components) ( **You are here** )
- Component APIs:
    * [OpenMPF C++ Component SDK](https://github.com/openmpf/openmpf-cpp-component-sdk)
    * [OpenMPF Java Component SDK](https://github.com/openmpf/openmpf-java-component-sdk)
    * [OpenMPF Python Component SDK](https://github.com/openmpf/openmpf-python-component-sdk)
- [OpenMPF Build Tools](https://github.com/openmpf/openmpf-build-tools)
- [OpenMPF Web Site Source](https://github.com/openmpf/openmpf.github.io)

## Getting Started

### Building All C++ Components

- In order to build the C++ components you must first install the  [OpenMPF C++ Component SDK](https://github.com/openmpf/openmpf-cpp-component-sdk).
- cd into the `openmpf-contrib-components/cpp directory`.
- Run the following commands:
```
mkdir build
cd build
cmake3 ..
make install
```

- The built plugin packages will be created in `openmpf-contrib-components/cpp/build/plugin-packages`.

**Building Individual C++ Components**

If you would like to only build a single component, you can cd into that component's directory and run the build commands listed above.

### Installing and Registering a Component

Once a component is built, follow the Installing and Registering a Component
[Installing and Registering a Component](https://openmpf.github.io/docs/site/Packaging-and-Registering-a-Component/#installing-and-registering-a-component)
instructions to utilize the component in OpenMPF

### Project Website

For more information about OpenMPF, including documentation, guides, and other material, visit our website: [https://openmpf.github.io/](https://openmpf.github.io/)

### Project Workboard

For a latest snapshot of what tasks are being worked on, what's available to pick up, and where the project stands as a whole, check out our [workboard](https://github.com/orgs/openmpf/projects/3).


