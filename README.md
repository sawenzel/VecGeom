# Introduction

VecGeom is a geometry modeller library with hit-detection features as needed by particle detector simulation at the LHC and beyond.
It was incubated by a Geant-R&D initiative and the motivation to combine the code of Geant4 and ROOT/TGeo into a single,
better maintainable piece of software within the EU-AIDA program. As such it is close in scope to TGeo and Geant4 geometry modellers. Its main features are:

- Build a hierarchic detector geometry out of simple primitives and use it on the CPU or GPU(CUDA)
- Calculate distances and other geometrical information
- Collision detection and navigation in complex scenes
- SIMD support in various flavours:
  - True vector interfaces to primitives with SIMD acceleration when benefical
  - SIMD acceleration of navigation through the use of special voxelization or bounding box hierarchies
- Runtime specialization of objects to improve execution speed via a factory mechanism and use of C++ templates
- VecGeom also compiles under CUDA
- Few generic kernels serve many instanteations of various simple or vectored interfaces or the CUDA version.

## Building/Installing VecGeom
### Requirements
- [CMake](http://www.cmake.org/) 3.16 or newer for configuration, plus a suitable build tool such as GNU make or Ninja
- C++ compiler supporting a minimum ISO Standard of 17
  - The library has been tested to compile with (but is not necessarily limited to):
    - GCC >= 8
    - Clang >= 10
- [VecCore](https://github.com/root-project/veccore) version 0.8.0 or newer
  - VecGeom can build/install its own copy of VecCore by setting the CMake variable `VECGEOM_BUILTIN_VECCORE` to `ON`
- _Optional_ 
  - [Vc](https://github.com/VcDevel/Vc) 1.3.3 or newer for SIMD support
  - [Xerces-C](https://xerces.apache.org/xerces-c/) for GDML support
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 11.0 or newer for CUDA support
    - _An NVidia GPU with sufficient compute capability must be present on the host system_
    - _It is recommended to use CMake 3.18 or newer for the better native CUDA support_

### Quick Start
```sh
$ mkdir build && cd build
$ cmake ..
$ cmake --build .
$ cmake --build . --target install
```

### Build Options
The table below shows the available CMake options for VecGeom that may be used to customize the build:

|Option|Default|Description|
|------|:-----:|-----------|
|VECGEOM_BACKEND|scalar|Vector backend API to be used|
|VECGEOM_BUILTIN_VECCORE|OFF|Build VecCore and its dependencies from source|
|VECGEOM_CUDA_VOLUME_SPECIALIZATION|OFF|Use specialized volumes for CUDA|
|VECGEOM_DISTANCE_DEBUG|OFF|Enable comparison of calculated distances againt ROOT/Geant4 behind the scenes|
|VECGEOM_EMBREE|OFF|Enable Intel Embree|
|VECGEOM_ENABLE_CUDA|OFF|Enable compilation for CUDA|
|VECGEOM_FAST_MATH|OFF|Enable the -ffast-math compiler option in Release builds|
|VECGEOM_GDML|OFF|Enable GDML persistency. Requres Xerces-C|
|VECGEOM_GDMLDEBUG|OFF|Enable additional debug information in GDML module|
|VECGEOM_BVH_SINGLE|OFF|Enable single precision for BVH traversal|
|VECGEOM_INPLACE_TRANSFORMATIONS|ON|Put transformation as members rather than pointers into PlacedVolume objects|
|VECGEOM_NO_SPECIALIZATION|ON|Disable specialization of volumes|
|VECGEOM_PLANESHELL|ON|Enable the use of PlaneShell class for the trapezoid|
|VECGEOM_QUADRILATERAL_ACCELERATION|ON|Enable SIMD vectorization when looping over quadrilaterals|
|VECGEOM_SANITIZER|OFF|Enable memory sanitizer|
|VECGEOM_SINGLE_PRECISION|OFF|Use single precision throughout the package|
|VECGEOM_USE_CACHED_TRANSFORMATIONS|OFF|Use cached transformations in navigation states|
|VECGEOM_USE_INDEXEDNAVSTATES|ON|Use indices rather than volume pointers in NavigationState objects|
|VECGEOM_USE_NAVINDEX|OFF|Use navigation index table and index states|
|VECGEOM_USE_NAVTUPLE|ON|Use navigation index table and tuple states|
|DVECGEOM_NAVTUPLE_MAXDEPTH|1|Maximum depth for navigation tuple states|
|VECGEOM_VECTOR|sse2|Vector instruction set to be used|

The following options are available for enabling, building, and running tests:

|Option|Default|Description|
|------|:-----:|-----------|
|BUILD_TESTING|ON|Enable build of tests and integration with CTest|
|VECGEOM_GEANT4|OFF|Build with support for Geant4 (https://geant4.web.cern.ch)|
|VECGEOM_ROOT|OFF|Build with support for ROOT (https://root.cern)|
|VECGEOM_TEST_BENCHMARK|OFF|Enable performance comparisons|
|VECGEOM_TEST_COVERAGE|OFF|Enable coverage testing flags|
|VECGEOM_TEST_STATIC_ANALYSIS|OFF|Enable static analysis on VecGeom|
|VECGEOM_TEST_VALIDATION|OFF|Enable validation tests from CMS geometry|

Please note that the Geant4 and ROOT options are only for testing and should not be enabled
in builds for production use. Both Geant4 and ROOT provide their own interfaces for
use of VecGeom by their consumers.

## Documentation
- [Doxygen reference guide](https://lcgapp-services.cern.ch/spi-jenkins/job/VecGeom-Doxygen/doxygen/)

### Note on using VecGeom CUDA Support
If VecGeom is built with `VECGEOM_ENABLE_CUDA`, then it is still usable by CPU-only code but
you must link your binaries to _both_ the `vecgeom` _and_ `vecgeomcuda` libraries.

If your code uses VecGeom's CUDA interface in device code or kernels, then it must either:

1. Link _and_ device-link to the `vecgeomcuda_static` target. In CMake, this is automatically handled
   by, e.g.

   ```cmake
   find_package(VecGeom)
   add_executable(MyCUDA MyCUDA.cu)
   set_target_properties(MyCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
   target_link_libraries(MyCUDA PRIVATE VecGeom::vecgeomcuda_static)
   ```

2. Link to the `vecgeomcuda` shared library, _and_ device-link to the `vecgeomcuda_static` target, e.g.
   in CMake (only CMake 3.18 and newer):

   ```cmake
   find_package(VecGeom)
   add_library(MyCUDA MyCUDA.cu)
   set_target_properties(MyCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
   target_link_libraries(MyCUDA PRIVATE VecGeom::vecgeomcuda)
   # Requires CMake 3.18 or newer
   target_compile_options(MyCUDA PRIVATE $<DEVICE_LINK:$<TARGET_FILE:vecgeomcuda_static>>)
   ```

It is strongly recommended to use the first option unless you must use shared libraries.
It is also the developer's responsibility to handle any further device-linking of `vecgeomcuda`
using libraries that may be required if these libraries expose device/kernel interfaces.

## Bug Reports 
Please report all issues on our [JIRA Issue tracking system](http://sft.its.cern.ch/jira/projects/VECGEOM)
