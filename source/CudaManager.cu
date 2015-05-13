/// \file CudaManager.cu
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "management/CudaManager.h"

#include <stdio.h>

#include "backend/cuda/Backend.h"

namespace vecgeom {
inline namespace cuda {

__global__
void CudaManagerPrintGeometryKernel(
    vecgeom::cuda::VPlacedVolume const *const world) {
  printf("Geometry loaded on GPU:\n");
  world->PrintContent();
}

void CudaManagerPrintGeometry(vecgeom::cuda::VPlacedVolume const *const world) {
  CudaManagerPrintGeometryKernel<<<1, 1>>>(world);
  cxx::CudaAssertError();
  cudaDeviceSynchronize();
}

} } // End namespace vecgeom

#ifdef VECGEOM_NVCC_SINGLE_OBJ // Cuda single compilation

#include "source/AOS3D.cpp"
#include "source/Vector.cpp"
#include "source/SOA3D.cpp"
#include "source/Transformation3D.cpp"

#include "source/LogicalVolume.cpp"
#include "source/PlacedVolume.cpp"
#include "source/UnplacedVolume.cpp"

#include "source/PlacedBox.cpp"
#include "source/UnplacedBox.cpp"
#include "source/SpecializedBox.cpp"

#include "source/PlacedCone.cpp"
#include "source/UnplacedCone.cpp"

#include "source/PlacedTube.cpp"
#include "source/UnplacedTube.cpp"

#include "source/PlacedTorus.cpp"
#include "source/UnplacedTorus.cpp"

#include "source/PlacedTrd.cpp"
#include "source/UnplacedTrd.cpp"

#include "source/PlacedParallelepiped.cpp"
#include "source/UnplacedParallelepiped.cpp"

#include "source/PlacedParaboloid.cpp"
#include "source/UnplacedParaboloid.cpp"

#include "source/PlacedTrapezoid.cpp"
#include "source/UnplacedTrapezoid.cpp"


#include "source/NavigationState.cpp"
#include "source/SimpleNavigator.cpp"

#include "source/UnplacedOrb.cpp"
#include "source/PlacedOrb.cpp"

#include "source/UnplacedSphere.cpp"
#include "source/PlacedSphere.cpp"

#include "source/UnplacedBooleanVolume.cpp"
#include "source/PlacedBooleanVolume.cpp"

#endif
