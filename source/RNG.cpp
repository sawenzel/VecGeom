/// \file RNG.cpp
/// \author Philippe Canal (pcanal@fnal.gov)

#include "base/RNG.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_NVCC
   class RNG;

   // Emulating static class member ..
   namespace RNGvar {
      VECGEOM_CUDA_HEADER_DEVICE unsigned long gMaxInstance;
      VECGEOM_CUDA_HEADER_DEVICE RNG **gInstances;
   }
#endif
}
}
