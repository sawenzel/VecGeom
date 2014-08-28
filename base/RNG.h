/// \file RNG.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_RNG_H_
#define VECGEOM_BASE_RNG_H_

#include "base/Global.h"

#ifdef VECGEOM_NVCC
#include <cuda.h>
#include <curand_kernel.h>
#else
#include <random>
#endif

namespace VECGEOM_NAMESPACE {

/**
 * @brief Singleton random number generator.
 */
class RNG {

private:

#ifdef VECGEOM_NVCC
#ifdef __CUDA_ARCH__
  curandState fState;
  
  VECGEOM_CUDA_HEADER_DEVICE
  VECGEOM_INLINE
  Precision GetUniform() {
     return curand_uniform(&fState);
  }
#else
  VECGEOM_CUDA_HEADER_HOST
  VECGEOM_INLINE
  Precision GetUniform() {
     return (Precision) rand() / RAND_MAX;
  }
#endif

#else
  std::mt19937 rng;
  std::uniform_real_distribution<> uniform_dist;

  VECGEOM_INLINE
  Precision GetUniform() {
     return uniform_dist(rng);
  }

#endif

protected:

#ifdef VECGEOM_NVCC
#ifdef __CUDA_ARCH__
// The state should really be 'thread' specific
  VECGEOM_CUDA_HEADER_DEVICE
  RNG() { curand_init(0 /*seed */, 0 /* subsequence */, 0 /* offset */ , &fState); }
#else
  RNG() {}
#endif
#else
  RNG() : rng(0), uniform_dist(0, 1) {}
#endif

public:

  /**
   * Access singleton instance.
   */
#ifdef VECGEOM_NVCC
#ifdef __CUDA_ARCH__
  VECGEOM_CUDA_HEADER_DEVICE
  static RNG& Instance() {
     __shared__ RNG instance;
#else
  VECGEOM_CUDA_HEADER_HOST
  static RNG Instance() {
     RNG instance;
#endif
     return instance;
  }
#else
  static RNG& Instance() {
    static RNG instance;
    return instance;
  }
#endif

  /**
   * @return Uniformly distributed floating point number between 0 and 1 unless
   *         range arguments are passed.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision uniform(const Precision min = 0., const Precision max = 1.) {
    return min + (max - min) * GetUniform();
  }

private:

  RNG(RNG const&);
  RNG& operator=(RNG const&);

};

} // End global namespace

#endif // VECGEOM_BASE_RNG_H_
