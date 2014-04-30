/**
 * @file rng.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_RNG_H_
#define VECGEOM_BASE_RNG_H_

#include "base/global.h"

#include <random>

namespace VECGEOM_NAMESPACE {

/**
 * @brief Singleton random number generator.
 */
class RNG {

private:

  std::mt19937 rng;
  std::uniform_real_distribution<> uniform_dist;

protected:

  RNG() : rng(0), uniform_dist(0, 1) {}

public:

  /**
   * Access singleton instance.
   */
  static RNG& Instance() {
    static RNG instance;
    return instance;
  }

  /**
   * @return Uniformly distributed floating point number between 0 and 1 unless
   *         range arguments are passed.
   */
  VECGEOM_INLINE
  Precision uniform(const Precision min = 0., const Precision max = 1.) {
    return min + (max - min) * uniform_dist(rng);
  }

private:

  RNG(RNG const&);
  RNG& operator=(RNG const&);

};

} // End global namespace

#endif // VECGEOM_BASE_RNG_H_