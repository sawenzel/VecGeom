/// \file RNG.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_RNG_H_
#define VECGEOM_BASE_RNG_H_

#include "VecGeom/base/Global.h"

#include <random>

namespace vecgeom {

/**
 * @brief Singleton random number generator.
 */
class RNG {

private:
  std::mt19937 rng;
  std::uniform_real_distribution<> uniform_dist;

  VECGEOM_FORCE_INLINE
  Precision GetUniform() { return uniform_dist(rng); }


public:
  RNG() : rng(0), uniform_dist(0, 1) {}

public:
  void seed(unsigned long seed_val) { rng.seed(seed_val); }

  /**
   * Access singleton instance.
   */
  static RNG &Instance()
  {
    static RNG instance;
    return instance;
  }

  /**
   * @return Uniformly distributed floating point number between 0 and 1 unless
   *         range arguments are passed.
   */
  Precision uniform(const Precision min = 0., const Precision max = 1.) { return min + (max - min) * GetUniform(); }

  int Poisson(const Precision lambda)
  {
    int k                  = 0;
    const Precision target = exp(-lambda);
    Precision p            = GetUniform();
    while (p < target) {
      p *= GetUniform();
      ++k;
    }
    return k;
  }

  // interface for ROOT compatibility
  Precision Gaus(Precision ave = 0.0, Precision sig = 1.0) { return Gauss(ave, sig); }

  Precision Gauss(Precision ave = 0.0, Precision sig = 1.0)
  {
    Precision x1, x2, w;

    do {
      x1 = 2.0 * GetUniform() - 1.0;
      x2 = 2.0 * GetUniform() - 1.0;
      w  = x1 * x1 + x2 * x2;
    } while (w >= 1.0);

    w = std::sqrt((-2.0 * std::log(w)) / w);
    return ave + (x1 * w * sig);
  }

  /**
   * Uniformly distributed array of floating point number between 0 and 1 unless
   *         range arguments are passed.
   */
  void uniform_array(size_t n, Precision *array, const Precision min = 0., const Precision max = 1.)
  {
    for (size_t i = 0; i < n; ++i) {
      array[i] = min + (max - min) * GetUniform();
    }
  }

private:
  RNG(RNG const &);
  RNG &operator=(RNG const &);
};

} // End global namespace

#endif // VECGEOM_BASE_RNG_H_
