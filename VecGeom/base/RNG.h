/// \file RNG.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_RNG_H_
#define VECGEOM_BASE_RNG_H_

#include "VecGeom/base/Global.h"

#include <cstdint>
#include <random>

namespace vecgeom {

/**
 * @brief Thread-local singleton random number generator on CPU.
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
  using Seed_t = std::mt19937::result_type;

  void seed(unsigned long seed_val) { rng.seed(static_cast<Seed_t>(seed_val)); }

  /**
   * Build a deterministic seed for one logical stream starting from a base
   * seed. The stream id is part of the contract so callers can reproduce a
   * failing randomized test independent of which worker thread executes it.
   */
  static Seed_t MakeStreamSeed(unsigned long base_seed, unsigned long stream_id = 0)
  {
    std::uint64_t z = static_cast<std::uint64_t>(base_seed);
    z += 0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(stream_id);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return static_cast<Seed_t>(z);
  }

  /// Seed an arbitrary RNG instance from a deterministic base-seed/stream-id pair.
  static void SeedEngine(RNG &engine, unsigned long base_seed, unsigned long stream_id = 0)
  {
    engine.seed(MakeStreamSeed(base_seed, stream_id));
  }

  /**
   * Access the calling thread's singleton RNG instance.
   */
  static RNG &Instance()
  {
    // The singleton is thread-local on CPU so randomized helpers can be used
    // safely in parallel without racing on one shared generator.
    static thread_local RNG instance;
    return instance;
  }

  /// Seed the calling thread's RNG stream deterministically.
  static void SeedStream(unsigned long base_seed, unsigned long stream_id = 0)
  {
    SeedEngine(Instance(), base_seed, stream_id);
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

} // namespace vecgeom

#endif // VECGEOM_BASE_RNG_H_
