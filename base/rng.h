/**
 * \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_RNG_H_
#define VECGEOM_BASE_RNG_H_

#include "base/global.h"

namespace vecgeom {

class RNG {

private:

  std::mt19937 rng;
  std::uniform_real_distribution<> uniform_dist;

protected:

  RNG() : rng(0), uniform_dist(0, 1) {}

public:

  static RNG& Instance() {
    static RNG instance;
    return instance;
  }

  VECGEOM_INLINE
  Precision uniform() { return uniform_dist(rng); }

private:

  RNG(RNG const&);
  RNG& operator=(RNG const&);

};

} // End namespace vecgeom

#endif // VECGEOM_BASE_RNG_H_