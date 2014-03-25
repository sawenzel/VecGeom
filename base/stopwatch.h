/**
 * @file stopwatch.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_STOPWATCH_H_
#define VECGEOM_BASE_STOPWATCH_H_

#include "base/global.h"

#include "tbb/tick_count.h"

namespace VECGEOM_NAMESPACE {

/**
 * @brief Timer for benchmarking purposes implemented using the Intel TBB
 *        library.
 */
class Stopwatch {

private:

  tbb::tick_count t1;
  tbb::tick_count t2;

public:

  void Start() { t1 = tbb::tick_count::now(); }

  /**
   * @return Elapsed time since start.
   */
  double Stop() {
    t2 = tbb::tick_count::now();
    return Elapsed();
  }

  double Elapsed() const { return (t2-t1).seconds(); }

};

} // End global namespace

#endif // VECGEOM_BASE_STOPWATCH_H_