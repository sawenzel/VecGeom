/**
 * \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_STOPWATCH_H_
#define VECGEOM_BASE_STOPWATCH_H_

#include "tbb/tick_count.h"

class Stopwatch {

private:

  tbb::tick_count t1;
  tbb::tick_count t2;

public:

  void Start() { t1 = tbb::tick_count::now(); }

  double Stop() {
    t2 = tbb::tick_count::now();
    return Elapsed();
  }

  double Elapsed() const { return (t2-t1).seconds(); }

};

#endif // VECGEOM_BASE_STOPWATCH_H_