/**
 * @file BenchmarkResult.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_
#define VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_

#include "base/global.h"

#include "management/volume_pointers.h"

#include <ostream>

namespace vecgeom {

struct BenchmarkResult {
public:
  const Precision elapsed;
  const BenchmarkType type;
  static char const *const benchmark_labels[];
  const unsigned repetitions;
  const unsigned volumes;
  const unsigned points;
  const Precision bias;
};

std::ostream& operator<<(std::ostream &os, BenchmarkResult const &benchmark);

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_