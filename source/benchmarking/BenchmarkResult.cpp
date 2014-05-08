/**
 * @file BenchmarkResult.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "benchmarking/BenchmarkResult.h"

namespace vecgeom {

char const *const BenchmarkResult::benchmark_labels[] = {
  "Specialized",
  "Unspecialized",
  "USolids",
  "ROOT"
};

std::ostream& operator<<(std::ostream &os, BenchmarkResult const &benchmark) {
  os << benchmark.elapsed << "s | " << benchmark.volumes << " "
     << BenchmarkResult::benchmark_labels[benchmark.type] << " volumes, "
     << benchmark.points << " points, " << benchmark.bias
     << " bias, repeated " << benchmark.repetitions << " times.";
  return os;
}

} // End namespace vecgeom