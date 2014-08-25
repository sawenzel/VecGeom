/// \file BenchmarkResult.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "benchmarking/BenchmarkResult.h"

namespace vecgeom {

char const *const BenchmarkResult::fgLibraryLabels[] = {
  "Specialized",
  "Vectorized",
  "Unspecialized",
  "CUDA",
  "USolids",
  "ROOT",
  "CUDAMemory",
  "Geant4"
};

char const *const BenchmarkResult::fgMethodLabels[] = {
  "Contains",
  "Inside",
  "DistanceToIn",
  "SafetyToIn",
  "DistanceToOut",
  "SafetyToOut"
};

void BenchmarkResult::WriteCsvHeader(std::ostream &os) {
  os << "elapsed,method,library,repetitions,volumes,points,bias\n";
}

void BenchmarkResult::WriteToCsv(std::ostream &os) {
  os << elapsed << "," << fgMethodLabels[method] << ","
     << fgLibraryLabels[library] << "," << repetitions << "," << volumes << ","
     << points << "," << bias << "\n";
}

std::ostream& operator<<(std::ostream &os, BenchmarkResult const &benchmark) {
  os << benchmark.elapsed << "s | "
     << BenchmarkResult::fgMethodLabels[benchmark.method] << " for "
     << BenchmarkResult::fgLibraryLabels[benchmark.library] << ", using "
     << benchmark.volumes << " volumes and " << benchmark.points
     << " points for " << benchmark.repetitions << " repetitions with a "
     << benchmark.bias << " bias.\n";
  return os;
}

} // End namespace vecgeom
