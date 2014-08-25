/// \file BenchmarkResult.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_
#define VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_

#include "base/Global.h"

#include <ostream>

namespace vecgeom {

enum EBenchmarkedLibrary {
  kBenchmarkSpecialized = 0,
  kBenchmarkVectorized = 1,
  kBenchmarkUnspecialized = 2,
  kBenchmarkCuda = 3,
  kBenchmarkUSolids = 4,
  kBenchmarkRoot = 5,
  kBenchmarkCudaMemory = 6,
  kBenchmarkGeant4 = 7
};

enum EBenchmarkedMethod {
  kBenchmarkContains = 0,
  kBenchmarkInside = 1,
  kBenchmarkDistanceToIn = 2,
  kBenchmarkSafetyToIn = 3,
  kBenchmarkDistanceToOut = 4,
  kBenchmarkSafetyToOut = 5
};

struct BenchmarkResult {
public:
  const Precision elapsed;
  const EBenchmarkedMethod method;
  const EBenchmarkedLibrary library;
  static char const *const fgMethodLabels[];
  static char const *const fgLibraryLabels[];
  const unsigned repetitions;
  const unsigned volumes;
  const unsigned points;
  const Precision bias;
  void WriteToCsv(std::ostream &os);
  static void WriteCsvHeader(std::ostream &os);
};

std::ostream& operator<<(std::ostream &os, BenchmarkResult const &benchmark);

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_BENCHMARKRESULT_H_