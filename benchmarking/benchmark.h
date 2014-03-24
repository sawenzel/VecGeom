/**
 * @file benchmark.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BENCHMARKING_BENCHMARK_H_
#define VECGEOM_BENCHMARKING_BENCHMARK_H_

#include <vector>

#include "base/global.h"

#include "base/track_container.h"
#include "management/volume_pointers.h"
#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

struct BenchmarkResult;

class Benchmark {

private:

  unsigned repetitions_ = 1e3;
  unsigned verbose_ = 0;

protected:

  VPlacedVolume const *world_ = NULL;
  std::vector<BenchmarkResult> results_;

public:
  
  Benchmark() {}

  Benchmark(VPlacedVolume const *const world);

  virtual ~Benchmark() {}

  virtual void BenchmarkAll() =0;
  virtual void BenchmarkSpecialized() =0;
 // virtual void BenchmarkSpecializedVec() =0;
  virtual void BenchmarkUnspecialized() =0;
  virtual void BenchmarkUSolids() =0;
  virtual void BenchmarkRoot() =0;

  BenchmarkResult PopResult();
  std::vector<BenchmarkResult> PopResults();

  VPlacedVolume const* world() const;

  unsigned repetitions() const { return repetitions_; }

  unsigned verbose() const { return verbose_; }

  std::vector<BenchmarkResult> results() const { return results_; }

  void set_world(VPlacedVolume const *const world);

  void set_repetitions(const unsigned repetitions) {
    repetitions_ = repetitions;
  }

  void set_verbose(const unsigned verbose) { verbose_ = verbose; }

};

struct BenchmarkResult {
public:
  const Precision elapsed;
  const BenchmarkType type;
  static char const *const benchmark_labels[];
  const unsigned repetitions;
  const unsigned volumes;
  const unsigned points;
  const Precision bias;
  friend std::ostream& operator<<(std::ostream &os,
                                  BenchmarkResult const &benchmark);
};



} // End global namespace

#endif // VECGEOM_BENCHMARKING_BENCHMARK_H_
