/**
 * @file distance_to_in.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BENCHMARKING_DISTANCETOIN_H_
#define VECGEOM_BENCHMARKING_DISTANCETOIN_H_

#include <vector>

#include "base/global.h"

#include "base/soa3d.h"
#include "benchmarking/benchmark.h"
#include "management/volume_pointers.h"
#include "volumes/placed_volume.h"

namespace VECGEOM_NAMESPACE {

class DistanceToInBenchmarker : public Benchmark {

private:

  std::vector<VolumePointers> volumes_;
  unsigned n_points_ = 1024;
  double bias_ = 0.8;
  unsigned pool_multiplier_ = 1;
  SOA3D<Precision> *point_pool_ = NULL;
  SOA3D<Precision> *dir_pool_ = NULL;
  // needed to call basket functions
  Precision * psteps_ = NULL;

public:

  virtual void BenchmarkAll();
  virtual void BenchmarkSpecialized();
  virtual void BenchmarkUnspecialized();
  virtual void BenchmarkUSolids();
  virtual void BenchmarkRoot();
  
  DistanceToInBenchmarker() {}

  DistanceToInBenchmarker(VPlacedVolume const *const world);

  virtual ~DistanceToInBenchmarker();

  unsigned n_points() const { return n_points_; }
  double bias() const { return bias_; }
  unsigned pool_multiplier() const { return pool_multiplier_; }
  void set_n_points(const unsigned n_points) { n_points_ = n_points; }
  void set_bias(const double bias) { bias_ = bias; }
  void set_pool_multiplier(const unsigned pool_multiplier_);

private:
    
  void GenerateVolumePointers(VPlacedVolume const *const vol);

  BenchmarkResult GenerateBenchmarkResult(const double elapsed,
                                    const BenchmarkType type) const;

  BenchmarkResult RunSpecialized(double *distances) const;
  BenchmarkResult RunSpecializedVec(double *distances) const;
  BenchmarkResult RunUnspecialized(double *distances) const;
  BenchmarkResult RunUSolids(double *distances) const;
  BenchmarkResult RunRoot(double *distances) const;

  void PrepareBenchmark();

  double* AllocateDistance() const {
    return (double*) _mm_malloc(n_points_*sizeof(double), kAlignmentBoundary);
  }

  static void FreeDistance(double *const distance) {
    _mm_free(distance);
  }

};

} // End global namespace

#endif // VECGEOM_BENCHMARKING_DISTANCETOIN_H_
