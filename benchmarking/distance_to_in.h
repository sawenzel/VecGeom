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

namespace vecgeom {

class DistanceToInBenchmarker : public Benchmark {

private:

  std::vector<VolumePointers> volumes_;
  unsigned n_points_;
  double bias_;
  unsigned pool_multiplier_;
  SOA3D<Precision> *point_pool_;
  SOA3D<Precision> *dir_pool_;
  // needed to call basket functions
  Precision * psteps_;

public:

  virtual void BenchmarkAll();
  virtual void BenchmarkSpecialized();
  virtual void BenchmarkUnspecialized();
  virtual void BenchmarkSpecializedVec();
#ifdef VECGEOM_USOLIDS
  virtual void BenchmarkUSolids();
#endif
#ifdef VECGEOM_ROOT
  virtual void BenchmarkRoot();
#endif
#ifdef VECGEOM_CUDA
  virtual void BenchmarkCuda();
#endif
  
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
#ifdef VECGEOM_USOLIDS
  BenchmarkResult RunUSolids(double *distances) const;
#endif
#ifdef VECGEOM_ROOT
  BenchmarkResult RunRoot(double *distances) const;
#endif
#ifdef VECGEOM_CUDA
  double RunCuda(
    Precision *const pos_x, Precision *const pos_y,
    Precision *const pos_z, Precision *const dir_x,
    Precision *const dir_y, Precision *const dir_z,
    Precision *const distances) const;
#endif

  void PrepareBenchmark();

  double* AllocateDistance() const;

  static void FreeDistance(double *const distance);

};

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_DISTANCETOIN_H_