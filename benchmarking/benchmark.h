#ifndef VECGEOM_BENCHMARKING_BENCHMARK_H_
#define VECGEOM_BENCHMARKING_BENCHMARK_H_

#include <iostream>
#include <random>
#include <vector>

#include "base/global.h"

#include "volumes/placed_volume.h"

namespace vecgeom {

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

  Benchmark(LogicalVolume const *const world);

  virtual ~Benchmark() {}

  virtual void BenchmarkAll() =0;
  virtual void BenchmarkSpecialized() =0;
  virtual void BenchmarkUnspecialized() =0;
  virtual void BenchmarkUSolids() =0;
  virtual void BenchmarkRoot() =0;

  BenchmarkResult PopResult();
  std::vector<BenchmarkResult> PopResults();

  LogicalVolume const* world() const;

  unsigned repetitions() const { return repetitions_; }

  unsigned verbose() const { return verbose_; }

  std::vector<BenchmarkResult> results() const { return results_; }

  void set_world(LogicalVolume const *const world);

  void set_repetitions(const unsigned repetitions) {
    repetitions_ = repetitions;
  }

  void set_verbose(const unsigned verbose) { verbose_ = verbose; }

  static Vector3D<Precision> SamplePoint(Vector3D<Precision> const &size,
                                         const Precision scale = 1);

  static Vector3D<Precision> SampleDirection();

  static void FillRandomDirections(TrackContainer<Precision> *const dirs);

  VECGEOM_INLINE
  static bool IsFacingVolume(Vector3D<Precision> const &point,
                             Vector3D<Precision> const &dir,
                             VPlacedVolume const &volume);

  static void FillBiasedDirections(VPlacedVolume const &volume,
                                   TrackContainer<Precision> const &points,
                                   const Precision bias,
                                   TrackContainer<Precision> *const dirs);

  static void FillBiasedDirections(LogicalVolume const &volume,
                                   TrackContainer<Precision> const &points,
                                   const Precision bias,
                                   TrackContainer<Precision> *const dirs);

  static void FillUncontainedPoints(VPlacedVolume const &volume,
                                    TrackContainer<Precision> *const points);

  static void FillUncontainedPoints(LogicalVolume const &volume,
                                    TrackContainer<Precision> *const points);

};

enum BenchmarkType {kSpecialized, kUnspecialized, kUSolids, kRoot};

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

VECGEOM_INLINE
bool Benchmark::IsFacingVolume(Vector3D<Precision> const &point,
                               Vector3D<Precision> const &dir,
                               VPlacedVolume const &volume) {
  return volume.DistanceToIn(point, dir, kInfinity) < kInfinity;
}

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_BENCHMARK_H_