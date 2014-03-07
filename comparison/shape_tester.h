#ifndef VECGEOM_COMPARISON_SHAPETESTER_H_
#define VECGEOM_COMPARISON_SHAPETESTER_H_

#include <iostream>
#include <vector>
#include "base/global.h"
#include "base/soa3d.h"
#include "comparison/volume_converter.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

enum BenchmarkType {kSpecialized, kUnspecialized, kUSolids, kRoot};

struct ShapeBenchmark {
public:
  const double elapsed;
  const BenchmarkType type;
  static char const *const benchmark_labels[];
  const unsigned repetitions;
  const unsigned volumes;
  const unsigned points;
  const double bias;
  friend std::ostream& operator<<(std::ostream &os,
                                        ShapeBenchmark const &benchmark) {
    os << benchmark.elapsed << "s | " << benchmark.volumes << " "
       << ShapeBenchmark::benchmark_labels[benchmark.type] << " volumes, "
       << benchmark.points << " points, " << benchmark.bias
       << " bias, repeated " << benchmark.repetitions << " times.";
    return os;
  }
};

class ShapeTester {

private:

  VPlacedVolume const *world_ = NULL;
  std::vector<VolumeConverter> volumes_;
  unsigned n_vols_ = 0;
  unsigned n_points_ = 1<<10;
  unsigned repetitions_ = 1e3;
  double bias_ = 0.8;
  unsigned pool_multiplier_ = 1;
  std::vector<ShapeBenchmark> results_;
  unsigned verbose_ = 0;
  SOA3D<Precision> *point_pool_, *dir_pool_;
  double *steps_ = NULL;

public:

  void BenchmarkAll();
  void BenchmarkSpecialized();
  void BenchmarkUnspecialized();
  void BenchmarkUSolids();
  void BenchmarkROOT();

  ShapeBenchmark PopResult();
  std::vector<ShapeBenchmark> PopResults();
  
  ShapeTester() {}

  ShapeTester(LogicalVolume const *const world);
  
  ~ShapeTester();

  // Accessors

  LogicalVolume const* world() const;

  unsigned n_points() const { return n_points_; }

  unsigned repetitions() const { return repetitions_; }

  double bias() const { return bias_; }

  unsigned pool_multiplier() const { return pool_multiplier_; }

  unsigned verbose() const { return verbose_; }

  std::vector<ShapeBenchmark> results() const { return results_; }

  // Mutators

  void set_world(LogicalVolume const *const world);

  void set_n_points(const unsigned n_points) { n_points_ = n_points; }

  void set_repetitions(const unsigned repetitions) {
    repetitions_ = repetitions;
  }

  void set_bias(const double bias) { bias_ = bias; }

  void set_pool_multiplier(const unsigned pool_multiplier_);

  void set_verbose(const unsigned verbose) { verbose_ = verbose; }

  // Utility

  static Vector3D<double> SamplePoint(Vector3D<double> const &size,
                                      const double scale = 1);

  static Vector3D<double> SampleDirection();

  static void FillRandomDirections(SOA3D<double> *const dirs);

  VECGEOM_INLINE
  static bool IsFacingVolume(Vector3D<double> const &point,
                             Vector3D<double> const &dir,
                             VPlacedVolume const &volume);

  static void FillBiasedDirections(VPlacedVolume const &volume,
                                   SOA3D<double> const &points,
                                   const double bias,
                                   SOA3D<double> *const dirs);

  static void FillBiasedDirections(LogicalVolume const &volume,
                                   SOA3D<double> const &points,
                                   const double bias,
                                   SOA3D<double> *const dirs);

  static void FillUncontainedPoints(VPlacedVolume const &volume,
                                    SOA3D<double> *const points);

  static void FillUncontainedPoints(LogicalVolume const &volume,
                                    SOA3D<double> *const points);

private:
    
  void GenerateVolumePointers(VPlacedVolume const *const vol);

  ShapeBenchmark GenerateBenchmark(const double elapsed,
                                   const BenchmarkType type) const;

  ShapeBenchmark RunSpecialized(double *distances) const;

  ShapeBenchmark RunUnspecialized(double *distances) const;

  ShapeBenchmark RunUSolids(double *distances) const;

  ShapeBenchmark RunROOT(double *distances) const;

  void PrepareBenchmark();

  double* AllocateDistance() const {
    return (double*) _mm_malloc(n_points_*pool_multiplier_*sizeof(double),
                                kAlignmentBoundary);
  }

  static void FreeDistance(double *distance) {
    _mm_free(distance);
  }

};

VECGEOM_INLINE
bool ShapeTester::IsFacingVolume(Vector3D<double> const &point,
                                 Vector3D<double> const &dir,
                                 VPlacedVolume const &volume) {
  return volume.DistanceToIn(point, dir, kInfinity) < kInfinity;
}

} // End namespace vecgeom

#endif // VECGEOM_COMPARISON_SHAPETESTER_H_