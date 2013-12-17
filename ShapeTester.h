#ifndef SHAPE_TESTER_H
#define SHAPE_TESTER_H

#include <vector>
#include <iostream>
#include "GeoManager.h"

enum BenchmarkType {kPlaced, kUnplaced, kUSolids, kROOT};

typedef struct {
  const double elapsed;
  const BenchmarkType type;
  static const char * const benchmark_labels[];
  const unsigned repetitions;
  const unsigned volumes;
  const unsigned points;
  const double bias;
  void Print() const {
    std::cout << elapsed << "s | "
              << volumes << " " << benchmark_labels[type] << " volumes, "
              << points << " points, " << bias << " bias, repeated "
              << repetitions << " times.\n";
  }
} ShapeBenchmark;


typedef struct {
  PhysicalVolume const *fastgeom;
  VUSolid const *usolids;
  TGeoShape const *root;
} VolumePointers;


class ShapeTester {

private:

  PhysicalVolume const *world = nullptr;
  std::vector<VolumePointers> volumes;
  unsigned n_vols = 0;
  unsigned n_points = 1<<10;
  unsigned reps = 1e3;
  double bias = 0.8;
  unsigned pool_multiplier = 1;
  std::vector<ShapeBenchmark> results;
  bool verbose = false;
  Vectors3DSOA point_pool, dir_pool;
  double *steps = nullptr;

public:

  void BenchmarkAll();
  void BenchmarkPlaced();
  void BenchmarkUnplaced();
  void BenchmarkUSolids();
  void BenchmarkROOT();

  ShapeBenchmark PopResult();
  std::vector<ShapeBenchmark> PopResults();
  
  ShapeTester() {}

  ShapeTester(PhysicalVolume const *world_) {
    SetWorld(world_);
  }
  
  ~ShapeTester();

  void SetWorld(PhysicalVolume const *world_) {
    world = world_;
  }

  void SetNumberOfPoints(const unsigned n_points_) {
    n_points = n_points_;
  }

  void SetRepetitions(const unsigned reps_) {
    reps = reps_;
  }

  void SetBias(const double bias_) {
    bias = bias_;
  }

  void SetPoolMultiplier(const unsigned /*pool_multiplier_*/);

  void SetVerbose(const bool verbose_) {
    verbose = verbose_;
  }

  std::vector<ShapeBenchmark> Results() const {
    return results;
  }

private:
    
  void GenerateVolumePointers(PhysicalVolume const* /*vol*/);

  ShapeBenchmark GenerateBenchmark(const double /*elapsed*/,
                                   const BenchmarkType /*type*/) const;

  ShapeBenchmark RunPlaced(double* /*distances*/) const;

  ShapeBenchmark RunUnplaced(double* /*distances*/) const;

  ShapeBenchmark RunUSolids(double* /*distances*/) const;

  ShapeBenchmark RunROOT(double* /*distances*/) const;

  void PrepareBenchmark();

  double* AllocateDistance() {
    return (double*) _mm_malloc(n_points*pool_multiplier*sizeof(double),
                                ALIGNMENT_BOUNDARY);
  }

  void FreeDistance(double *distance) {
    _mm_free(distance);
  }

};

#endif /* SHAPE_TESTER_H */