#include "base/iterator.h"
#include "base/soa3d.h"
#include "base/stopwatch.h"
#include "base/transformation_matrix.h"
#include "comparison/shape_tester.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_box.h"
#include <random>

static std::mt19937 rng(0);
std::uniform_real_distribution<> uniform_dist(0,1);

namespace vecgeom {

const char * const ShapeBenchmark::benchmark_labels[] = {
  "Specialized",
  "Unspecialized",
  "USolids",
  "ROOT"
};

ShapeTester::ShapeTester(LogicalVolume const *const world) {
  set_world(world);
}

ShapeTester::~ShapeTester() {
  _mm_free(steps_);
  delete point_pool_;
  delete dir_pool_;
}

void ShapeTester::set_pool_multiplier(const unsigned pool_multiplier) {
  if (pool_multiplier < 1) {
    std::cerr << "Pool multiplier must be an integral number >= 1.\n";
    return;
  }
  pool_multiplier_ = pool_multiplier;
}

LogicalVolume const* ShapeTester::world() const {
  return world_->logical_volume();
}

void ShapeTester::set_world(LogicalVolume const *const world) {
  delete world_;
  volumes_.clear();
  world_ = world->Place();
}

void ShapeTester::GenerateVolumePointers(VPlacedVolume const *const vol) {

  volumes_.push_back(VolumeConverter(vol));

  for (Iterator<Daughter> i = vol->logical_volume()->daughters().begin();
       i != vol->logical_volume()->daughters().end(); ++i) {
    GenerateVolumePointers(*i);
  }

}

ShapeBenchmark ShapeTester::GenerateBenchmark(const double elapsed,
                                              const BenchmarkType type) const {
  const ShapeBenchmark benchmark = {
    .elapsed = elapsed,
    .type = type,
    .repetitions = repetitions_,
    .volumes = static_cast<unsigned>(volumes_.size()),
    .points = n_points_,
    .bias = bias_
  };
  return benchmark;
}

Vector3D<double> ShapeTester::SamplePoint(Vector3D<double> const &size,
                                          const double scale) {
  const Vector3D<double> ret(
    scale * (1. - 2. * uniform_dist(rng)) * size[0],
    scale * (1. - 2. * uniform_dist(rng)) * size[1],
    scale * (1. - 2. * uniform_dist(rng)) * size[2]
  );
  return ret;
}

Vector3D<double> ShapeTester::SampleDirection() {

  Vector3D<double> dir(
    (1. - 2. * uniform_dist(rng)),
    (1. - 2. * uniform_dist(rng)),
    (1. - 2. * uniform_dist(rng))
  );

  const double inverse_norm =
      1. / std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
  dir[0] *= inverse_norm;
  dir[1] *= inverse_norm;
  dir[2] *= inverse_norm;

  return dir;
}

void ShapeTester::FillRandomDirections(SOA3D<double> *const dirs) {

  const int size = dirs->size();
  for (int i = 0; i < size; ++i) {
    const Vector3D<double> temp = SampleDirection();
    dirs->x(i) = temp[0];
    dirs->y(i) = temp[1];
    dirs->z(i) = temp[2];
  }

}

void ShapeTester::FillBiasedDirections(VPlacedVolume const &volume,
                                       SOA3D<double> const &points,
                                       const double bias,
                                       SOA3D<double> *const dirs) {

  assert(bias >= 0. && bias <= 1.);

  const int size = dirs->size();
  int n_hits = 0;
  std::vector<bool> hit(size, false);
  int h;

  // Randomize points
  FillRandomDirections(dirs);

  // Check hits
  for (int i = 0; i < size; ++i) {
    for (Iterator<Daughter> j = volume.daughters().begin();
         j != volume.daughters().end(); ++j) {
      if (IsFacingVolume(points[i], (*dirs)[i], **j)) {
        n_hits++;
        hit[i] = true;
      }
    }
  }

  // Add hits until threshold
  while (double(n_hits) / double(size) >= bias) {
    h = int(double(size) * uniform_dist(rng));
    while (hit[h]) {
      dirs->Set(h, SampleDirection());
      for (Iterator<Daughter> i = volume.daughters().begin();
           i != volume.daughters().end(); ++i) {
        if (!IsFacingVolume(points[h], (*dirs)[h], **i)) {
          n_hits--;
          hit[h] = false;
          break;
        }
      }
    }
  }


  // Add hits until threshold
  while (double(n_hits) / double(size) < bias) {
    h = int(double(size) * uniform_dist(rng));
    while (!hit[h]) {
      dirs->Set(h, SampleDirection());
      for (Iterator<Daughter> i = volume.daughters().begin();
           i != volume.daughters().end(); ++i) {
        if (IsFacingVolume(points[h], (*dirs)[h], **i)) {
          n_hits++;
          hit[h] = true;
          break;
        }
      }
    }
  }

}

void ShapeTester::FillBiasedDirections(LogicalVolume const &volume,
                                       SOA3D<double> const &points,
                                       const double bias,
                                       SOA3D<double> *const dirs) {
  VPlacedVolume const *const placed = volume.Place();
  FillBiasedDirections(*placed, points, bias, dirs);
  delete placed;
}

void ShapeTester::FillUncontainedPoints(VPlacedVolume const &volume,
                                        SOA3D<double> *const points) {
  const int size = points->size();
  const Vector3D<double> dim = volume.bounding_box()->dimensions();
  for (int i = 0; i < size; ++i) {
    bool contained;
    do {
      points->Set(i, SamplePoint(dim));
      contained = false;
      for (Iterator<Daughter> j = volume.daughters().begin();
           j != volume.daughters().end(); ++j) {
        if ((*j)->Inside((*points)[i])) {
          contained = true;
          break;
        }
      }
    } while (contained);
  }
}

void ShapeTester::FillUncontainedPoints(LogicalVolume const &volume,
                                        SOA3D<Precision> *const points) {
  VPlacedVolume const *const placed = volume.Place();
  FillUncontainedPoints(*placed, points);
  delete placed;
}

void ShapeTester::PrepareBenchmark() {

  // Allocate memory
  if (steps_) _mm_free(steps_);
  delete point_pool_;
  delete dir_pool_;
  point_pool_ = new SOA3D<Precision>(n_points_*pool_multiplier_);
  point_pool_ = new SOA3D<Precision>(n_points_*pool_multiplier_);
  steps_ = static_cast<double*>(_mm_malloc(n_points_*sizeof(double),
                                           kAlignmentBoundary));
  for (unsigned i = 0; i < n_points_; ++i) steps_[i] = kInfinity;

  // Generate pointers to representations in each geometry
  volumes_.clear();
  GenerateVolumePointers(world_);

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  FillUncontainedPoints(*world_, point_pool_);
  FillBiasedDirections(*world_, *point_pool_, bias_, dir_pool_);

}

// void ShapeTester::BenchmarkAll() {

//   PrepareBenchmark();

//   // Allocate output memory
//   double *distances_placed   = AllocateDistance();
//   double *distances_unplaced = AllocateDistance();
//   double *distances_usolids  = AllocateDistance();
//   double *distances_root     = AllocateDistance();

//   // Run all four benchmarks
//   results.push_back(RunPlaced(distances_placed));
//   results.push_back(RunUnplaced(distances_unplaced));
//   results.push_back(RunUSolids(distances_usolids));
//   results.push_back(RunROOT(distances_root));

//   // Compare results
//   unsigned mismatches = 0;
//   const double precision = 1e-12;
//   for (int i = 0; i < n_points * pool_multiplier; ++i) {
//     const bool root_mismatch =
//         abs(distances_placed[i] - distances_root[i]) > precision &&
//         !(distances_placed[i] == Utils::kInfinity && distances_root[i] == 1e30);
//     const bool usolids_mismatch =
//         abs(distances_placed[i] - distances_usolids[i]) > precision &&
//         !(distances_placed[i] == Utils::kInfinity &&
//           distances_usolids[i] == UUtils::kInfinity);
//     if (root_mismatch || usolids_mismatch) {
//       if (!mismatches) std::cout << "Placed / USolids / ROOT\n";
//       std::cout << distances_placed[i]  << " / "
//                 << distances_usolids[i] << " / "
//                 << distances_root[i]    << std::endl;
//       mismatches++;
//     }
//   }
//   if (verbose) {
//     std::cout << mismatches << " / " << n_points * pool_multiplier
//               << " mismatches detected.\n";
//   }

//   // Clean up memory
//   FreeDistance(distances_placed);
//   FreeDistance(distances_unplaced);
//   FreeDistance(distances_usolids);
//   FreeDistance(distances_root);
// }

// void ShapeTester::BenchmarkPlaced() {
//   PrepareBenchmark();
//   double *distances = AllocateDistance();
//   results.push_back(RunPlaced(distances));
//   FreeDistance(distances);
// }

// void ShapeTester::BenchmarkUnplaced() {
//   PrepareBenchmark();
//   double *distances = AllocateDistance();
//   results.push_back(RunUnplaced(distances));
//   FreeDistance(distances);
// }

// void ShapeTester::BenchmarkUSolids() {
//   PrepareBenchmark();
//   double *distances = AllocateDistance();
//   results.push_back(RunUnplaced(distances));
//   FreeDistance(distances);
// }


// void ShapeTester::BenchmarkROOT() {
//   PrepareBenchmark();
//   double *distances = AllocateDistance();
//   results.push_back(RunROOT(distances));
//   FreeDistance(distances);
// }

// ShapeBenchmark ShapeTester::PopResult() {
//   ShapeBenchmark result = results.back();
//   results.pop_back();
//   return result;
// }

// std::vector<ShapeBenchmark> ShapeTester::PopResults() {
//   std::vector<ShapeBenchmark> results_ = results;
//   results.clear();
//   return results_;
// }

ShapeBenchmark ShapeTester::RunSpecialized(double *const distances) const {
  if (verbose_) std::cout << "Running Placed benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions_; ++r) {
    int index = (rand() % pool_multiplier_) * n_points_;
    index = index;
    // Vectors3DSOA points(point_pool, index, n_points);
    // Vectors3DSOA dirs(dir_pool, index, n_points);
    // for (int v = 0; v < n_vols; ++v) {
    //   volumes[v].fastgeom->DistanceToIn(points, dirs, steps, &distances[index]);
    // }
  }
  timer.Stop();
  // const double elapsed = timer.getDeltaSecs();
  // if (verbose) std::cout << " Finished in " << elapsed << "s.\n";
  // return GenerateBenchmark(elapsed, kPlaced);
  return GenerateBenchmark(0, kSpecialized);
}

// ShapeBenchmark ShapeTester::RunUnplaced(double *distances) const {
//   if (verbose) std::cout << "Running Unplaced benchmark...";
//   Vectors3DSOA point_pool_transform(point_pool);
//   Vectors3DSOA dir_pool_transform(dir_pool);
//   StopWatch timer;
//   timer.Start();
//   for (int r = 0; r < reps; ++r) {
//     const int index = (rand() % pool_multiplier) * n_points;
//     Vectors3DSOA points(point_pool, index, n_points);
//     Vectors3DSOA dirs(dir_pool, index, n_points);
//     Vectors3DSOA points_transform(point_pool_transform, index, n_points);
//     Vectors3DSOA dirs_transform(dir_pool_transform, index, n_points);
//     for (int v = 0; v < n_vols; ++v) {
//       PhysicalVolume const *unplaced =
//           volumes[v].fastgeom->GetAsUnplacedVolume();
//       TransformationMatrix const *matrix = volumes[v].fastgeom->getMatrix();
//       matrix->MasterToLocal(points, points_transform);
//       matrix->MasterToLocalVec(dirs, dirs_transform);
//       unplaced->DistanceToIn(
//         points_transform, dirs_transform, steps, &distances[index]
//       );
//     }
//   }
//   timer.Stop();
//   const double elapsed = timer.getDeltaSecs();
//   if (verbose) std::cout << " Finished in " << elapsed << "s.\n";
//   return GenerateBenchmark(elapsed, kUnplaced);
// }

// ShapeBenchmark ShapeTester::RunUSolids(double* distances) const {
//   std::vector<Vector3D> point_pool_vec(n_points * pool_multiplier);
//   std::vector<Vector3D> dir_pool_vec(n_points * pool_multiplier);
//   point_pool.toStructureOfVector3D(point_pool_vec);
//   dir_pool.toStructureOfVector3D(dir_pool_vec);
//   if (verbose) std::cout << "Running USolids benchmark...";
//   StopWatch timer;
//   timer.Start();
//   for (int r = 0; r < reps; ++r) {
//     const int index = (rand() % pool_multiplier) * n_points;
//     for (int v = 0; v < n_vols; ++v) {
//       TransformationMatrix const *matrix = volumes[v].fastgeom->getMatrix();
//       for (int p = 0; p < n_points; ++p) {
//         Vector3D point_local, dir_local;
//         matrix->MasterToLocal<1,-1>(point_pool_vec[index+p], point_local);
//         matrix->MasterToLocalVec<-1>(dir_pool_vec[index+p], dir_local);
//         distances[index+p] = volumes[v].usolids->DistanceToIn(
//           reinterpret_cast<UVector3 const&>(point_local),
//           reinterpret_cast<UVector3 const&>(dir_local), steps[p]
//         );
//       }
//     }
//   }
//   timer.Stop();
//   const double elapsed = timer.getDeltaSecs();
//   if (verbose) std::cout << " Finished in " << elapsed << "s.\n";
//   return GenerateBenchmark(elapsed, kUSolids);
// }

// ShapeBenchmark ShapeTester::RunROOT(double *distances) const {
//   std::vector<Vector3D> point_pool_vec(n_points * pool_multiplier);
//   std::vector<Vector3D> dir_pool_vec(n_points * pool_multiplier);
//   point_pool.toStructureOfVector3D(point_pool_vec);
//   dir_pool.toStructureOfVector3D(dir_pool_vec);
//   if (verbose) std::cout << "Running ROOT benchmark...";
//   StopWatch timer;
//   timer.Start();
//   for (int r = 0; r < reps; ++r) {
//     const int index = (rand() % pool_multiplier) * n_points;
//     for (int v = 0; v < n_vols; ++v) {
//       TGeoMatrix const *matrix =
//           volumes[v].fastgeom->getMatrix()->GetAsTGeoMatrix();
//       for (int p = 0; p < n_points; ++p) {
//         Vector3D point_local, dir_local;
//         matrix->MasterToLocal(&point_pool_vec[index+p].x, &point_local.x);
//         matrix->MasterToLocalVect(&dir_pool_vec[index+p].x, &dir_local.x);
//         distances[index+p] =
//             volumes[v].root->DistFromOutside(&point_local.x, &dir_local.x,
//                                              3, steps[p], 0);
//       }
//     }
//   }
//   timer.Stop();
//   const double elapsed = timer.getDeltaSecs();
//   if (verbose) std::cout << " Finished in " << elapsed << "s.\n";
//   return GenerateBenchmark(elapsed, kROOT);
// }

} // End namespace vecgeom