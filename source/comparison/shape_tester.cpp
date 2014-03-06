#include "base/iterator.h"
#include "base/soa3d.h"
#include "comparison/shape_tester.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

const char * const ShapeBenchmark::benchmark_labels[] = {
  "Specialized",
  "Placed",
  "USolids",
  "ROOT"
};

ShapeTester::ShapeTester(LogicalVolume const *const world) {
  world_ = world;
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

// ShapeBenchmark ShapeTester::RunPlaced(double* distances) const {
//   if (verbose) std::cout << "Running Placed benchmark...";
//   StopWatch timer;
//   timer.Start();
//   for (int r = 0; r < reps; ++r) {
//     const int index = (rand() % pool_multiplier) * n_points;
//     Vectors3DSOA points(point_pool, index, n_points);
//     Vectors3DSOA dirs(dir_pool, index, n_points);
//     for (int v = 0; v < n_vols; ++v) {
//       volumes[v].fastgeom->DistanceToIn(points, dirs, steps, &distances[index]);
//     }
//   }
//   timer.Stop();
//   const double elapsed = timer.getDeltaSecs();
//   if (verbose) std::cout << " Finished in " << elapsed << "s.\n";
//   return GenerateBenchmark(elapsed, kPlaced);
// }

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

void ShapeTester::PrepareBenchmark() {

  // Allocate memory
  if (steps_) _mm_free(steps_);
  delete point_pool_;
  delete dir_pool_;
  point_pool_ = new SOA3D<Precision>(n_points_ * pool_multiplier_);
  point_pool_ = new SOA3D<Precision>(n_points_ * pool_multiplier_);
  steps_ = static_cast<double*>(_mm_malloc(n_points_*sizeof(double),
                                           kAlignmentBoundary));
  for (unsigned i = 0; i < n_points_; ++i) steps_[i] = kInfinity;

  // Generate pointers to volume objects
  volumes_.clear();
  // GenerateVolumePointers(world);
  // world->fillWithRandomPoints(point_pool, n_points * pool_multiplier);
  // world->fillWithBiasedDirections(point_pool, dir_pool,
  //                                 n_points * pool_multiplier, bias);

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

} // End namespace vecgeom