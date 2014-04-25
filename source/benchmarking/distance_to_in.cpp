/**
 * @file distance_to_in.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "benchmarking/distance_to_in.h"

#include <random>

#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#endif // VECGEOM_USOLIDS
#ifdef VECGEOM_ROOT
#include "TGeoBBox.h"
#endif // VECGEOM_ROOT
#include "base/iterator.h"
#include "base/soa3d.h"
#include "base/stopwatch.h"
#include "base/transformation3d.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_box.h"
#include "volumes/utilities/volume_utilities.h"

namespace vecgeom {

DistanceToInBenchmarker::DistanceToInBenchmarker(
    VPlacedVolume const *const world)
    : Benchmark(world), n_points_(1024), bias_(0.8), pool_multiplier_(1),
      point_pool_(NULL), dir_pool_(NULL), psteps_(NULL) {
   psteps_ =
      (Precision *) _mm_malloc( sizeof(Precision) * n_points_, kAlignmentBoundary );
   for(unsigned i=0;i<n_points_;++i)
   {
      psteps_[i]=kInfinity;
   }
}

DistanceToInBenchmarker::~DistanceToInBenchmarker() {
  delete point_pool_;
  delete dir_pool_;
  _mm_free(psteps_);
}

void DistanceToInBenchmarker::set_pool_multiplier(const unsigned pool_multiplier) {
  if (pool_multiplier < 1) {
    std::cerr << "Pool multiplier must be an integral number >= 1.\n";
    return;
  }
  pool_multiplier_ = pool_multiplier;
}

void DistanceToInBenchmarker::GenerateVolumePointers(
    VPlacedVolume const *const vol) {

  volumes_.emplace(volumes_.end(), vol);

  for (Iterator<Daughter> i = vol->daughters().begin(),
       i_end = vol->daughters().end(); i != i_end; ++i) {
    GenerateVolumePointers(*i);
  }

}

BenchmarkResult DistanceToInBenchmarker::GenerateBenchmarkResult(
    const Precision elapsed, const BenchmarkType type) const {
  const BenchmarkResult benchmark = {
    .elapsed = elapsed,
    .type = type,
    .repetitions = repetitions(),
    .volumes = static_cast<unsigned>(volumes_.size()),
    .points = n_points_,
    .bias = bias_
  };
  return benchmark;
}

void DistanceToInBenchmarker::PrepareBenchmark() {

  // Allocate memory
  if (point_pool_) delete point_pool_;
  if (dir_pool_) delete dir_pool_;
  point_pool_ = new SOA3D<Precision>(n_points_*pool_multiplier_);
  dir_pool_ = new SOA3D<Precision>(n_points_*pool_multiplier_);

  // Generate pointers to representations in each geometry
  volumes_.clear();
  GenerateVolumePointers(world_);

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  volumeutilities::FillUncontainedPoints(*world_, *point_pool_);
  volumeutilities::FillBiasedDirections(*world_, *point_pool_, bias_, *dir_pool_);

  point_pool_->set_size(n_points_*pool_multiplier_);
  dir_pool_->set_size(n_points_*pool_multiplier_);
}

void DistanceToInBenchmarker::BenchmarkAll() {

  if (verbose()) std::cout << "Running DistanceToIn benchmark for "
                           << n_points_ << " points for " << repetitions()
                           << " repetitions.\n";

  PrepareBenchmark();

  // Allocate output memory
  Precision *const distances_specialized = AllocateDistance();
  Precision *const distances_vec = AllocateDistance();
  Precision *const distances_unspecialized = AllocateDistance();

  // Run all benchmarks
  results_.push_back(RunSpecialized(distances_specialized));
  results_.push_back(RunSpecializedVec(distances_vec));
  results_.push_back(RunUnspecialized(distances_unspecialized));
#ifdef VECGEOM_USOLIDS
  Precision *const distances_usolids = AllocateDistance();
  results_.push_back(RunUSolids(distances_usolids));
#endif
#ifdef VECGEOM_ROOT
  Precision *const distances_root = AllocateDistance();
  results_.push_back(RunRoot(distances_root));
#endif
#ifdef VECGEOM_CUDA
  Precision *const distances_cuda = AllocateDistance();
  results_.push_back(
    GenerateBenchmarkResult(RunCuda(
      point_pool_->x(), point_pool_->y(), point_pool_->z(),
      dir_pool_->x(),   dir_pool_->y(),   dir_pool_->z(),
      distances_cuda
    ), kCuda)
  );
#endif

  // Compare results
  unsigned mismatches = 0;
  for (unsigned i = 0; i < n_points_; ++i) {
    bool mismatch = false;
    mismatch +=
        abs(distances_specialized[i] - distances_vec[i]) > kTolerance &&
        !(distances_specialized[i] == kInfinity &&
          distances_vec[i] == kInfinity);
#ifdef VECGEOM_ROOT
    mismatch +=
        abs(distances_specialized[i] - distances_root[i]) > kTolerance &&
        !(distances_specialized[i] == kInfinity &&
          distances_root[i] == 1e30);
#endif
#ifdef VECGEOM_USOLIDS
    mismatch +=
        abs(distances_specialized[i] - distances_usolids[i]) > kTolerance &&
        !(distances_specialized[i] == kInfinity &&
          distances_usolids[i] == UUtils::kInfinity);
#endif
#ifdef VECGEOM_CUDA
    mismatch +=
        abs(distances_specialized[i] - distances_cuda[i]) > kTolerance &&
        !(distances_specialized[i] == kInfinity &&
          distances_cuda[i] == kInfinity);
#endif
    mismatches += mismatch;
  }
  if (verbose()) {
    std::cout << mismatches << " / " << n_points_
              << " mismatches detected.\n";
  }

  // Clean up memory
  FreeDistance(distances_specialized);
  FreeDistance(distances_unspecialized);
#ifdef VECGEOM_USOLIDS
  FreeDistance(distances_usolids);
#endif
#ifdef VECGEOM_ROOT
  FreeDistance(distances_root);
#endif
#ifdef VECGEOM_CUDA
  FreeDistance(distances_cuda);
#endif
}

void DistanceToInBenchmarker::BenchmarkSpecialized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunSpecialized(distances));
  FreeDistance(distances);
}

void DistanceToInBenchmarker::BenchmarkSpecializedVec() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunSpecializedVec(distances));
  FreeDistance(distances);
}

void DistanceToInBenchmarker::BenchmarkUnspecialized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunUnspecialized(distances));
  FreeDistance(distances);
}

#ifdef VECGEOM_USOLIDS
void DistanceToInBenchmarker::BenchmarkUSolids() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunUSolids(distances));
  FreeDistance(distances);
}
#endif

#ifdef VECGEOM_ROOT
void DistanceToInBenchmarker::BenchmarkRoot() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunRoot(distances));
  FreeDistance(distances);
}
#endif

#ifdef VECGEOM_CUDA
void DistanceToInBenchmarker::BenchmarkCuda() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(
    GenerateBenchmarkResult(RunCuda(
      point_pool_->x(), point_pool_->y(), point_pool_->z(),
      dir_pool_->x(),   dir_pool_->y(),   dir_pool_->z(),
      distances
    ), kCuda)
  );
  FreeDistance(distances);
}
#endif

BenchmarkResult DistanceToInBenchmarker::RunSpecialized(
    Precision *const distances) const {
  if (verbose()) std::cout << "Running specialized benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions(); ++r) {
    const int index = (rand() % pool_multiplier_) * n_points_;
    for (std::vector<VolumePointers>::const_iterator d = volumes_.begin();
         d != volumes_.end(); ++d) {
      for (unsigned i = 0; i < n_points_; ++i) {
        const int p = index + i;
        distances[i] = d->specialized()->DistanceToIn(
          (*point_pool_)[p], (*dir_pool_)[p]
        );
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (verbose()) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmarkResult(elapsed, kSpecialized);
}

// QUESTION: HOW DO I GET A DIFFERENT VECTOR CONTAINER FROM THE POOL EACH TIME?
BenchmarkResult DistanceToInBenchmarker::RunSpecializedVec(Precision *const distances) const {
  if (verbose()) std::cout << "Running specialized benchmark with vector interface...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions(); ++r) {
    // const int index = (rand() % pool_multiplier_) * n_points_;
    for (std::vector<VolumePointers>::const_iterator d = volumes_.begin();
         d != volumes_.end(); ++d){
          // call vector interface
          d->specialized()->DistanceToIn(   (*point_pool_), (*dir_pool_), psteps_, distances );
    }
  }
  const Precision elapsed = timer.Stop();
  if (verbose()) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmarkResult(elapsed, kSpecializedVector);
}

BenchmarkResult DistanceToInBenchmarker::RunUnspecialized(Precision *const distances) const {
  if (verbose()) std::cout << "Running unspecialized benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions(); ++r) {
    const int index = (rand() % pool_multiplier_) * n_points_;
    for (std::vector<VolumePointers>::const_iterator d = volumes_.begin();
         d != volumes_.end(); ++d) {
      for (unsigned i = 0; i < n_points_; ++i) {
        const int p = index + i;
        distances[i] = d->unspecialized()->DistanceToIn(
          (*point_pool_)[p], (*dir_pool_)[p]
        );
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (verbose()) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmarkResult(elapsed, kUnspecialized);
}

#ifdef VECGEOM_USOLIDS
BenchmarkResult DistanceToInBenchmarker::RunUSolids(
    Precision *const distances) const {
  if (verbose()) std::cout << "Running USolids benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions(); ++r) {
    const int index = (rand() % pool_multiplier_) * n_points_;
    for (std::vector<VolumePointers>::const_iterator v = volumes_.begin();
         v != volumes_.end(); ++v) {
      Transformation3D const *transformation =
          v->unspecialized()->transformation();
      for (unsigned i = 0; i < n_points_; ++i) {
        const int p = index + i;
        const Vector3D<Precision> point =
            transformation->Transform((*point_pool_)[p]);
        const Vector3D<Precision> dir =
            transformation->TransformDirection((*dir_pool_)[p]);
        distances[i] = v->usolids()->DistanceToIn(
          UVector3(point[0], point[1], point[2]),
          UVector3(dir[0], dir[1], dir[2])
        );
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (verbose()) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmarkResult(elapsed, kUSolids);
}
#endif

#ifdef VECGEOM_ROOT
BenchmarkResult DistanceToInBenchmarker::RunRoot(
    Precision *const distances) const {
  if (verbose()) std::cout << "Running ROOT benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions(); ++r) {
    const int index = (rand() % pool_multiplier_) * n_points_;
    for (auto v = volumes_.begin(); v != volumes_.end(); ++v) {
      Transformation3D const *transformation =
          v->unspecialized()->transformation();
      for (unsigned i = 0; i < n_points_; ++i) {
        const int p = index + i;
        Vector3D<Precision> point =
            transformation->Transform((*point_pool_)[p]);
        Vector3D<Precision> dir =
            transformation->TransformDirection((*dir_pool_)[p]);
        distances[i] = v->root()->DistFromOutside(&point[0], &dir[0]);
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (verbose()) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmarkResult(elapsed, kRoot);
}
#endif

double* DistanceToInBenchmarker::AllocateDistance() const {
  return (double*) _mm_malloc(n_points_*sizeof(double), kAlignmentBoundary);
}

void DistanceToInBenchmarker::FreeDistance(double *const distance) {
  _mm_free(distance);
}

} // End namespace vecgeom