#include "UBox.hh"
#include "TGeoBBox.h"
#include "base/iterator.h"
#include "base/soa3d.h"
#include "base/stopwatch.h"
#include "base/transformation_matrix.h"
#include "comparison/shape_tester.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_box.h"
#include <random>

namespace vecgeom {

static std::mt19937 rng(0);
std::uniform_real_distribution<> uniform_dist(0,1);

std::ostream& operator<<(std::ostream &os, ShapeBenchmark const &benchmark) {
  os << benchmark.elapsed << "s | " << benchmark.volumes << " "
     << ShapeBenchmark::benchmark_labels[benchmark.type] << " volumes, "
     << benchmark.points << " points, " << benchmark.bias
     << " bias, repeated " << benchmark.repetitions << " times.";
  return os;
}

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

  volumes_.emplace(volumes_.end(), vol);

  for (Iterator<Daughter> i = vol->daughters().begin();
       i != vol->daughters().end(); ++i) {
    GenerateVolumePointers(*i);
  }

}

ShapeBenchmark ShapeTester::GenerateBenchmark(const Precision elapsed,
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

Vector3D<Precision> ShapeTester::SamplePoint(Vector3D<Precision> const &size,
                                             const Precision scale) {
  const Vector3D<Precision> ret(
    scale * (1. - 2. * uniform_dist(rng)) * size[0],
    scale * (1. - 2. * uniform_dist(rng)) * size[1],
    scale * (1. - 2. * uniform_dist(rng)) * size[2]
  );
  return ret;
}

Vector3D<Precision> ShapeTester::SampleDirection() {

  Vector3D<Precision> dir(
    (1. - 2. * uniform_dist(rng)),
    (1. - 2. * uniform_dist(rng)),
    (1. - 2. * uniform_dist(rng))
  );

  const Precision inverse_norm =
      1. / std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
  dir[0] *= inverse_norm;
  dir[1] *= inverse_norm;
  dir[2] *= inverse_norm;

  return dir;
}

void ShapeTester::FillRandomDirections(SOA3D<Precision> *const dirs) {

  const int size = dirs->size();
  for (int i = 0; i < size; ++i) {
    const Vector3D<Precision> temp = SampleDirection();
    dirs->x(i) = temp[0];
    dirs->y(i) = temp[1];
    dirs->z(i) = temp[2];
  }

}

void ShapeTester::FillBiasedDirections(VPlacedVolume const &volume,
                                       SOA3D<Precision> const &points,
                                       const Precision bias,
                                       SOA3D<Precision> *const dirs) {

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
  while (static_cast<Precision>(n_hits)/static_cast<Precision>(size) >= bias) {
    h = static_cast<int>(static_cast<Precision>(size) * uniform_dist(rng));
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
  while (static_cast<Precision>(n_hits)/static_cast<Precision>(size) < bias) {
    h = static_cast<int>(static_cast<Precision>(size) * uniform_dist(rng));
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
                                       SOA3D<Precision> const &points,
                                       const Precision bias,
                                       SOA3D<Precision> *const dirs) {
  VPlacedVolume const *const placed = volume.Place();
  FillBiasedDirections(*placed, points, bias, dirs);
  delete placed;
}

void ShapeTester::FillUncontainedPoints(VPlacedVolume const &volume,
                                        SOA3D<Precision> *const points) {
  const int size = points->size();
  const Vector3D<Precision> dim = volume.bounding_box()->dimensions();
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
  if (point_pool_) delete point_pool_;
  if (dir_pool_) delete dir_pool_;
  point_pool_ = new SOA3D<Precision>(n_points_*pool_multiplier_);
  dir_pool_ = new SOA3D<Precision>(n_points_*pool_multiplier_);

  // Generate pointers to representations in each geometry
  volumes_.clear();
  GenerateVolumePointers(world_);

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  FillUncontainedPoints(*world_, point_pool_);
  FillBiasedDirections(*world_, *point_pool_, bias_, dir_pool_);

}

void ShapeTester::BenchmarkAll() {

  PrepareBenchmark();

  // Allocate output memory
  Precision *const distances_specialized   = AllocateDistance();
  Precision *const distances_unspecialized = AllocateDistance();
  Precision *const distances_usolids       = AllocateDistance();
  Precision *const distances_root          = AllocateDistance();

  // Run all four benchmarks
  results_.push_back(RunSpecialized(distances_specialized));
  results_.push_back(RunUnspecialized(distances_unspecialized));
  results_.push_back(RunUSolids(distances_usolids));
  results_.push_back(RunRoot(distances_root));

  // Compare results
  unsigned mismatches = 0;
  const Precision tolerance = 1e-12;
  for (unsigned i = 0; i < n_points_; ++i) {
    const bool root_mismatch =
        abs(distances_specialized[i] - distances_root[i]) > tolerance &&
        !(distances_specialized[i] == kInfinity &&
          distances_root[i] == 1e30);
    const bool usolids_mismatch =
        abs(distances_specialized[i] - distances_usolids[i]) > tolerance &&
        !(distances_specialized[i] == kInfinity &&
          distances_usolids[i] == UUtils::kInfinity);
    if (root_mismatch || usolids_mismatch) {
      if (verbose_ > 1) {
        if (!mismatches) std::cout << "VecGeom / USolids / ROOT\n";
        std::cout << distances_specialized[i]  << " / "
                  << distances_usolids[i] << " / "
                  << distances_root[i]    << std::endl;
      }
      mismatches++;
    }
  }
  if (verbose_) {
    std::cout << mismatches << " / " << n_points_
              << " mismatches detected.\n";
  }

  // Clean up memory
  FreeDistance(distances_specialized);
  FreeDistance(distances_unspecialized);
  FreeDistance(distances_usolids);
  FreeDistance(distances_root);
}

void ShapeTester::BenchmarkSpecialized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunSpecialized(distances));
  FreeDistance(distances);
}

void ShapeTester::BenchmarkUnspecialized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunUnspecialized(distances));
  FreeDistance(distances);
}

void ShapeTester::BenchmarkUSolids() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunUSolids(distances));
  FreeDistance(distances);
}


void ShapeTester::BenchmarkRoot() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  results_.push_back(RunRoot(distances));
  FreeDistance(distances);
}

ShapeBenchmark ShapeTester::PopResult() {
  ShapeBenchmark result = results_.back();
  results_.pop_back();
  return result;
}

std::vector<ShapeBenchmark> ShapeTester::PopResults() {
  std::vector<ShapeBenchmark> results = results_;
  results_.clear();
  return results;
}

ShapeBenchmark ShapeTester::RunSpecialized(Precision *const distances) const {
  if (verbose_) std::cout << "Running specialized benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions_; ++r) {
    const int index = (rand() % pool_multiplier_) * n_points_;
    for (std::vector<VolumeConverter>::const_iterator d = volumes_.begin();
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
  if (verbose_) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmark(elapsed, kSpecialized);
}

ShapeBenchmark ShapeTester::RunUnspecialized(Precision *const distances) const {
  if (verbose_) std::cout << "Running unspecialized benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions_; ++r) {
    const int index = (rand() % pool_multiplier_) * n_points_;
    for (std::vector<VolumeConverter>::const_iterator d = volumes_.begin();
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
  if (verbose_) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmark(elapsed, kUnspecialized);
}

ShapeBenchmark ShapeTester::RunUSolids(Precision *const distances) const {
  if (verbose_) std::cout << "Running USolids benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions_; ++r) {
    const int index = (rand() % pool_multiplier_) * n_points_;
    for (std::vector<VolumeConverter>::const_iterator v = volumes_.begin();
         v != volumes_.end(); ++v) {
      TransformationMatrix const *matrix = v->unspecialized()->matrix();
      for (unsigned i = 0; i < n_points_; ++i) {
        const int p = index + i;
        const Vector3D<Precision> point =
            matrix->Transform<1, 0>((*point_pool_)[p]);
        const Vector3D<Precision> dir =
            matrix->TransformRotation<0>((*dir_pool_)[p]);
        distances[i] = v->usolids()->DistanceToIn(
          UVector3(point[0], point[1], point[2]),
          UVector3(dir[0], dir[1], dir[2])
        );
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (verbose_) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmark(elapsed, kUSolids);
}

ShapeBenchmark ShapeTester::RunRoot(Precision *const distances) const {
  if (verbose_) std::cout << "Running ROOT benchmark...";
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < repetitions_; ++r) {
    const int index = (rand() % pool_multiplier_) * n_points_;
    for (std::vector<VolumeConverter>::const_iterator v = volumes_.begin();
         v != volumes_.end(); ++v) {
      TransformationMatrix const *matrix = v->unspecialized()->matrix();
      for (unsigned i = 0; i < n_points_; ++i) {
        const int p = index + i;
        const Vector3D<Precision> point =
            matrix->Transform<1, 0>((*point_pool_)[p]);
        const Vector3D<Precision> dir =
            matrix->TransformRotation<0>((*dir_pool_)[p]);
        distances[i] = v->root()->DistFromOutside(&point[0], &dir[0]);
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (verbose_) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmark(elapsed, kRoot);
}

} // End namespace vecgeom