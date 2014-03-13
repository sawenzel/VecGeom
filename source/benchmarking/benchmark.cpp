#include "base/soa3d.h"
#include "base/rng.h"
#include "benchmarking/benchmark.h"
#include "volumes/placed_box.h"

namespace vecgeom {

Benchmark::Benchmark(LogicalVolume const *const world) {
  set_world(world);
}

LogicalVolume const* Benchmark::world() const {
  return world_->logical_volume();
}

void Benchmark::set_world(LogicalVolume const *const world) {
  delete world_;
  world_ = world->Place();
}

BenchmarkResult Benchmark::PopResult() {
  BenchmarkResult result = results_.back();
  results_.pop_back();
  return result;
}

std::vector<BenchmarkResult> Benchmark::PopResults() {
  std::vector<BenchmarkResult> results = results_;
  results_.clear();
  return results;
}

Vector3D<Precision> Benchmark::SamplePoint(Vector3D<Precision> const &size,
                                           const Precision scale) {
  const Vector3D<Precision> ret(
    scale * (1. - 2. * RNG::Instance().uniform()) * size[0],
    scale * (1. - 2. * RNG::Instance().uniform()) * size[1],
    scale * (1. - 2. * RNG::Instance().uniform()) * size[2]
  );
  return ret;
}

Vector3D<Precision> Benchmark::SampleDirection() {

  Vector3D<Precision> dir(
    (1. - 2. * RNG::Instance().uniform()),
    (1. - 2. * RNG::Instance().uniform()),
    (1. - 2. * RNG::Instance().uniform())
  );

  const Precision inverse_norm =
      1. / std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
  dir[0] *= inverse_norm;
  dir[1] *= inverse_norm;
  dir[2] *= inverse_norm;

  return dir;
}

void Benchmark::FillRandomDirections(TrackContainer<Precision> *const dirs) {

  const int size = dirs->size();
  for (int i = 0; i < size; ++i) {
    const Vector3D<Precision> temp = SampleDirection();
    dirs->x(i) = temp[0];
    dirs->y(i) = temp[1];
    dirs->z(i) = temp[2];
  }

}

void Benchmark::FillBiasedDirections(VPlacedVolume const &volume,
                                     TrackContainer<Precision> const &points,
                                     const Precision bias,
                                     TrackContainer<Precision> *const dirs) {

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

  // Remove hits until threshold
  while (static_cast<Precision>(n_hits)/static_cast<Precision>(size) >= bias) {
    h = static_cast<int>(
          static_cast<Precision>(size) * RNG::Instance().uniform()
        );
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
    h = static_cast<int>(
          static_cast<Precision>(size) * RNG::Instance().uniform()
        );
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

void Benchmark::FillBiasedDirections(LogicalVolume const &volume,
                                     TrackContainer<Precision> const &points,
                                     const Precision bias,
                                     TrackContainer<Precision> *const dirs) {
  VPlacedVolume const *const placed = volume.Place();
  FillBiasedDirections(*placed, points, bias, dirs);
  delete placed;
}

void Benchmark::FillUncontainedPoints(VPlacedVolume const &volume,
                                      TrackContainer<Precision> *const points) {
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

void Benchmark::FillUncontainedPoints(LogicalVolume const &volume,
                                      TrackContainer<Precision> *const points) {
  VPlacedVolume const *const placed = volume.Place();
  FillUncontainedPoints(*placed, points);
  delete placed;
}

char const *const BenchmarkResult::benchmark_labels[] = {
  "Specialized",
  "Unspecialized",
  "USolids",
  "ROOT"
};

std::ostream& operator<<(std::ostream &os, BenchmarkResult const &benchmark) {
  os << benchmark.elapsed << "s | " << benchmark.volumes << " "
     << BenchmarkResult::benchmark_labels[benchmark.type] << " volumes, "
     << benchmark.points << " points, " << benchmark.bias
     << " bias, repeated " << benchmark.repetitions << " times.";
  return os;
}

} // End namespace vecgeom