/**
 * @file benchmark.cpp
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#include "base/soa3d.h"
#include "base/rng.h"
#include "benchmarking/benchmark.h"
#include "volumes/placed_box.h"

namespace VECGEOM_NAMESPACE {

Benchmark::Benchmark(VPlacedVolume const *const world)
    : repetitions_(1e3), verbose_(0), world_(NULL) {
  set_world(world);
}

VPlacedVolume const* Benchmark::world() const {
  return world_;
}

void Benchmark::set_world(VPlacedVolume const *const world) {
  world_ = world;
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

} // End global namespace
