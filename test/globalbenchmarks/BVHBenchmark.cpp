//===-- test/globalbenchmarks/BVHBenchmark.cpp ---------------------*- C++ -*-===//
//
// BVHBenchmark — compare ray/daughter intersection strategies inside one volume
// ============================================================================
//
// What it measures
// ----------------
// For a single logical volume (chosen with -vol) it shoots random rays from
// inside the volume and finds, for each ray, the distance to the nearest
// daughter (DistanceToIn). The same query is answered several ways and compared:
//
//   - plain      : one bounding box per daughter (standard VecGeom ABBox/BVH).
//   - auto-M     : each daughter split into a SAH-driven number of tight slab
//                  boxes (AABBSubdivider::SubdivideAuto). More than one BVH leaf
//                  can refer to the same daughter, so the leaves hug rotated /
//                  diagonal solids more closely.
//   - fixed-M=N  : same idea, but a constant N slabs per daughter
//                  (AABBSubdivider::Subdivide, N from -fixedM).
//   - brute      : loop over all daughters, no acceleration structure. This is
//                  the reference every other method is checked against.
//
// How to read the output
// -----------------------
//   checksum  : sum of nearest-hit distances. Every method MUST equal the brute
//               reference; a smaller subdivided checksum means slab boxes are
//               under-covering the solid and the BVH wrongly culls real hits.
//               Mismatches are flagged FAIL and the program exits non-zero.
//   intersect : number of leaf-box intersection tests — the efficiency metric
//               (lower is better). Subdivision trades more/tighter boxes for
//               fewer of these tests.
//
// Usage
// -----
//   BVHBenchmark -tgeofile geom.root -vol <volname> [-npoints N] [-bvh_depth D]
//                [-fixedM M]
//
// Initial version June 2026; sandro.wenzel@cern.ch

#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "../benchmark/ArgParser.h"

#include "TGeoManager.h"
#include "VecGeomTest/RootGeoManager.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/BVH.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/volumes/utilities/AABBSubdivider.h"

using namespace vecgeom;

namespace {

// One row of the final comparison table.
struct Result {
  std::string name;
  size_t nboxes    = 0;  // BVH leaves (== ndaughters for the plain layout)
  size_t intersect = 0;  // leaf-box intersection tests
  double checksum  = 0.; // sum of nearest-hit distances
  double build     = 0.; // BVH build time [s]
  double query     = 0.; // fastest query pass over all rays [s]
};

// Run the nearest-daughter query for every ray against a subdivided BVH.
//
// The whole ray loop is repeated `nrep` times and the fastest pass is reported,
// which suppresses one-off noise (counts and checksum are identical each pass).
//
// `boxid_to_primid` maps each BVH leaf back to its daughter index (identity for
// the plain layout). When a daughter owns several leaves we skip re-evaluating
// it on consecutive leaf hits; this is deliberately consecutive-only, matching
// how a real navigator early-outs, and does not affect the result.
Result BenchmarkBVH(std::string name, LogicalVolume const &volume, BVH<float> const &bvh,
                    std::vector<unsigned int> const &boxid_to_primid, size_t nboxes, SOA3D<double> const &points,
                    SOA3D<double> const &directions, int nrep)
{
  auto const &daughters = volume.GetDaughters();
  Result r;
  r.name   = std::move(name);
  r.nboxes = nboxes;
  r.query  = kInfinity;

  for (int rep = 0; rep < nrep; ++rep) {
    r.intersect = 0;
    r.checksum  = 0.;
    Stopwatch timer;
    timer.Start();
    for (size_t i = 0; i < points.size(); ++i) {
      const Vector3D<double> ray_point = points[i];
      const Vector3D<double> ray_dir   = directions[i];
      double hit_distance              = kInfinity;
      int last_primid                  = -1;
      auto hook                        = [&](BVHIntersectContext<float> &ctx) {
        ++r.intersect;
        const int primid = boxid_to_primid[ctx.primID];
        if (primid == last_primid) return false; // same daughter as previous leaf
        last_primid     = primid;
        const auto dist = daughters[primid]->DistanceToIn(ray_point, ray_dir);
        if (dist < hit_distance) {
          hit_distance = dist;
          ctx.step_max = dist; // shrink the search so the BVH can prune
        }
        return false;
      };
      bvh.Intersect(ray_point, ray_dir, kInfinity, hook);
      if (hit_distance < kInfinity) r.checksum += hit_distance;
    }
    timer.Stop();
    r.query = std::min(r.query, timer.Elapsed());
  }
  return r;
}

// Build a BVH from per-daughter slab boxes. `slab_provider(daughter)` returns
// the tight boxes for one daughter; the strategies differ only in this provider.
template <typename SlabProvider>
BVH<float> BuildSubdividedBVH(LogicalVolume const &volume, SlabProvider &&slab_provider,
                              std::vector<Vector3D<double>> &boxes, std::vector<unsigned int> &boxid_to_primid,
                              int bvh_depth)
{
  auto const &daughters = volume.GetDaughters();
  boxes.clear();
  boxid_to_primid.clear();
  for (size_t di = 0; di < daughters.size(); ++di) {
    for (auto const &aabb : slab_provider(daughters[di])) {
      boxid_to_primid.push_back((unsigned int)di);
      boxes.push_back(aabb.min);
      boxes.push_back(aabb.max);
    }
  }
  return BVH<float>(0, &boxes[0], boxes.size() / 2, bvh_depth);
}

} // namespace

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10240);
  OPTION_STRING(tgeofile, "");
  OPTION_STRING(vol, "");
  // Fixed BVH tree depth; 0 = pick automatically from the number of boxes
  // (~<=4 boxes per leaf). Larger depth = thinner leaves (fewer boxes each).
  OPTION_INT(bvh_depth, 0);
  OPTION_INT(fixedM, 3);
  OPTION_INT(nrep, 1);

  if (tgeofile.empty() || vol.empty()) {
    std::cerr << "Usage: " << argv[0] << " -tgeofile geometry.root -vol volumename"
              << " [-npoints N] [-bvh_depth D] [-fixedM M] [-nrep N]\n";
    return 1;
  }
  if (nrep < 1) nrep = 1;

  // load geometry and select the volume to navigate in
  TGeoManager::Import(tgeofile.c_str());
  RootGeoManager::Instance().LoadRootGeometry();
  auto tgeovolume = gGeoManager->FindVolumeFast(vol.c_str());
  if (!tgeovolume) {
    std::cerr << "No TGeoVolume of name " << vol << " found\n";
    return 1;
  }
  auto volume = RootGeoManager::Instance().Convert(tgeovolume);

  std::cout << "BVH benchmark in volume " << volume->GetName() << " with " << volume->GetDaughters().size()
            << " daughters, " << npoints << " rays\n";

  // random rays starting inside the (uncontained part of the) volume
  SOA3D<double> points, directions;
  points.reserve(npoints);
  directions.reserve(npoints);
  volumeUtilities::FillUncontainedPoints(*volume, points);
  volumeUtilities::FillRandomDirections(directions);

  auto &subdivider = AABBSubdivider::Instance();
  std::vector<Result> results;

  // Reference: brute-force loop over all daughters (fastest of nrep passes).
  {
    auto const &daughters = volume->GetDaughters();
    Result r;
    r.name   = "brute";
    r.nboxes = daughters.size();
    r.query  = kInfinity;
    for (int rep = 0; rep < nrep; ++rep) {
      r.intersect = 0;
      r.checksum  = 0.;
      Stopwatch timer;
      timer.Start();
      for (size_t i = 0; i < points.size(); ++i) {
        double hit_distance = kInfinity;
        for (size_t d = 0; d < daughters.size(); ++d) {
          const auto dist = daughters[d]->DistanceToIn(points[i], directions[i]);
          if (dist < hit_distance) hit_distance = dist;
        }
        r.intersect += daughters.size();
        if (hit_distance < kInfinity) r.checksum += hit_distance;
      }
      timer.Stop();
      r.query = std::min(r.query, timer.Elapsed());
    }
    results.push_back(r);
  }

  // Reusable scratch buffers owned for the lifetime of each BVH that uses them.
  std::vector<Vector3D<double>> boxes;
  std::vector<unsigned int> boxid_to_primid;

  // Build the BVH for one subdivision strategy (timed), benchmark it, and record
  // the row. The strategies differ only in the per-daughter slab provider.
  auto run = [&](std::string name, auto &&slab_provider) {
    Stopwatch build_timer;
    build_timer.Start();
    auto bvh = BuildSubdividedBVH(*volume, slab_provider, boxes, boxid_to_primid, bvh_depth);
    build_timer.Stop();
    auto r  = BenchmarkBVH(std::move(name), *volume, bvh, boxid_to_primid, boxes.size() / 2, points, directions, nrep);
    r.build = build_timer.Elapsed();
    results.push_back(r);
  };

  // plain BVH: one box per daughter (subdivision with M = 1).
  run("plain", [&](VPlacedVolume const *p) { return subdivider.Subdivide(p, 1); });
  // auto-M: SAH-driven number of slabs per daughter.
  run("auto-M", [&](VPlacedVolume const *p) { return subdivider.SubdivideAuto(p).slabs; });
  // fixed-M: constant number of slabs per daughter.
  run("fixed-M=" + std::to_string(fixedM), [&](VPlacedVolume const *p) { return subdivider.Subdivide(p, fixedM); });

  // Report. Every method is checked against the brute-force reference checksum.
  const double reference = results.front().checksum;
  const double tolerance = 1e-6 * (reference != 0. ? std::abs(reference) : 1.);
  bool all_ok            = true;

  std::printf("\n(times are the fastest of %d pass%s)\n", nrep, nrep == 1 ? "" : "es");
  std::printf("%-12s %10s %12s %14s %10s %10s   %s\n", "method", "boxes", "intersect", "checksum", "build[s]",
              "query[s]", "status");
  for (auto const &r : results) {
    const bool ok = std::abs(r.checksum - reference) <= tolerance;
    all_ok &= ok;
    std::printf("%-12s %10zu %12zu %14.4f %10.5f %10.5f   %s\n", r.name.c_str(), r.nboxes, r.intersect, r.checksum,
                r.build, r.query, ok ? "OK" : "FAIL");
  }

  if (!all_ok) {
    std::cerr << "\nFAIL: a subdivided BVH disagrees with the brute-force reference"
              << " (slab boxes are under-covering some daughters)\n";
    return 1;
  }
  return 0;
}
