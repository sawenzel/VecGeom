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
// Each of plain/auto-M/fixed-M=N is run three times: once against the
// production BVH (base/BVH.h, implicit-heap layout), once against BVH_V2
// (base/BVH_V2.h, explicit-index layout with a cost-based sweep-SAH build),
// suffixed " (v2)", and once against the vendored madmann91/bvh "v2" library
// that ROOT's TGeoTessellated.cxx uses internally for its own facet BVH
// (geom/geom/inc/bvh/v2/, see TGeoTessellated::BuildBVH), suffixed " (root)".
// All three consume the exact same slab boxes, so the comparison isolates
// the tree layout/build from the subdivision strategy.
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
#include "VecGeom/base/BVH_V2.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/volumes/utilities/AABBSubdivider.h"

// Vendored madmann91/bvh "v2" header-only library that ROOT's TGeoTessellated.cxx builds against
// (geom/geom/inc/bvh/v2/, wrapped by bvh2_third_party.h, which also silences its third-party
// warnings). Exported as a public ROOT header next to TGeoManager.h, so it resolves through the
// same ROOT_INCLUDE_DIRS already required to build this benchmark -- no extra include path needed.
#include <bvh2_third_party.h>
#include <limits>

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
template <typename BVHType>
Result BenchmarkBVH(std::string name, LogicalVolume const &volume, BVHType const &bvh,
                    std::vector<unsigned int> const &boxid_to_primid, size_t nboxes, SOA3D<double> const &points,
                    SOA3D<double> const &directions, int nrep)
{
  auto const &daughters = volume.GetDaughters();
  Result r;
  r.name   = std::move(name);
  r.nboxes = nboxes;
  r.query  = kInfLength;

  for (int rep = 0; rep < nrep; ++rep) {
    r.intersect = 0;
    r.checksum  = 0.;
    Stopwatch timer;
    timer.Start();
    for (size_t i = 0; i < points.size(); ++i) {
      const Vector3D<double> ray_point = points[i];
      const Vector3D<double> ray_dir   = directions[i];
      double hit_distance              = kInfLength;
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
      bvh.template Intersect<false>(ray_point, ray_dir, kInfLength, hook);
      if (hit_distance < kInfLength) r.checksum += hit_distance;
    }
    timer.Stop();
    r.query = std::min(r.query, timer.Elapsed());
  }
  return r;
}

// Fill `boxes`/`boxid_to_primid` with the per-daughter slab boxes for one subdivision strategy.
// `slab_provider(daughter)` returns the tight boxes for one daughter; the strategies differ only
// in this provider. Called once per strategy and shared by every BVH implementation under test
// (RunStrategy below, and BenchmarkRootBVH's caller), so each tree is built from the exact same
// boxes instead of recomputing them per implementation.
template <typename SlabProvider>
void FillSlabBoxes(LogicalVolume const &volume, SlabProvider &&slab_provider, std::vector<Vector3D<double>> &boxes,
                   std::vector<unsigned int> &boxid_to_primid)
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
}

// Build, benchmark and time one BVHType against slab boxes that were already filled by the caller
// (FillSlabBoxes is comparatively expensive -- it samples surface points and runs a SAH search --
// so every BVHType under test for a given subdivision strategy shares one set of boxes instead of
// recomputing them). `BVHType` selects the tree implementation under test (BVH<float> or
// BVH_V2<float>); both share the constructor signature (rootId, AABB corners, nChild, depth).
template <typename BVHType>
Result RunStrategy(std::string name, LogicalVolume const &volume, std::vector<Vector3D<double>> &boxes,
                   std::vector<unsigned int> const &boxid_to_primid, int bvh_depth, SOA3D<double> const &points,
                   SOA3D<double> const &directions, int nrep)
{
  Stopwatch build_timer;
  build_timer.Start();
  BVHType bvh(0, &boxes[0], boxes.size() / 2, bvh_depth);
  build_timer.Stop();
  auto r  = BenchmarkBVH(std::move(name), volume, bvh, boxid_to_primid, boxes.size() / 2, points, directions, nrep);
  r.build = build_timer.Elapsed();
  return r;
}

// Same query as BenchmarkBVH/RunStrategy above, but against a bvh::v2::Bvh built the same way
// TGeoTessellated::BuildBVH builds its facet BVH (DefaultBuilder, Quality::High) -- here over the
// same per-daughter slab boxes the other strategies use, rather than over facets. `boxes` is the
// (min,max)-pair layout produced by FillSlabBoxes; `boxid_to_primid` maps each box back to its
// daughter index exactly as for the VecGeom strategies.
Result BenchmarkRootBVH(std::string name, LogicalVolume const &volume, std::vector<Vector3D<double>> const &boxes,
                        std::vector<unsigned int> const &boxid_to_primid, SOA3D<double> const &points,
                        SOA3D<double> const &directions, int nrep)
{
  using Scalar = float;
  using BBox   = bvh::v2::BBox<Scalar, 3>;
  using Vec3   = bvh::v2::Vec<Scalar, 3>;
  using Node   = bvh::v2::Node<Scalar, 3>;
  using Bvh    = bvh::v2::Bvh<Node>;
  using Ray    = bvh::v2::Ray<Scalar, 3>;

  auto const &daughters = volume.GetDaughters();
  const size_t nprim    = boxid_to_primid.size();

  Result r;
  r.name   = std::move(name);
  r.nboxes = nprim;
  r.query  = kInfLength;

  Stopwatch build_timer;
  build_timer.Start();
  std::vector<BBox> bboxes;
  std::vector<Vec3> centers;
  bboxes.reserve(nprim);
  centers.reserve(nprim);
  for (size_t i = 0; i < nprim; ++i) {
    const Vec3 lo(static_cast<Scalar>(boxes[2 * i].x()), static_cast<Scalar>(boxes[2 * i].y()),
                  static_cast<Scalar>(boxes[2 * i].z()));
    const Vec3 hi(static_cast<Scalar>(boxes[2 * i + 1].x()), static_cast<Scalar>(boxes[2 * i + 1].y()),
                  static_cast<Scalar>(boxes[2 * i + 1].z()));
    bboxes.emplace_back(lo, hi);
    centers.push_back(bboxes.back().get_center());
  }
  typename bvh::v2::DefaultBuilder<Node>::Config config;
  config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High; // matches TGeoTessellated::BuildBVH
  Bvh bvh        = bvh::v2::DefaultBuilder<Node>::build(bboxes, centers, config);
  build_timer.Stop();
  r.build = build_timer.Elapsed();

  bvh::v2::GrowingStack<Bvh::Index> stack;

  for (int rep = 0; rep < nrep; ++rep) {
    r.intersect = 0;
    r.checksum  = 0.;
    Stopwatch timer;
    timer.Start();
    for (size_t i = 0; i < points.size(); ++i) {
      const Vector3D<double> ray_point = points[i];
      const Vector3D<double> ray_dir   = directions[i];
      double hit_distance              = kInfLength;
      int last_primid                  = -1;

      Ray ray(
          Vec3(static_cast<Scalar>(ray_point.x()), static_cast<Scalar>(ray_point.y()),
               static_cast<Scalar>(ray_point.z())),
          Vec3(static_cast<Scalar>(ray_dir.x()), static_cast<Scalar>(ray_dir.y()), static_cast<Scalar>(ray_dir.z())),
          Scalar(0.), std::numeric_limits<Scalar>::max());

      stack.clear();
      bvh.intersect<false, true>(ray, bvh.get_root().index, stack, [&](size_t begin, size_t end) {
        for (size_t slot = begin; slot < end; ++slot) {
          ++r.intersect;
          const int primid = boxid_to_primid[bvh.prim_ids[slot]];
          if (primid == last_primid) continue; // same daughter as previous leaf
          last_primid     = primid;
          const auto dist = daughters[primid]->DistanceToIn(ray_point, ray_dir);
          if (dist < hit_distance) {
            hit_distance = dist;
            ray.tmax     = static_cast<Scalar>(dist); // shrink the search so the BVH can prune
          }
        }
        return false;
      });
      if (hit_distance < kInfLength) r.checksum += hit_distance;
    }
    timer.Stop();
    r.query = std::min(r.query, timer.Elapsed());
  }
  return r;
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
    r.query  = kInfLength;
    for (int rep = 0; rep < nrep; ++rep) {
      r.intersect = 0;
      r.checksum  = 0.;
      Stopwatch timer;
      timer.Start();
      for (size_t i = 0; i < points.size(); ++i) {
        double hit_distance = kInfLength;
        for (size_t d = 0; d < daughters.size(); ++d) {
          const auto dist = daughters[d]->DistanceToIn(points[i], directions[i]);
          if (dist < hit_distance) hit_distance = dist;
        }
        r.intersect += daughters.size();
        if (hit_distance < kInfLength) r.checksum += hit_distance;
      }
      timer.Stop();
      r.query = std::min(r.query, timer.Elapsed());
    }
    results.push_back(r);
  }

  // Reusable scratch buffers owned for the lifetime of each BVH that uses them.
  std::vector<Vector3D<double>> boxes;
  std::vector<unsigned int> boxid_to_primid;

  // Fill the slab boxes for one subdivision strategy once, then benchmark all three BVH
  // implementations (production BVH, BVH_V2, ROOT's vendored bvh::v2) against that exact same set
  // of boxes, so the (comparatively expensive) box setup isn't repeated per implementation.
  auto run_all = [&](std::string base_name, auto &&slab_provider) {
    FillSlabBoxes(*volume, slab_provider, boxes, boxid_to_primid);
    results.push_back(
        RunStrategy<BVH<float>>(base_name, *volume, boxes, boxid_to_primid, bvh_depth, points, directions, nrep));
    results.push_back(RunStrategy<BVH_V2<float>>(base_name + " (v2)", *volume, boxes, boxid_to_primid, bvh_depth,
                                                 points, directions, nrep));
    results.push_back(
        BenchmarkRootBVH(base_name + " (root)", *volume, boxes, boxid_to_primid, points, directions, nrep));
  };

  // plain BVH: one box per daughter (subdivision with M = 1).
  run_all("plain", [&](VPlacedVolume const *p) { return subdivider.Subdivide(p, 1); });
  // auto-M: SAH-driven number of slabs per daughter.
  run_all("auto-M", [&](VPlacedVolume const *p) { return subdivider.SubdivideAuto(p).slabs; });
  // fixed-M: constant number of slabs per daughter.
  run_all("fixed-M=" + std::to_string(fixedM), [&](VPlacedVolume const *p) { return subdivider.Subdivide(p, fixedM); });

  // Report. Every method is checked against the brute-force reference checksum.
  const double reference = results.front().checksum;
  const double tolerance = 1e-6 * (reference != 0. ? std::abs(reference) : 1.);
  bool all_ok            = true;

  std::printf("\n(times are the fastest of %d pass%s)\n", nrep, nrep == 1 ? "" : "es");

  std::printf("%-16s %10s %12s %14s %10s %10s   %s\n", "method", "boxes", "intersect", "checksum", "build[s]",
              "query[s]", "status");

  for (auto const &r : results) {
    const bool ok = std::abs(r.checksum - reference) <= tolerance;
    all_ok &= ok;

    std::printf("%-16s %10zu %12zu %14.4f %10.5f %10.5f   %s\n", r.name.c_str(), r.nboxes, r.intersect, r.checksum,
                r.build, r.query, ok ? "OK" : "FAIL");
  }

  if (!all_ok) {
    std::cerr << "\nFAIL: a subdivided BVH disagrees with the brute-force reference"
              << " (slab boxes are under-covering some daughters)\n";
    return 1;
  }
  return 0;
}
