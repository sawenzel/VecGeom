//===-- VecGeom/volumes/utilities/AABBSubdivider.h ------------------*- C++ -*-===//
//
/// @file   AABBSubdivider.h
/// @brief  Split a placed volume's bounding box into several tighter sub-boxes.
///
/// A single axis-aligned bounding box around an elongated, rotated or
/// diagonally-placed solid encloses a lot of empty space, so a BVH built from
/// such boxes wastes work on rays that hit the box but miss the solid.
/// AABBSubdivider replaces that one loose box by a small collection of tighter
/// "slab" boxes (in the mother frame) whose union still conservatively covers
/// the solid but contains much less dead space, letting the BVH prune more rays
/// and visit fewer candidate primitives.
///
/// Main entry points (class AABBSubdivider):
///   Subdivide(placed, M)   -> exactly M slabs along the best axis.
///   SubdivideAuto(placed)  -> chooses M automatically (surface-area heuristic)
///                             and returns the slabs plus diagnostics.
/// Both return boxes in the mother frame; M = 1 gives the plain Extent() AABB.
///
/// Initial version June 2026; sandro.wenzel@cern.ch
///
//===--------------------------------------------------------------------------===//

#pragma once

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"

#include <vector>
#include <unordered_map>
#include <mutex>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <string>

namespace vecgeom {
inline namespace cxx {

// ─────────────────────────────────────────────────────────────────────────────
//  Result type
// ─────────────────────────────────────────────────────────────────────────────

/// One tight axis-aligned bounding box in the MOTHER frame.
struct SlabBox {
  Vector3D<Precision> min{};
  Vector3D<Precision> max{};
  int slabIndex{0}; ///< 0-based within the M slabs

  bool Valid() const { return min[0] <= max[0] && min[1] <= max[1] && min[2] <= max[2]; }
  Precision SurfaceArea() const
  {
    if (!Valid()) return Precision(0);
    auto d = max - min;
    return Precision(2) * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0]);
  }
  Vector3D<Precision> HalfExtents() const { return (max - min) * Precision(0.5); }
  Vector3D<Precision> Center() const { return (max + min) * Precision(0.5); }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────────────────────────────────────

struct AABBSubdividerConfig {
  // ── Grid / sampling ───────────────────────────────────────────────────────

  /// Number of surface points sampled per unplaced volume (cached).
  /// Used for axis-variance ranking and axis selection.
  int nSurfaceSamples = 4000;

  /// Resolution of the 3-D Inside() probe grid along each axis.
  /// Total Inside() calls per slab ≈ gridRes³ / M (slab rejection is cheap).
  int gridRes = 48;

  /// Absolute safety margin added to every face of each tight box [length units].
  Precision safetyMargin = Precision(1e-4);

  // ── Auto-M convergence (used only when M = 0 is passed to Subdivide) ─────

  /// Hard upper limit on M in auto mode.
  int autoMaxM = 20;

  /// Stop when  (SA(M-1) - SA(M)) / SA(1)  <  saRelThreshold.
  /// Relative to the M=1 (global) SA so it is scale-independent.
  /// Default 2 %: each new slab must reduce total SA by at least 2 % of
  /// the original global box SA to be worth keeping.
  Precision saRelThreshold = Precision(0.02);

  /// Stop when  SA(M-1) - SA(M)  <  saAbsThreshold.
  /// Catches the axis-aligned case where the global box is already tight:
  /// splitting gains essentially zero and the first split already fails here.
  /// Set to 0 to disable (rely on relative threshold only).
  Precision saAbsThreshold = Precision(1e-6);
};

// ─────────────────────────────────────────────────────────────────────────────
//  Diagnostic result (returned by SubdivideAuto for inspection / logging)
// ─────────────────────────────────────────────────────────────────────────────

struct AABBSubdivisionResult {
  std::vector<SlabBox> slabs; ///< final tight boxes in mother frame
  int chosenM{1};             ///< M that was selected
  int chosenAxis{0};          ///< 0=X, 1=Y, 2=Z
  Precision globalSA{0};      ///< SA of the M=1 global AABB
  Precision finalSA{0};       ///< total SA of chosen slabs
  /// Per-step SA values indexed by M-1 (entry 0 = global SA at M=1,
  /// entry k = total SA at M=k+1).  Useful for plotting convergence.
  std::vector<Precision> saHistory;
};

// ─────────────────────────────────────────────────────────────────────────────
//  AABBSubdivider
// ─────────────────────────────────────────────────────────────────────────────

class AABBSubdivider {
public:
  // ── Singleton (callers may also construct their own instance) ─────────────
  static AABBSubdivider &Instance()
  {
    static AABBSubdivider inst;
    return inst;
  }

  // ── Fixed-M entry point ───────────────────────────────────────────────────

  /// Subdivide @p placed into exactly @p M tight AABBs.
  ///
  /// @param placed  The placed volume.
  /// @param M       Number of slabs, must be >= 1.
  ///                M = 1  →  returns the plain transformed-Extent() AABB
  ///                          (no grid probe, exact match with BVH baseline).
  /// @param cfg     Tuning knobs.
  /// @returns       Vector of SlabBox in the mother frame; size in [1, M].
  std::vector<SlabBox> Subdivide(VPlacedVolume const *placed, int M, AABBSubdividerConfig const &cfg = {}) const;

  // ── Auto-M entry point ────────────────────────────────────────────────────

  /// Automatically determine M using the SAH convergence criterion and return
  /// full diagnostics.
  ///
  /// Iterates M = 1, 2, … until improvement per step falls below threshold
  /// or autoMaxM is reached.  Returns slabs for the winning M together with
  /// convergence history for logging / visualisation.
  ///
  /// @param placed  The placed volume.
  /// @param cfg     Tuning knobs (saRelThreshold, saAbsThreshold, autoMaxM).
  AABBSubdivisionResult SubdivideAuto(VPlacedVolume const *placed, AABBSubdividerConfig const &cfg = {}) const;

  // ── Surface-point cache ───────────────────────────────────────────────────

  void WarmCache(VUnplacedVolume const *unplaced, int nSamples = 4000);
  void ClearCache();
  std::size_t CacheSize() const;

private:
  mutable std::mutex fMutex;
  mutable std::unordered_map<VUnplacedVolume const *, std::vector<Vector3D<Precision>>> fSurfaceCache;

  // ── Internal helpers ──────────────────────────────────────────────────────

  std::vector<Vector3D<Precision>> const &GetSurfacePoints(VUnplacedVolume const *unplaced, int nSamples) const;

  static SlabBox ComputeGlobalAABB(VUnplacedVolume const *unplaced, Transformation3D const *tr);

  static SlabBox ComputeOneSlabAABB(VUnplacedVolume const *unplaced, Transformation3D const *tr, int axisIdx,
                                    Precision lo, Precision hi, SlabBox const &globalBox,
                                    std::vector<Vector3D<Precision>> const &motherSurfPts, int gridRes,
                                    Precision safetyMargin, int slabIndex);

  static std::vector<SlabBox> ComputeSlabs(VUnplacedVolume const *unplaced, Transformation3D const *tr, int axisIdx,
                                           int M, SlabBox const &globalBox,
                                           std::vector<Vector3D<Precision>> const &motherSurfPts,
                                           AABBSubdividerConfig const &cfg);

  /// Rank the three world axes by descending surface-point variance.
  /// Returns {best, second, third} axis indices.
  static void RankAxes(std::vector<Vector3D<Precision>> const &motherPts, int outOrder[3]);

  /// Select the best axis for a given M by running ComputeSlabs on all
  /// three axes (with early exit) and returning the axis + slabs with
  /// minimum total SA.
  static std::vector<SlabBox> BestAxisSlabs(VUnplacedVolume const *unplaced, Transformation3D const *tr,
                                            int const axisOrder[3], Precision const variance[3], int M,
                                            SlabBox const &globalBox,
                                            std::vector<Vector3D<Precision>> const &motherSurfPts,
                                            AABBSubdividerConfig const &cfg, int &outBestAxis);
};

// ══════════════════════════════════════════════════════════════════════════════
//  Inline implementation
// ══════════════════════════════════════════════════════════════════════════════

// ── Cache ─────────────────────────────────────────────────────────────────────

inline void AABBSubdivider::ClearCache()
{
  std::lock_guard<std::mutex> lk(fMutex);
  fSurfaceCache.clear();
}

inline std::size_t AABBSubdivider::CacheSize() const
{
  std::lock_guard<std::mutex> lk(fMutex);
  return fSurfaceCache.size();
}

inline void AABBSubdivider::WarmCache(VUnplacedVolume const *unplaced, int nSamples)
{
  GetSurfacePoints(unplaced, nSamples);
}

inline std::vector<Vector3D<Precision>> const &AABBSubdivider::GetSurfacePoints(VUnplacedVolume const *unplaced,
                                                                                int nSamples) const
{
  {
    std::lock_guard<std::mutex> lk(fMutex);
    auto it = fSurfaceCache.find(unplaced);
    if (it != fSurfaceCache.end()) return it->second;
  }

  // Sample outside the lock – SamplePointOnSurface is const / re-entrant.
  std::vector<Vector3D<Precision>> pts;
  pts.reserve(nSamples);
  for (int i = 0; i < nSamples; ++i)
    pts.push_back(unplaced->SamplePointOnSurface());

  std::lock_guard<std::mutex> lk(fMutex);
  // Another thread may have beaten us – keep whichever got there first.
  auto [it, inserted] = fSurfaceCache.emplace(unplaced, std::move(pts));
  return it->second;
}

// ── M=1: exact transformed-Extent() AABB (BVH baseline) ──────────────────────

inline SlabBox AABBSubdivider::ComputeGlobalAABB(VUnplacedVolume const *unplaced, Transformation3D const *tr)
{
  Vector3D<Precision> eMin, eMax;
  unplaced->Extent(eMin, eMax);

  constexpr Precision kBig = std::numeric_limits<Precision>::max();
  SlabBox box;
  box.min       = {kBig, kBig, kBig};
  box.max       = {-kBig, -kBig, -kBig};
  box.slabIndex = 0;

  for (int ix = 0; ix < 2; ++ix)
    for (int iy = 0; iy < 2; ++iy)
      for (int iz = 0; iz < 2; ++iz) {
        Vector3D<Precision> local(ix ? eMax[0] : eMin[0], iy ? eMax[1] : eMin[1], iz ? eMax[2] : eMin[2]);
        Vector3D<Precision> mother;
        tr->InverseTransform(local, mother); // local → mother
        for (int i = 0; i < 3; ++i) {
          box.min[i] = std::min(box.min[i], mother[i]);
          box.max[i] = std::max(box.max[i], mother[i]);
        }
      }
  return box;
}

// ── Tight AABB for one slab via 3-D Inside() grid + surface-point expansion ───
//
// Correctness guarantee
// ─────────────────────
// The grid probe alone can miss the solid surface near a slab boundary when
// the slab is thin relative to the grid step (gridStep ~ globalExtent/gridRes,
// slabWidth = globalExtent/M, so only gridRes/M planes fall inside each slab).
// We therefore apply two additional expansions after the grid probe:
//
//  (1) Surface-point expansion
//      Every cached mother-frame surface point that falls inside this slab is
//      included in the AABB.  Surface points are distributed over the actual
//      surface, so they naturally catch boundary regions the grid misses.
//
//  (2) Slab-boundary floor/ceiling along the slab axis
//      If the solid is present anywhere in this slab (result.Valid()), the box
//      must reach all the way to lo and hi along the slab axis.  The solid is a
//      closed body; if it has material in the slab interior it also has surface
//      at (or arbitrarily close to) the slab boundary planes.  Expanding to
//      [lo, hi] is therefore conservative by a solid geometric argument and
//      directly prevents the "box too small at boundary → ray misses → wrong
//      distance checksum" failure that appears when M is increased.

inline SlabBox AABBSubdivider::ComputeOneSlabAABB(VUnplacedVolume const *unplaced, Transformation3D const *tr,
                                                  int axisIdx, Precision lo, Precision hi, SlabBox const &globalBox,
                                                  std::vector<Vector3D<Precision>> const &motherSurfPts, int gridRes,
                                                  Precision safetyMargin, int slabIndex)
{
  constexpr Precision kBig = std::numeric_limits<Precision>::max();
  SlabBox result;
  result.min       = {kBig, kBig, kBig};
  result.max       = {-kBig, -kBig, -kBig};
  result.slabIndex = slabIndex;

  // ── Phase 1: uniform 3-D grid probe ──────────────────────────────────────
  //
  // Grid is sized to the full global AABB so transverse resolution is the
  // same for all slabs regardless of M.  The slab-axis rejection is hoisted
  // to the earliest (outermost matching) loop to minimise iterations.

  const Precision dx = (globalBox.max[0] - globalBox.min[0]) / gridRes;
  const Precision dy = (globalBox.max[1] - globalBox.min[1]) / gridRes;
  const Precision dz = (globalBox.max[2] - globalBox.min[2]) / gridRes;

  // Tiny epsilon so points exactly on a slab boundary are not discarded.
  const Precision eps = (hi - lo) * Precision(1e-6);

  for (int ix = 0; ix <= gridRes; ++ix) {
    const Precision px = globalBox.min[0] + ix * dx;
    if (axisIdx == 0 && (px < lo - eps || px > hi + eps)) continue;

    for (int iy = 0; iy <= gridRes; ++iy) {
      const Precision py = globalBox.min[1] + iy * dy;
      if (axisIdx == 1 && (py < lo - eps || py > hi + eps)) continue;

      for (int iz = 0; iz <= gridRes; ++iz) {
        const Precision pz = globalBox.min[2] + iz * dz;
        if (axisIdx == 2 && (pz < lo - eps || pz > hi + eps)) continue;

        Vector3D<Precision> local;
        tr->Transform(Vector3D<Precision>(px, py, pz), local); // mother→local

        if (unplaced->Inside(local) == vecgeom::kOutside) continue;

        result.min[0] = std::min(result.min[0], px);
        result.min[1] = std::min(result.min[1], py);
        result.min[2] = std::min(result.min[2], pz);
        result.max[0] = std::max(result.max[0], px);
        result.max[1] = std::max(result.max[1], py);
        result.max[2] = std::max(result.max[2], pz);
      }
    }
  }

  // ── Phase 2: surface-point expansion ─────────────────────────────────────
  //
  // Expand using every cached surface point that falls inside this slab.
  // These points lie on the actual solid boundary, so they catch regions
  // the grid misses — especially near slab boundary planes where grid
  // planes are spaced by gridStep >> slabWidth/M.

  for (auto const &mp : motherSurfPts) {
    const Precision proj = mp[axisIdx];
    if (proj < lo - eps || proj > hi + eps) continue;

    result.min[0] = std::min(result.min[0], mp[0]);
    result.min[1] = std::min(result.min[1], mp[1]);
    result.min[2] = std::min(result.min[2], mp[2]);
    result.max[0] = std::max(result.max[0], mp[0]);
    result.max[1] = std::max(result.max[1], mp[1]);
    result.max[2] = std::max(result.max[2], mp[2]);
  }

  // ── Phase 3: slab-boundary floor/ceiling ─────────────────────────────────
  //
  // If the solid is present anywhere in this slab, the box must span the
  // full [lo, hi] interval along the slab axis.  A connected solid with
  // material in the slab interior has surface at (or touching) the boundary
  // planes, so expanding to [lo, hi] is conservative by geometric argument.
  //
  // This is the primary fix for the "checksum shrinks as M increases" bug:
  // without it, a slab whose grid/surface samples happen to lie slightly
  // inside the boundary produces a box that is too short along the slab
  // axis, leaving a gap through which rays escape undetected.

  if (result.Valid()) {
    result.min[axisIdx] = std::min(result.min[axisIdx], lo);
    result.max[axisIdx] = std::max(result.max[axisIdx], hi);

    // ── Transverse conservativeness padding ──────────────────────────────
    //
    // The transverse extent of the box comes purely from discrete sampling
    // (the Inside() grid and the random surface points).  Between two grid
    // planes the true surface may protrude by up to one full grid cell
    // beyond the outermost *inside* sample, so a box clamped to the samples
    // is too small transversely and rays grazing the solid in that sliver
    // are pruned (lower distance checksum).  We therefore pad each
    // transverse face by one grid cell, which conservatively covers the
    // worst-case discretisation gap.  The slab axis already spans [lo, hi]
    // exactly, so it only needs the fixed safety margin.
    const Vector3D<Precision> cell(dx, dy, dz);
    Vector3D<Precision> margin(safetyMargin, safetyMargin, safetyMargin);
    for (int i = 0; i < 3; ++i)
      if (i != axisIdx) margin[i] += cell[i];

    result.min -= margin;
    result.max += margin;
  }
  return result;
}

// ── All slabs for one axis ────────────────────────────────────────────────────

inline std::vector<SlabBox> AABBSubdivider::ComputeSlabs(VUnplacedVolume const *unplaced, Transformation3D const *tr,
                                                         int axisIdx, int M, SlabBox const &globalBox,
                                                         std::vector<Vector3D<Precision>> const &motherSurfPts,
                                                         AABBSubdividerConfig const &cfg)
{
  const Precision lo0   = globalBox.min[axisIdx];
  const Precision hi0   = globalBox.max[axisIdx];
  const Precision slabW = (hi0 - lo0) / M;

  std::vector<SlabBox> slabs;
  slabs.reserve(M);

  for (int k = 0; k < M; ++k) {
    SlabBox sb = ComputeOneSlabAABB(unplaced, tr, axisIdx, lo0 + k * slabW, lo0 + (k + 1) * slabW, globalBox,
                                    motherSurfPts, cfg.gridRes, cfg.safetyMargin, k);
    if (sb.Valid()) slabs.push_back(sb);
  }
  return slabs;
}

// ── Axis ranking by surface-point variance ────────────────────────────────────

inline void AABBSubdivider::RankAxes(std::vector<Vector3D<Precision>> const &motherPts, int outOrder[3])
{
  Vector3D<Precision> mean(0, 0, 0);
  for (auto const &p : motherPts)
    mean += p;
  mean *= Precision(1) / Precision(motherPts.size());

  Precision var[3] = {0, 0, 0};
  for (auto const &p : motherPts)
    for (int i = 0; i < 3; ++i)
      var[i] += (p[i] - mean[i]) * (p[i] - mean[i]);

  outOrder[0] = 0;
  outOrder[1] = 1;
  outOrder[2] = 2;
  std::sort(outOrder, outOrder + 3, [&](int a, int b) { return var[a] > var[b]; });

  // Expose variances through the same array trick: store in a local copy
  // captured by value below – caller only needs the order, not the values,
  // but BestAxisSlabs needs the values for early-exit logic, so we compute
  // them there independently (cheap, O(N) over cached points).
}

// ── Best axis for a given M ───────────────────────────────────────────────────

inline std::vector<SlabBox> AABBSubdivider::BestAxisSlabs(VUnplacedVolume const *unplaced, Transformation3D const *tr,
                                                          int const axisOrder[3], Precision const variance[3], int M,
                                                          SlabBox const &globalBox,
                                                          std::vector<Vector3D<Precision>> const &motherSurfPts,
                                                          AABBSubdividerConfig const &cfg, int &outBestAxis)
{
  std::vector<SlabBox> bestSlabs;
  Precision bestSA = std::numeric_limits<Precision>::max();
  outBestAxis      = axisOrder[0];

  for (int ai = 0; ai < 3; ++ai) {
    const int axis = axisOrder[ai];

    // Skip degenerate extents
    if (globalBox.max[axis] - globalBox.min[axis] < Precision(1e-10)) continue;

    auto slabs = ComputeSlabs(unplaced, tr, axis, M, globalBox, motherSurfPts, cfg);

    Precision totalSA = 0;
    for (auto const &sb : slabs)
      totalSA += sb.SurfaceArea();

    if (totalSA < bestSA) {
      bestSA      = totalSA;
      outBestAxis = axis;
      bestSlabs   = std::move(slabs);
    }

    // Early exit: the leading axis has variance >> all others, very
    // unlikely any other axis wins.
    if (ai == 0 && variance[axisOrder[1]] > Precision(0) &&
        variance[axisOrder[0]] > Precision(4) * variance[axisOrder[1]])
      break;
  }
  return bestSlabs;
}

// ── Fixed-M Subdivide ─────────────────────────────────────────────────────────

inline std::vector<SlabBox> AABBSubdivider::Subdivide(VPlacedVolume const *placed, int M,
                                                      AABBSubdividerConfig const &cfg) const
{
  assert(placed && "AABBSubdivider::Subdivide: null placed volume");
  assert(M >= 1 && "AABBSubdivider::Subdivide: M must be >= 1");

  VUnplacedVolume const *unplaced = placed->GetLogicalVolume()->GetUnplacedVolume();
  Transformation3D const *tr      = placed->GetTransformation();

  // M=1: no grid probe needed; return exact BVH-baseline AABB.
  if (M == 1) return {ComputeGlobalAABB(unplaced, tr)};

  SlabBox globalBox = ComputeGlobalAABB(unplaced, tr);

  // Rank axes using cached surface points transformed to mother frame.
  auto const &localPts = GetSurfacePoints(unplaced, cfg.nSurfaceSamples);
  std::vector<Vector3D<Precision>> motherPts;
  motherPts.reserve(localPts.size());
  for (auto const &lp : localPts) {
    Vector3D<Precision> mp;
    tr->InverseTransform(lp, mp);
    motherPts.push_back(mp);
  }

  // Compute variance per axis (needed for early-exit in BestAxisSlabs).
  Vector3D<Precision> mean(0, 0, 0);
  for (auto const &p : motherPts)
    mean += p;
  mean *= Precision(1) / Precision(motherPts.size());
  Precision variance[3] = {0, 0, 0};
  for (auto const &p : motherPts)
    for (int i = 0; i < 3; ++i)
      variance[i] += (p[i] - mean[i]) * (p[i] - mean[i]);

  int axisOrder[3] = {0, 1, 2};
  std::sort(axisOrder, axisOrder + 3, [&](int a, int b) { return variance[a] > variance[b]; });

  int dummy;
  return BestAxisSlabs(unplaced, tr, axisOrder, variance, M, globalBox, motherPts, cfg, dummy);
}

// ── Auto-M SubdivideAuto ──────────────────────────────────────────────────────

inline AABBSubdivisionResult AABBSubdivider::SubdivideAuto(VPlacedVolume const *placed,
                                                           AABBSubdividerConfig const &cfg) const
{
  assert(placed && "AABBSubdivider::SubdivideAuto: null placed volume");

  VUnplacedVolume const *unplaced = placed->GetLogicalVolume()->GetUnplacedVolume();
  Transformation3D const *tr      = placed->GetTransformation();

  AABBSubdivisionResult res;

  // ── M=1 baseline ─────────────────────────────────────────────────────────
  SlabBox globalBox = ComputeGlobalAABB(unplaced, tr);
  res.globalSA      = globalBox.SurfaceArea();
  res.saHistory.push_back(res.globalSA); // index 0 = M=1

  // Degenerate: zero-SA global box (flat / point-like solid).
  if (res.globalSA < std::numeric_limits<Precision>::epsilon()) {
    res.slabs      = {globalBox};
    res.chosenM    = 1;
    res.chosenAxis = 0;
    res.finalSA    = res.globalSA;
    return res;
  }

  // ── Prepare surface points and axis ranking (done once) ───────────────────
  auto const &localPts = GetSurfacePoints(unplaced, cfg.nSurfaceSamples);
  std::vector<Vector3D<Precision>> motherPts;
  motherPts.reserve(localPts.size());
  for (auto const &lp : localPts) {
    Vector3D<Precision> mp;
    tr->InverseTransform(lp, mp);
    motherPts.push_back(mp);
  }

  Vector3D<Precision> mean(0, 0, 0);
  for (auto const &p : motherPts)
    mean += p;
  mean *= Precision(1) / Precision(motherPts.size());
  Precision variance[3] = {0, 0, 0};
  for (auto const &p : motherPts)
    for (int i = 0; i < 3; ++i)
      variance[i] += (p[i] - mean[i]) * (p[i] - mean[i]);

  int axisOrder[3] = {0, 1, 2};
  std::sort(axisOrder, axisOrder + 3, [&](int a, int b) { return variance[a] > variance[b]; });

  // ── Iterative M search ────────────────────────────────────────────────────
  //
  // We keep the last accepted result so we can return it when the current M
  // fails the threshold.
  //
  // Start: bestSlabs = {globalBox}, bestSA = globalSA, bestM = 1.

  std::vector<SlabBox> bestSlabs = {globalBox};
  Precision prevSA               = res.globalSA; // SA of the previous M
  int bestAxis                   = axisOrder[0];
  res.chosenM                    = 1;
  res.chosenAxis                 = axisOrder[0];

  for (int M = 2; M <= cfg.autoMaxM; ++M) {
    int candidateAxis;
    auto candidateSlabs = BestAxisSlabs(unplaced, tr, axisOrder, variance, M, globalBox, motherPts, cfg, candidateAxis);

    Precision candidateSA = 0;
    for (auto const &sb : candidateSlabs)
      candidateSA += sb.SurfaceArea();

    res.saHistory.push_back(candidateSA); // index M-1

    const Precision absImprovement = prevSA - candidateSA;
    const Precision relImprovement = absImprovement / res.globalSA;

    // ── Stopping criteria ────────────────────────────────────────────────
    //
    // (a) Relative improvement per step fell below threshold:
    //     each additional slab must reduce total SA by at least
    //     saRelThreshold × SA(M=1).
    //
    // (b) Absolute improvement fell below threshold:
    //     catches axis-aligned volumes where the global box is already
    //     tight and splitting gains essentially nothing regardless of M.
    //
    // Either condition alone is sufficient to stop.  We stop BEFORE
    // accepting this M and return the previous best.

    if (relImprovement < cfg.saRelThreshold || absImprovement < cfg.saAbsThreshold) {
      // Current M not worth it.  Return what we had before.
      break;
    }

    // Accept this M.
    prevSA         = candidateSA;
    bestSlabs      = std::move(candidateSlabs);
    bestAxis       = candidateAxis;
    res.chosenM    = M;
    res.chosenAxis = bestAxis;
  }

  res.slabs   = std::move(bestSlabs);
  res.finalSA = prevSA;
  return res;
}

} // namespace cxx
} // namespace vecgeom
