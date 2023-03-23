#ifndef VECGEOM_SURFACE_POLYHEDRONCONVERTER_H_
#define VECGEOM_SURFACE_POLYHEDRONCONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Polyhedron.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Polyhedron
/// @tparam Real_t Precision type
/// @param upoly Polyhedron solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreatePolyhedronSurfaces(vecgeom::UnplacedPolyhedron const &upoly, int logical_id)
{
  using QuadMask_t = QuadrilateralMask<Real_t>;

  const bool use_surf_safety = false;
  int isurf;
  int isurfZlast = -1;
  LogicExpressionCPU logic; // OR logic: section0 | section1 | ... | sectionN
  auto const &poly    = upoly.GetStruct();
  size_t nplanes      = poly.fZPlanes.size();
  size_t nseg         = nplanes - 1;
  size_t sideCount    = poly.fSideCount;
  auto phiStart       = poly.fPhiStart;
  auto phiDelta       = poly.fPhiDelta;
  bool smallerPi      = phiDelta < (vecgeom::kPi - vecgeom::kTolerance);
  auto const &rMin    = poly.fRMin;
  auto const &rMax    = poly.fRMax;
  auto const &zPlanes = poly.fZPlanes;

  auto sidePhi         = phiDelta / sideCount;
  auto cosHalfDeltaPhi = vecCore::math::Cos(0.5 * sidePhi);
  Real_t conv          = 1. / cosHalfDeltaPhi;
  auto tanHalfDeltaPhi = vecCore::math::Tan(0.5 * sidePhi);

  // parameters for bottom face frames
  auto drB        = rMax[0] - rMin[0];
  auto rtransB    = 0.5 * (rMin[0] + rMax[0]);
  auto zframeB    = zPlanes[0];
  Real_t dxInnerB = rMin[0] * tanHalfDeltaPhi;
  Real_t dxOuterB = rMax[0] * tanHalfDeltaPhi;
  auto iframeBottom =
      builder::CreateFrame<Real_t>(kQuadrilateral, QuadMask_t{-dxOuterB, -0.5 * drB, dxOuterB, -0.5 * drB, dxInnerB,
                                                              0.5 * drB, -dxInnerB, 0.5 * drB});
  // parameters for top face frames
  auto drT        = rMax[nseg] - rMin[nseg];
  auto rtransT    = 0.5 * (rMin[nseg] + rMax[nseg]);
  auto zframeT    = zPlanes[nseg];
  Real_t dxInnerT = rMin[nseg] * tanHalfDeltaPhi;
  Real_t dxOuterT = rMax[nseg] * tanHalfDeltaPhi;
  auto iframeTop  = builder::CreateFrame<Real_t>(kQuadrilateral, QuadMask_t{-dxInnerT, -0.5 * drT, dxInnerT, -0.5 * drT,
                                                                           dxOuterT, 0.5 * drT, -dxOuterT, 0.5 * drT});

  // lambda to get the phi angle of the current side
  auto getPhi = [&](size_t side) {
    if (!poly.fHasPhiCutout && side == sideCount) {
      side = 0;
    }
    return vecgeom::NormalizeAngle<vecgeom::kScalar>(phiStart + side * sidePhi);
  };

  // lambda to convert a full Z segment
  auto convertZsegment = [&](size_t iseg) {
    auto dz             = zPlanes[iseg + 1] - zPlanes[iseg];
    auto realSeg        = dz > 0;
    auto drI            = rMin[iseg + 1] - rMin[iseg];
    auto drO            = rMax[iseg + 1] - rMax[iseg];
    auto dzframeI       = 0.5 * vecCore::math::Sqrt(drI * drI + dz * dz);
    auto dzframeO       = 0.5 * vecCore::math::Sqrt(drO * drO + dz * dz);
    bool hasInnerRadius = rMin[iseg] > 0 || rMin[iseg + 1] > 0;
    bool hasOuter       = realSeg || drO != 0;
    bool hasInner       = hasInnerRadius && (realSeg || drI != 0);
    bool hasPhi         = poly.fHasPhiCutout && realSeg;
    if (!hasOuter && !hasInner && !hasPhi) return;

    if (realSeg) {
      if (iseg > 0) logic.push_back(lor); // `OR` between sections
      logic.push_back(lplus);             // '(' begin section logic
    }
    // Add section Z planes as limiters
    isurf = isurfZlast;
    if (isurf < 0) {
      isurf = builder::CreateLocalSurface<Real_t>(
          builder::CreateUnplacedSurface<Real_t>(kPlanar), Frame{kNoFrame},
          builder::CreateLocalTransformation<Real_t>({0, 0, zPlanes[iseg], 0, 0, 0}), use_surf_safety);
      builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    }

    if (realSeg) {
      logic.push_back(lnot);
      logic.push_back(isurf);
      logic.push_back(land);
      isurf = builder::CreateLocalSurface<Real_t>(
          builder::CreateUnplacedSurface<Real_t>(kPlanar), Frame{kNoFrame},
          builder::CreateLocalTransformation<Real_t>({0, 0, zPlanes[iseg + 1], 0, 0, 0}), use_surf_safety);
      builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
      logic.push_back(isurf);
      isurfZlast = isurf;
    }

    auto thetaRmin     = vecCore::math::ATan2(dz, rMin[iseg + 1] - rMin[iseg]) * vecgeom::kRadToDeg;
    auto thetaRmax     = vecCore::math::ATan2(dz, rMax[iseg] - rMax[iseg + 1]) * vecgeom::kRadToDeg;
    Real_t rtransInner = 0.5 * (rMin[iseg] + rMin[iseg + 1]);
    Real_t rtransOuter = 0.5 * (rMax[iseg] + rMax[iseg + 1]);
    Real_t zframe      = 0.5 * (zPlanes[iseg] + zPlanes[iseg + 1]);
    Real_t dx1Inner    = rMin[iseg] * tanHalfDeltaPhi;
    Real_t dx2Inner    = rMin[iseg + 1] * tanHalfDeltaPhi;
    Real_t dx1Outer    = rMax[iseg] * tanHalfDeltaPhi;
    Real_t dx2Outer    = rMax[iseg + 1] * tanHalfDeltaPhi;
    auto iframeInner   = builder::CreateFrame<Real_t>(
        kQuadrilateral, QuadMask_t{-dx1Inner, -dzframeI, dx1Inner, -dzframeI, dx2Inner, dzframeI, -dx2Inner, dzframeI});
    auto iframeOuter = builder::CreateFrame<Real_t>(
        kQuadrilateral, QuadMask_t{-dx1Outer, -dzframeO, dx1Outer, -dzframeO, dx2Outer, dzframeO, -dx2Outer, dzframeO});

    // outer surfaces
    if (hasOuter) {
      for (size_t iside = 0; iside < sideCount; iside++) {
        auto phi       = getPhi(iside) + 0.5 * sidePhi; // center of each edge
        auto phiDeg    = (phi - vecgeom::kHalfPi) * vecgeom::kRadToDeg;
        Real_t dxyz[3] = {rtransOuter * vecCore::math::Cos(phi), rtransOuter * vecCore::math::Sin(phi), zframe};

        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(kPlanar), iframeOuter,
            builder::CreateLocalTransformation<Real_t>({dxyz[0], dxyz[1], dxyz[2], phiDeg + 180, thetaRmax, 0}),
            use_surf_safety);
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        if (realSeg) {
          logic.push_back(land);
          logic.push_back(isurf);
        }
      }
    }

    // inner surfaces
    if (hasInner) {
      for (size_t iside = 0; iside < sideCount; iside++) {
        auto phi    = getPhi(iside) + 0.5 * sidePhi; // center of each edge
        auto phiDeg = (phi - vecgeom::kHalfPi) * vecgeom::kRadToDeg;

        // translation components and rotation angles of inner surface (iseg, iside)
        Real_t dxyz[3] = {rtransInner * vecCore::math::Cos(phi), rtransInner * vecCore::math::Sin(phi), zframe};

        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(kPlanar), iframeInner,
            builder::CreateLocalTransformation<Real_t>({dxyz[0], dxyz[1], dxyz[2], phiDeg, thetaRmin, 0}),
            use_surf_safety);
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        if (realSeg) {
          if (iside == 0) {
            logic.push_back(land);
            logic.push_back(lplus);
          } else {
            logic.push_back(lor);
          }
          logic.push_back(isurf);
          if (iside == sideCount - 1) {
            logic.push_back(lminus);
          }
        }
      }
    }

    if (hasPhi) {
      auto phi1 = phiStart * vecgeom::kRadToDeg;
      auto phi2 = (phiStart + phiDelta) * vecgeom::kRadToDeg;
      isurf     = builder::CreateLocalSurface<Real_t>(
          builder::CreateUnplacedSurface<Real_t>(kPlanar),
          builder::CreateFrame<Real_t>(kQuadrilateral,
                                       QuadMask_t{conv * rMin[iseg], zPlanes[iseg], conv * rMax[iseg], zPlanes[iseg],
                                                  conv * rMax[iseg + 1], zPlanes[iseg + 1], conv * rMin[iseg + 1],
                                                  zPlanes[iseg + 1]}),
          builder::CreateLocalTransformation<Real_t>({0, 0, 0, phi1, 90, 0}), use_surf_safety /* && smallerPi*/);

      builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(lplus); // '('
      logic.push_back(isurf);
      isurf = builder::CreateLocalSurface<Real_t>(
          builder::CreateUnplacedSurface<Real_t>(kPlanar),
          builder::CreateFrame<Real_t>(kQuadrilateral,
                                       QuadMask_t{conv * rMin[iseg + 1], -zPlanes[iseg + 1], conv * rMax[iseg + 1],
                                                  -zPlanes[iseg + 1], conv * rMax[iseg], -zPlanes[iseg],
                                                  conv * rMin[iseg], -zPlanes[iseg]}),
          builder::CreateLocalTransformation<Real_t>({0, 0, 0, phi2, -90, 0}), use_surf_safety /* && smallerPi*/);
      builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
      logic.push_back(smallerPi ? land : lor);
      logic.push_back(isurf);
      logic.push_back(lminus); // ')'
    }

    for (size_t iside = 0; iside < sideCount; iside++) {
      auto phi    = getPhi(iside) + 0.5 * sidePhi; // center of each edge
      auto phiDeg = (phi - vecgeom::kHalfPi) * vecgeom::kRadToDeg;
      // Add bottom frames
      if (iseg == 0) {
        Real_t dxyz[3] = {rtransB * vecCore::math::Cos(phi), rtransB * vecCore::math::Sin(phi), zframeB};
        isurf          = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(kPlanar), iframeBottom,
            builder::CreateLocalTransformation<Real_t>({dxyz[0], dxyz[1], dxyz[2], phiDeg, 180, 0}), use_surf_safety);
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
      }

      // Add top frames
      if (iseg == (nseg - 1)) {
        Real_t dxyz[3] = {rtransT * vecCore::math::Cos(phi), rtransT * vecCore::math::Sin(phi), zframeT};
        isurf          = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(kPlanar), iframeTop,
            builder::CreateLocalTransformation<Real_t>({dxyz[0], dxyz[1], dxyz[2], phiDeg, 0, 0}), use_surf_safety);
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
      }
    }

    if (realSeg) logic.push_back(lminus); // ')' end section logic
  };

  for (size_t i = 0; i < nseg; ++i)
    convertZsegment(i);

  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
