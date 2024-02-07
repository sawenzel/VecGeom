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
  using Vector3D = vecgeom::Vector3D<Real_t>;

  const bool use_surf_safety = false;
  int isurf;
  std::vector<Vector3D> vertices;
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

  if (sideCount == 1) assert(smallerPi && "Polyhedron with one segment cannot have angle larger than pi!");

  auto sidePhi         = phiDelta / sideCount;
  auto cosHalfDeltaPhi = vecCore::math::Cos(0.5 * sidePhi);
  Real_t conv          = 1. / cosHalfDeltaPhi;

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
    bool hasInnerRadius = rMin[iseg] > 0 || rMin[iseg + 1] > 0;
    bool hasOuter       = realSeg || drO != 0;
    bool hasInner       = hasInnerRadius && (realSeg || drI != 0);
    bool hasPhi         = poly.fHasPhiCutout && realSeg;
    if (!hasOuter && !hasInner && !hasPhi) return;

    auto z1 = zPlanes[iseg];
    auto z2 = zPlanes[iseg + 1];

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

    // outer surfaces
    if (hasOuter) {
      for (size_t iside = 0; iside < sideCount; iside++) {
        auto sphi = getPhi(iside);
        auto ephi = getPhi(iside + 1);

        // corners of the outer surfaces
        vertices = {
            {rMax[iseg] * conv * vecCore::math::Cos(sphi), rMax[iseg] * conv * vecCore::math::Sin(sphi), z1},
            {rMax[iseg] * conv * vecCore::math::Cos(ephi), rMax[iseg] * conv * vecCore::math::Sin(ephi), z1},
            {rMax[iseg + 1] * conv * vecCore::math::Cos(ephi), rMax[iseg + 1] * conv * vecCore::math::Sin(ephi), z2},
            {rMax[iseg + 1] * conv * vecCore::math::Cos(sphi), rMax[iseg + 1] * conv * vecCore::math::Sin(sphi), z2}};
        isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id, use_surf_safety);
        if (realSeg) {
          logic.push_back(land);
          logic.push_back(isurf);
        }
      }
    }

    // inner surfaces
    if (hasInner) {
      for (size_t iside = 0; iside < sideCount; iside++) {
        auto sphi = getPhi(iside);
        auto ephi = getPhi(iside + 1);

        vertices = {
            {rMin[iseg + 1] * conv * vecCore::math::Cos(sphi), rMin[iseg + 1] * conv * vecCore::math::Sin(sphi), z2},
            {rMin[iseg + 1] * conv * vecCore::math::Cos(ephi), rMin[iseg + 1] * conv * vecCore::math::Sin(ephi), z2},
            {rMin[iseg] * conv * vecCore::math::Cos(ephi), rMin[iseg] * conv * vecCore::math::Sin(ephi), z1},
            {rMin[iseg] * conv * vecCore::math::Cos(sphi), rMin[iseg] * conv * vecCore::math::Sin(sphi), z1}};
        isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id, use_surf_safety);
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
      auto phi1 = phiStart;
      auto phi2 = (phiStart + phiDelta);

      vertices = {
          {rMax[iseg + 1] * conv * vecCore::math::Cos(phi1), rMax[iseg + 1] * conv * vecCore::math::Sin(phi1), z2},
          {rMin[iseg + 1] * conv * vecCore::math::Cos(phi1), rMin[iseg + 1] * conv * vecCore::math::Sin(phi1), z2},
          {rMin[iseg] * conv * vecCore::math::Cos(phi1), rMin[iseg] * conv * vecCore::math::Sin(phi1), z1},
          {rMax[iseg] * conv * vecCore::math::Cos(phi1), rMax[iseg] * conv * vecCore::math::Sin(phi1), z1}};
      isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id, use_surf_safety);
      logic.push_back(land);
      logic.push_back(lplus); // '('
      logic.push_back(isurf);

      vertices = {
          {rMax[iseg] * conv * vecCore::math::Cos(phi2), rMax[iseg] * conv * vecCore::math::Sin(phi2), z1},
          {rMin[iseg] * conv * vecCore::math::Cos(phi2), rMin[iseg] * conv * vecCore::math::Sin(phi2), z1},
          {rMin[iseg + 1] * conv * vecCore::math::Cos(phi2), rMin[iseg + 1] * conv * vecCore::math::Sin(phi2), z2},
          {rMax[iseg + 1] * conv * vecCore::math::Cos(phi2), rMax[iseg + 1] * conv * vecCore::math::Sin(phi2), z2}};
      isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id, use_surf_safety);
      logic.push_back(smallerPi ? land : lor);
      logic.push_back(isurf);
      logic.push_back(lminus); // ')'
    }

    for (size_t iside = 0; iside < sideCount; iside++) {
      auto sphi = getPhi(iside);
      auto ephi = getPhi(iside + 1);
      // Add bottom frames
      if (iseg == 0) {
        vertices = {{rMin[iseg] * conv * vecCore::math::Cos(ephi), rMin[iseg] * conv * vecCore::math::Sin(ephi), z1},
                    {rMax[iseg] * conv * vecCore::math::Cos(ephi), rMax[iseg] * conv * vecCore::math::Sin(ephi), z1},
                    {rMax[iseg] * conv * vecCore::math::Cos(sphi), rMax[iseg] * conv * vecCore::math::Sin(sphi), z1},
                    {rMin[iseg] * conv * vecCore::math::Cos(sphi), rMin[iseg] * conv * vecCore::math::Sin(sphi), z1}};
        isurf    = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id, use_surf_safety);
      }

      // Add top frames
      if (iseg == (nseg - 1)) {
        vertices = {
            {rMin[iseg + 1] * conv * vecCore::math::Cos(sphi), rMin[iseg + 1] * conv * vecCore::math::Sin(sphi), z2},
            {rMax[iseg + 1] * conv * vecCore::math::Cos(sphi), rMax[iseg + 1] * conv * vecCore::math::Sin(sphi), z2},
            {rMax[iseg + 1] * conv * vecCore::math::Cos(ephi), rMax[iseg + 1] * conv * vecCore::math::Sin(ephi), z2},
            {rMin[iseg + 1] * conv * vecCore::math::Cos(ephi), rMin[iseg + 1] * conv * vecCore::math::Sin(ephi), z2}};
        isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id, use_surf_safety);
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
