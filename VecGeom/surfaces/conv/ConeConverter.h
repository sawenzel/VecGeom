#ifndef VECGEOM_SURFACE_CONECONVERTER_H_
#define VECGEOM_SURFACE_CONECONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Cone.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Cone
/// @tparam Real_t Precision type
/// @param cone Cone solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateConeSurfaces(vecgeom::UnplacedCone const &cone, int logical_id)
{
  using RingMask_t = RingMask<Real_t>;
  using ZPhiMask_t = ZPhiMask<Real_t>;
  using Vector3D   = vecgeom::Vector3D<Real_t>;

  LogicExpressionCPU logic; // top & bottom & [rmin] & rmax & (dphi < 180) ? sphi * ephi : sphi | ephi
  auto rmin1 = cone.GetRmin1();
  auto rmax1 = cone.GetRmax1();
  auto rmin2 = cone.GetRmin2();
  auto rmax2 = cone.GetRmax2();
  auto dz    = cone.GetDz();
  auto sphi  = cone.GetSPhi();
  auto dphi  = cone.GetDPhi();
  auto ephi  = dphi + sphi;

  assert(dphi > vecgeom::kTolerance);
  assert(rmax1 - rmin1 > -vecgeom::kTolerance);
  assert(rmax2 - rmin2 > -vecgeom::kTolerance);

  bool fullCirc  = ApproxEqual(dphi, vecgeom::kTwoPi);
  bool smallerPi = dphi < (vecgeom::kPi - vecgeom::kTolerance);

  int isurf;
  Real_t surfdata[2];

  // We need angles in degrees for transformations
  auto sphid = vecgeom::kRadToDeg * sphi;
  auto ephid = vecgeom::kRadToDeg * ephi;

  // surface at +Dz
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmin2, rmax2, fullCirc, sphi, ephi}),
      builder::CreateLocalTransformation<Real_t>({0, 0, dz, 0, 0, 0}));
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);
  // surface at -Dz
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmin1, rmax1, fullCirc, sphi, ephi}),
      builder::CreateLocalTransformation<Real_t>({0, 0, -dz, 0, 180, -sphid - ephid}));
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // inner cone
  if (rmin2 > vecgeom::kTolerance || rmin1 > vecgeom::kTolerance) {
    surfdata[0] = 0.5 * (rmin1 + rmin2);
    surfdata[1] = 0.5 * (rmin2 - rmin1) / dz;
    auto stype  = std::abs(surfdata[1]) > vecgeom::kTolerance ? SurfaceType::kConical : SurfaceType::kCylindrical;
    isurf       = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(stype, surfdata, /*flipped=*/true),
        builder::CreateFrame<Real_t>(FrameType::kZPhi, ZPhiMask_t{-dz, dz, fullCirc, sphi, ephi, rmin1, rmin2}),
        /*identity transformation*/ 0);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
  }
  // outer cone
  surfdata[0] = 0.5 * (rmax1 + rmax2);
  surfdata[1] = 0.5 * (rmax2 - rmax1) / dz;
  auto stype  = std::abs(surfdata[1]) > vecgeom::kTolerance ? SurfaceType::kConical : SurfaceType::kCylindrical;
  isurf       = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(stype, surfdata),
      builder::CreateFrame<Real_t>(FrameType::kZPhi, ZPhiMask_t{-dz, dz, fullCirc, sphi, ephi, rmax1, rmax2}),
      /*identity transformation*/ 0);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);

  if (ApproxEqual(dphi, vecgeom::kTwoPi)) {
    builder::AddLogicToShell<Real_t>(logical_id, logic);
    return true;
  }

  vecgeom::Transformation3D transformation;
  std::vector<Vector3D> vert; // Stores 3D coordinates of the four corners that create a quadrilateral.
  auto csphi = std::cos(sphi);
  auto ssphi = std::sin(sphi);
  auto cephi = std::cos(ephi);
  auto sephi = std::sin(ephi);

  // corners of the sphi and ephi faces
  std::vector<Vector3D> corners = {{rmin1 * csphi, rmin1 * ssphi, -dz}, {rmax1 * csphi, rmax1 * ssphi, -dz},
                                   {rmax2 * csphi, rmax2 * ssphi, dz},  {rmin2 * csphi, rmin2 * ssphi, dz},
                                   {rmax1 * cephi, rmax1 * sephi, -dz}, {rmin1 * cephi, rmin1 * sephi, -dz},
                                   {rmin2 * cephi, rmin2 * sephi, dz},  {rmax2 * cephi, rmax2 * sephi, dz}};

  // plane cap at sphi
  vert  = {corners[0], corners[1], corners[2], corners[3]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (isurf >= 0) {
    logic.push_back(land);
    logic.push_back(lplus); // '('
    logic.push_back(isurf);
  }

  // plane cap at sphi+dphi
  vert  = {corners[4], corners[5], corners[6], corners[7]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (isurf >= 0) {
    logic.push_back(smallerPi ? land : lor);
    logic.push_back(isurf);
    logic.push_back(lminus); // ')'
  }

  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
