#ifndef VECGEOM_SURFACE_CUTTUBECONVERTER_H_
#define VECGEOM_SURFACE_CUTTUBECONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/CutTube.h>

namespace vgbrep {
namespace conv {

// sign function
template <typename Real_t>
int sgn(Real_t val)
{
  return (Real_t(0) < val) - (val < Real_t(0));
}

/// @brief Converter for Tube
/// @tparam Real_t Precision type
/// @param tube Tube solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateTubeSurfaces(vecgeom::UnplacedCutTube const &tube, int logical_id)
{
  using ZPhiMask_t   = ZPhiMask<Real_t>;
  using WindowMask_t = WindowMask<Real_t>;
  using Vector3      = vecgeom::Vector3D<Real_t>;

  LogicExpressionCPU logic; // top & bottom & [rmin] & rmax & (dphi < 180) ? sphi * ephi : sphi | ephi
  auto sphi  = tube.sphi();
  auto dphi  = tube.dphi();
  auto ephi  = tube.sphi() + tube.dphi();
  auto csphi = vecCore::math::Cos(sphi);
  auto ssphi = vecCore::math::Sin(sphi);
  auto cephi = vecCore::math::Cos(ephi);
  auto sephi = vecCore::math::Sin(ephi);
  auto rmin  = tube.rmin();
  auto rmax  = tube.rmax();

  // get normal of the end caps
  auto bottom_normal = tube.BottomNormal();
  auto top_normal    = tube.TopNormal();

  // get maximum extent
  Vector3 aMin, aMax;
  tube.Extent(aMin, aMax);

  assert(dphi > vecgeom::kTolerance);

  bool fullCirc        = ApproxEqual(dphi, vecgeom::kTwoPi);
  bool smallerPi       = dphi < (vecgeom::kPi - vecgeom::kTolerance);
  bool use_surf_safety = true;

  int isurf;
  Real_t surfdata[2];

  // We need angles in degrees for transformations
  auto thetad_top    = top_normal.Theta() * vecgeom::kRadToDeg;
  auto phid_top      = top_normal.Phi() * vecgeom::kRadToDeg;
  auto thetad_bottom = bottom_normal.Theta() * vecgeom::kRadToDeg;
  auto phid_bottom   = bottom_normal.Phi() * vecgeom::kRadToDeg;
  // assert for 0 <= theta top <= 90 and 90 <= theta bottom <= 180
  assert(top_normal[2] >= 0 && bottom_normal[2] <= 0);

  auto &cpudata = CPUsurfData<Real_t>::Instance();

  // surface at +dz
  // due to the cut the top and bottom caps are elliptical, which is handled by using a rectangle frame
  // that fully contains the ellipse and making the surface logical such that the full boolean expression
  // must be evaluated to decide whether it is a hit or not.
  // As a consequence, all surfaces of this volume must be logical surfaces.
  assert(std::abs(top_normal.z()) > vecgeom::kTolerance); // assert before division
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kWindow, WindowMask_t{tube.rmax(), tube.rmax() / top_normal.z()}),
      builder::CreateLocalTransformation<Real_t>({0, 0, tube.z(), phid_top - 90, -thetad_top, 0}), use_surf_safety);
  auto &surf = cpudata.fLocalSurfaces[isurf];
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  // Make the surface "logical"
  surf.fLogicId = isurf;
  logic.push_back(isurf);
  logic.push_back(land);

  // surface at -dz
  assert(std::abs(std::cos(vecgeom::kPi - bottom_normal.Theta()) > vecgeom::kTolerance)); // assert before division
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kWindow,
                                   WindowMask_t{tube.rmax(), tube.rmax() / cos(vecgeom::kPi - bottom_normal.Theta())}),
      builder::CreateLocalTransformation<Real_t>({0, 0, -tube.z(), phid_bottom - 90, -thetad_bottom, 0}),
      use_surf_safety);
  auto &surf2 = cpudata.fLocalSurfaces[isurf];
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  // Make the surface "logical"
  surf2.fLogicId = isurf;
  logic.push_back(isurf);

  // inner cylinder
  if (tube.rmin() > vecgeom::kTolerance) {
    surfdata[0] = tube.rmin();
    isurf       = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(SurfaceType::kCylindrical, surfdata, /*flipped=*/true),
        builder::CreateFrame<Real_t>(FrameType::kZPhi, ZPhiMask_t{aMin[2], aMax[2], fullCirc, sphi, ephi}),
        /*identity transformation*/ 0, use_surf_safety);
    auto &surf3 = cpudata.fLocalSurfaces[isurf];
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    // Make the surface "logical"
    surf3.fLogicId = isurf;
    logic.push_back(land);
    logic.push_back(isurf);
  }
  // outer cylinder
  surfdata[0] = tube.rmax();
  isurf       = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kCylindrical, surfdata),
      builder::CreateFrame<Real_t>(FrameType::kZPhi, ZPhiMask_t{aMin[2], aMax[2], fullCirc, sphi, ephi}),
      /*identity transformation*/ 0, use_surf_safety);
  auto &surf4 = cpudata.fLocalSurfaces[isurf];
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  // Make the surface "logical"
  surf4.fLogicId = isurf;
  logic.push_back(land);
  logic.push_back(isurf);

  if (ApproxEqual(dphi, vecgeom::kTwoPi)) {
    builder::AddLogicToShell<Real_t>(logical_id, logic);
    return true;
  }

  // Implementation of the plane caps at Sphi and Ephi by using a window with the size of the extent of the
  // quadrilaterals. Depending on the phi cut position, this window is a bit larger than the actual surface, but we
  // found this implementation to be faster than a quadrilateral with the correct size plane cap at Sphi
  auto zmin1 = std::min(-tube.z() - rmin * (bottom_normal.x() * csphi + bottom_normal.y() * ssphi) / bottom_normal.z(),
                        -tube.z() - rmax * (bottom_normal.x() * csphi + bottom_normal.y() * ssphi) / bottom_normal.z());
  auto zmax1 = std::max(tube.z() - rmin * (top_normal.x() * csphi + top_normal.y() * ssphi) / top_normal.z(),
                        tube.z() - rmax * (top_normal.x() * csphi + top_normal.y() * ssphi) / top_normal.z());
  auto zmin2 = std::min(-tube.z() - rmin * (bottom_normal.x() * cephi + bottom_normal.y() * sephi) / bottom_normal.z(),
                        -tube.z() - rmax * (bottom_normal.x() * cephi + bottom_normal.y() * sephi) / bottom_normal.z());
  auto zmax2 = std::max(tube.z() - rmin * (top_normal.x() * cephi + top_normal.y() * sephi) / top_normal.z(),
                        tube.z() - rmax * (top_normal.x() * cephi + top_normal.y() * sephi) / top_normal.z());

  std::vector<Vector3> vert(4);
  vert[0].Set(rmin * csphi, rmin * ssphi, zmin1);
  vert[1].Set(rmax * csphi, rmax * ssphi, zmin1);
  vert[2].Set(rmax * csphi, rmax * ssphi, zmax1);
  vert[3].Set(rmin * csphi, rmin * ssphi, zmax1);
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id, use_surf_safety && smallerPi);
  assert(isurf >= 0);
  // Make the surface "logical"
  cpudata.fLocalSurfaces[isurf].fLogicId = isurf;
  logic.push_back(land);
  logic.push_back(lplus); // '('
  logic.push_back(isurf);

  // plane cap at Sphi+Dphi
  vert[0].Set(rmax * cephi, rmax * sephi, zmin2);
  vert[1].Set(rmin * cephi, rmin * sephi, zmin2);
  vert[2].Set(rmin * cephi, rmin * sephi, zmax2);
  vert[3].Set(rmax * cephi, rmax * sephi, zmax2);
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id, use_surf_safety && smallerPi);
  assert(isurf >= 0);
  // Make the surface "logical"
  cpudata.fLocalSurfaces[isurf].fLogicId = isurf;
  logic.push_back(smallerPi ? land : lor);
  logic.push_back(isurf);
  logic.push_back(lminus); // ')'
  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
