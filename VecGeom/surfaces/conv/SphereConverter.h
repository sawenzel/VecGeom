#ifndef VECGEOM_SURFACE_SPHERECONVERTER_H_
#define VECGEOM_SURFACE_SPHERECONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Sphere.h>
#include <VecGeom/volumes/UnplacedOrb.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for a sphere
/// @tparam Real_t Precision type
/// @param sph Sphere solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateSphereSurfaces(vecgeom::UnplacedSphere const &sph, int logical_id, bool intersection = false)
{
  // using Vector3D     = vecgeom::Vector3D<Real_t>;
  // const auto &sphstr = sph.GetStruct();
  auto rmax = sph.GetRmax();
  int isurf;
  LogicExpressionCPU logic; // AND logic: 0 (just Rmax supported for the moment)
  vecgeom::Transformation3DMP<Real_t> identity;

  vecgeom::Precision surfdata[1];
  surfdata[0] = rmax;
  isurf       = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kSpherical, surfdata, /*flipped=*/false),
      builder::CreateFrame<Real_t>(FrameType::kZPhi,
                                   ZPhiMask<Real_t>{-rmax, rmax, true, 0., rmax, 0., vecgeom::kTwoPi}),
      /*identity transformation*/ identity);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(isurf);

  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

/// @brief Converter for an orb
/// @tparam Real_t Precision type
/// @param orb Orb to be converted by way of sphere
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateSphereSurfaces(vecgeom::UnplacedOrb const &orb, int logical_id, bool intersection = false)
{
  vecgeom::UnplacedSphere temp_sph(0, orb.GetRadius());
  return CreateSphereSurfaces<Real_t>(temp_sph, logical_id, intersection);
}

} // namespace conv
} // namespace vgbrep
#endif
