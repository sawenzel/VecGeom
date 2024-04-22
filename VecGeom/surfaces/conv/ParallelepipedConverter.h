#ifndef VECGEOM_SURFACE_PARALLELEPIPEDCONVERTER_H_
#define VECGEOM_SURFACE_PARALLELEPIPEDCONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Parallelepiped.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Parallelepiped
/// @tparam Real_t Precision type
/// @param para Parallelepiped solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateParallelepipedSurfaces(vecgeom::UnplacedParallelepiped const &para, int logical_id)
{
  using Vector3D = vecgeom::Vector3D<vecgeom::Precision>;

  auto dx = para.GetX(); // half length in x
  auto dy = para.GetY(); // half length in y
  auto dz = para.GetZ(); // half length in z

  auto txy = para.GetTanAlpha();
  auto txz = para.GetTanThetaCosPhi();
  auto tyz = para.GetTanThetaSinPhi();

  int isurf;
  LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3 & 4 & 5
  vecgeom::Transformation3D transformation;
  std::vector<Vector3D> vert; // Stores coordinates of the four corners that create a parallelogram. It is initialised
                              // with 3D coordinates.

  // corners represented as vectors (-x, -y, -z), (+x, -y, -z), (+x, +y, -z), (-x, +y, -z), same for +z
  std::vector<Vector3D> corners = {
      {-dx - dy * txy - dz * txz, -dy - dz * tyz, -dz}, {-dx + dy * txy - dz * txz, +dy - dz * tyz, -dz},
      {+dx + dy * txy - dz * txz, +dy - dz * tyz, -dz}, {+dx - dy * txy - dz * txz, -dy - dz * tyz, -dz},
      {-dx - dy * txy + dz * txz, -dy + dz * tyz, +dz}, {-dx + dy * txy + dz * txz, +dy + dz * tyz, +dz},
      {+dx + dy * txy + dz * txz, +dy + dz * tyz, +dz}, {+dx - dy * txy + dz * txz, -dy + dz * tyz, +dz}};

  // surface at -dx:
  vert  = {corners[1], corners[0], corners[4], corners[5]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  logic.push_back(isurf);

  // surface at +dx:
  vert  = {corners[3], corners[2], corners[6], corners[7]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  logic.push_back(land);
  logic.push_back(isurf);

  // surface at -dy:
  vert  = {corners[0], corners[3], corners[7], corners[4]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  logic.push_back(land);
  logic.push_back(isurf);

  // surface at +dy:
  vert  = {corners[2], corners[1], corners[5], corners[6]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  logic.push_back(land);
  logic.push_back(isurf);

  // surface at -dz:
  vert  = {corners[0], corners[1], corners[2], corners[3]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  logic.push_back(land);
  logic.push_back(isurf);

  // surface at +dz:
  vert  = {corners[7], corners[6], corners[5], corners[4]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  logic.push_back(land);
  logic.push_back(isurf);
  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
