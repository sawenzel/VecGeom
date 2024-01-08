#ifndef VECGEOM_SURFACE_EXTRUDEDCONVERTER_H_
#define VECGEOM_SURFACE_EXTRUDEDCONVERTER_H_

#include <numeric>
#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/SExtru.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Extruded
/// @tparam Real_t Precision type
/// @param xtru Extruded solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateSExtrudedSurfaces(vecgeom::UnplacedSExtruVolume const &xtru, int logical_id)
{
  using Triangle_t      = TriangleMask<Real_t>;
  using Quadrilateral_t = QuadrilateralMask<Real_t>;
  using Vector3         = vecgeom::Vector3D<Real_t>;

  auto const &shell         = xtru.GetStruct();
  int n_vertices            = shell.GetPolygon().GetNVertices();
  auto const &vertices_poly = shell.GetPolygon().GetVertices();

  const auto vertx = vertices_poly.x();
  const auto verty = vertices_poly.y();

  const bool use_surf_safety = true;
  LogicExpressionCPU logic;
  int isurf;

  vecgeom::Transformation3D transformation;
  std::vector<Vector3> vertices;
  std::vector<Vector3> triangle_var;
  Vector3 vert1 = {0., 0., 0.};
  Vector3 vert2 = {0., 0., 0.};

  Vector3 section_origin1(0, 0, shell.GetLowerZ());
  Vector3 section_origin2(0, 0, shell.GetUpperZ());
  double section_scale1 = 1;
  double section_scale2 = 1;

  // Side surfaces in the convex polygon case
  for (int i = 0; i < n_vertices; ++i) {
    vert1.Set(vertx[i], verty[i], 0);
    vert2.Set(vertx[(i + 1) % n_vertices], verty[(i + 1) % n_vertices], 0);

    vertices       = {section_origin1 + section_scale1 * vert1, section_origin2 + section_scale2 * vert1,
                      section_origin2 + section_scale2 * vert2, section_origin1 + section_scale1 * vert2};
    transformation = builder::TransformationFromPlanarPoints<Real_t>(vertices);
    isurf          = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(kPlanar),
        builder::CreateFrame<Real_t>(kQuadrilateral, Quadrilateral_t{vertices[0].x(), vertices[0].y(), vertices[1].x(),
                                                                     vertices[1].y(), vertices[2].x(), vertices[2].y(),
                                                                     vertices[3].x(), vertices[3].y()}),
        builder::CreateLocalTransformation<Real_t>(transformation), use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(isurf);
    logic.push_back(land);
  }

  // Triangualion by ear clipping

  auto sign = [=](Vector3 v1, Vector3 v2, Vector3 v3) {
    return (v1[0] - v3[0]) * (v2[1] - v3[1]) - (v2[0] - v3[0]) * (v1[1] - v3[1]);
  };

  auto is_inside_triangle = [=](Vector3 const &vert_a, Vector3 const &vert_b, Vector3 const &vert_c,
                                Vector3 const &vert_p) {
    auto d1 = sign(vert_p, vert_a, vert_b);
    auto d2 = sign(vert_p, vert_b, vert_c);
    auto d3 = sign(vert_p, vert_c, vert_a);

    auto has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    auto has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
  };

  auto is_ear = [&](int a, int b, int c, Vector3 const &vert_a, Vector3 const &vert_b, Vector3 const &vert_c) {
    for (int k = 0; k < n_vertices; k++) {
      if (k != a && k != b && k != c) {
        Vector3 vert_p(vertx[k], verty[k], 0);
        if (is_inside_triangle(vert_a, vert_b, vert_c, vert_p)) {
          return false;
        }
      }
    }
    return true;
  };

  std::vector<int> n(n_vertices);
  std::iota(n.begin(), n.end(), 0);
  std::vector<std::vector<Vector3>> triangles;

  while (n.size() > 2) {
    for (unsigned i = 0; i < n.size(); ++i) {
      int a = n[(i + n.size() - 1) % n.size()];
      int b = n[i];
      int c = n[(i + 1) % n.size()];
      Vector3 vert_a(vertx[a], verty[a], 0);
      Vector3 vert_b(vertx[b], verty[b], 0);
      Vector3 vert_c(vertx[c], verty[c], 0);

      if (is_ear(a, b, c, vert_a, vert_b, vert_c)) {
        triangles.push_back({vert_c, vert_b, vert_a});
        n.erase(n.begin() + i);
      }
    }
  }

  for (unsigned j=0; j < triangles.size(); j++) {
    triangle_var = triangles[j];

    // bottom triangles
    vertices       = {section_origin1 + section_scale1 * triangle_var[2], section_origin1 + section_scale1 * triangle_var[1],
                      section_origin1 + section_scale1 * triangle_var[0]};
    transformation = builder::TransformationFromPlanarPoints<Real_t>(vertices);
    isurf          = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(kPlanar),
        builder::CreateFrame<Real_t>(kTriangle, Triangle_t{vertices[0].x(), vertices[0].y(), vertices[1].x(),
                                                           vertices[1].y(), vertices[2].x(), vertices[2].y()}),
        builder::CreateLocalTransformation<Real_t>(transformation), use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);

    // top triangles
    vertices       = {section_origin2 + section_scale2 * triangle_var[0], section_origin2 + section_scale2 * triangle_var[1],
                      section_origin2 + section_scale2 * triangle_var[2]};
    transformation = builder::TransformationFromPlanarPoints<Real_t>(vertices);
    isurf          = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(kPlanar),
        builder::CreateFrame<Real_t>(kTriangle, Triangle_t{vertices[0].x(), vertices[0].y(), vertices[1].x(),
                                                           vertices[1].y(), vertices[2].x(), vertices[2].y()}),
        builder::CreateLocalTransformation<Real_t>(transformation), use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  }

  // bottom virtual surface
  isurf = builder::CreateLocalSurface<Real_t>(
          builder::CreateUnplacedSurface<Real_t>(kPlanar), Frame{kNoFrame},
          builder::CreateLocalTransformation<Real_t>({0, 0, shell.GetLowerZ(), 0, 180, 0}), use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);
  logic.push_back(land);

  // top virtual surface
  isurf = builder::CreateLocalSurface<Real_t>(
          builder::CreateUnplacedSurface<Real_t>(kPlanar), Frame{kNoFrame},
          builder::CreateLocalTransformation<Real_t>({0, 0, shell.GetUpperZ(), 0, 0, 0}), use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);

  builder::AddLogicToShell<Real_t>(logical_id, logic);

  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
