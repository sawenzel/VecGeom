#ifndef VECGEOM_SURFACE_EXTRUDEDCONVERTER_H_
#define VECGEOM_SURFACE_EXTRUDEDCONVERTER_H_

#include <numeric>
#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/management/Logger.h>

#include <VecGeom/volumes/SExtru.h>

namespace vgbrep {
namespace conv {

template <typename Real_t>
struct ReducedPoly {
  using Vector3 = vecgeom::Vector3D<Real_t>;

  int Nconvex;              // number of convex vertices
  int Nvert;                // total number of vertices
  std::vector<int> ind_arr; // vector of all indices of the polygon
  int *ind_arr_conv;        // list of all indices of the outscribed convex polygon
  std::vector<std::vector<int>>
      list_of_gaps; // list of all gaps that consists of index lists of the concave polygons in the gap

  bool is_convex;

  // Constructor that takes a vector of ints to initialize the index array
  ReducedPoly(const std::vector<int> &input_vec)
      : Nconvex(0), Nvert(input_vec.size()), ind_arr(input_vec), is_convex(false)
  {
  }

  bool IsRightSided(Vector3 v1, Vector3 v2, Vector3 v3) const
  {
    Real_t dot = (v1[0] - v2[0]) * (v3[1] - v2[1]) - (v1[1] - v2[1]) * (v3[0] - v2[0]);
    return (dot < -vecgeom::kTolerance) ? false : true;
  };

  /// @brief Helper function whether a polygon is convex
  /// @param vertx vertices in x of original, global polygon
  /// @param verty vertices in y of original, global polygon
  /// @return If the ReducedPoly polygon is convex
  bool IsConvex(const Real_t *vertx, const Real_t *verty)
  {
    if (Nvert == 3) return true;
    int j, k;
    Vector3 point1;
    Vector3 point2;
    Vector3 point3;
    for (int i = 0; i < Nvert; i++) {
      j = (i + 1) % Nvert;
      k = (i + 2) % Nvert;
      if (!IsRightSided({vertx[ind_arr[i]], verty[ind_arr[i]], 0}, {vertx[ind_arr[j]], verty[ind_arr[j]], 0},
                        {vertx[ind_arr[k]], verty[ind_arr[k]], 0}))
        return false;
    }
    return true;
  };

  /// @brief Helper function whether a polygon is convex
  /// @param vertx vertices in x of original, global polygon
  /// @param verty vertices in y of original, global polygon
  /// @return If the ReducedPoly polygon is convex
  bool IsSegConvex(const Real_t *vertx, const Real_t *verty, int i1, int i2 = -1)
  {
    if (i2 < 0) i2 = (i1 + 1) % Nvert;

    for (int i = 0; i < Nvert; i++) {
      if (i == i1 || i == i2) continue;

      if (!IsRightSided({vertx[ind_arr[i]], verty[ind_arr[i]], 0}, {vertx[ind_arr[i1]], verty[ind_arr[i1]], 0},
                        {vertx[ind_arr[i2]], verty[ind_arr[i2]], 0}))
        return false;
    }
    return true;
  };

  /// @brief Helper function whether the surface is a real, framed surface or a virtual one
  /// @param index index of the convex vertex to be checked
  /// @param Ntotalvert number of vertices of the original, global polygon
  /// @return if the surface is real
  bool IsRealSurface(int index, int Ntotalvert)
  {
    // check whether two subsequent points in the local polygon correspond to two subsequent points
    // in the global polygon.
    int nstep = ind_arr_conv[(index + 1) % Nconvex] - ind_arr_conv[index];

    if (((ind_arr_conv[(index + 1) % Nconvex] == 0) && ind_arr_conv[index] == Ntotalvert - 1) ||
        ((ind_arr_conv[index] == 0) && ind_arr_conv[(index + 1) % Nconvex] == Ntotalvert - 1))
      return true;

    if (std::abs(nstep) == 1) {
      return true;
    } else {
      return false;
    }
  };

  /// @brief This function sets the array of convex indices ind_arr_conv,
  ///        and creates the list of gaps, which contains the list of concave indices per gap
  /// @param vertx vertices in x of original, global polygon
  /// @param verty vertices in y of original, global polygon
  void GetConvIndices(const Real_t *vertx, const Real_t *verty)
  {
    int iseg = 0;
    int ivnew;
    std::unique_ptr<int[]> indconv(new int[Nvert]);
    std::memset(indconv.get(), 0, Nvert * sizeof(int));

    while (iseg < Nvert) {                    // loop over all local indices
      if (!IsSegConvex(vertx, verty, iseg)) { // if a segment is not convex
                                              // Loop over segments to find the next convex segment.
        if (iseg + 2 > Nvert) {               // finished loop over indices
          break;
        }
        ivnew     = (iseg + 2) % Nvert; // next vertex
        bool conv = false;
        // check iseg with next vertices
        while (ivnew) {
          if (IsSegConvex(vertx, verty, iseg, ivnew)) {
            conv = true;
            break;
          }
          // increase indices until next vertex is convex. If last vertex is checked, ivnew = 0 and loop breaks
          ivnew = (ivnew + 1) % Nvert;
        }
        if (!conv) { // no convex found, jump to the next segment
          iseg++;
          continue;
        }
      } else { // if segment is convex, pick the next point as index
        ivnew = (iseg + 1) % Nvert;
      }

      // fill the convex array indconv and increase Nconvex simultaneously
      if (!Nconvex) {
        indconv[Nconvex++] = iseg;
      } else if (indconv[Nconvex - 1] != iseg) {
        indconv[Nconvex++] = iseg;
      }
      if (iseg < Nvert - 1) {
        indconv[Nconvex++] = ivnew;
      }
      if (ivnew < iseg) break; // next index is smaller than iseg, so we looped through all indices.
      iseg = ivnew;            // set iseg to next point in the convex outscribed polygon
    }
    if (!Nconvex) {
      VECGEOM_LOG(warning) << " ERROR, no convex point found in polygon generation for a supposedly convex polygon!"
                           << std::endl;
      return;
    }
    // copy local convex indices into array
    // note, these are not the global indices yet!
    ind_arr_conv = new int[Nvert];
    memcpy(ind_arr_conv, indconv.get(), Nconvex * sizeof(int));

    std::vector<int> ind_arr_conc;

    // Loop over convex indices and check whether there is a jump between two convex indices
    int idx_cx = 0;
    int indnext, indback;
    int nskip;
    while (idx_cx < Nconvex) {
      indnext = (idx_cx + 1) % Nconvex;
      nskip   = ind_arr_conv[indnext] - ind_arr_conv[idx_cx];
      if (nskip < 0) nskip += Nvert;
      if (nskip == 1) {
        idx_cx++;
        continue;
      }

      // A gap was found. Generate a polygon out of concave indices
      // in a backward manner so that it has a right-handed orientation
      ind_arr_conc.push_back(ind_arr[ind_arr_conv[idx_cx]]);
      ind_arr_conc.push_back(ind_arr[ind_arr_conv[indnext]]);
      indback = ind_arr_conv[indnext] - 1;
      if (indback < 0) indback += Nvert;
      while (indback != ind_arr_conv[idx_cx]) {
        ind_arr_conc.push_back(ind_arr[indback]);
        indback--;
        if (indback < 0) indback += Nvert;
      }

      // push gap to list of gaps
      list_of_gaps.push_back(ind_arr_conc);
      ind_arr_conc.clear();
      idx_cx++;
    }

    // transform local indices in the convex index array to global indices.
    for (idx_cx = 0; idx_cx < Nconvex; idx_cx++) {
      ind_arr_conv[idx_cx] = ind_arr[ind_arr_conv[idx_cx]];
    }

    return;
  };
};

/// @brief Converter for Extruded
/// @tparam Real_t Precision type
/// @param xtru Extruded solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateSExtrudedSurfaces(vecgeom::UnplacedSExtruVolume const &xtru, int logical_id)
{
  using Vector3 = vecgeom::Vector3D<Real_t>;

  auto const &shell         = xtru.GetStruct();
  int n_vertices            = shell.GetPolygon().GetNVertices();
  auto const &vertices_poly = shell.GetPolygon().GetVertices();

  // vector of global indices
  std::vector<int> idx_vec(n_vertices);
  std::iota(idx_vec.begin(), idx_vec.end(), 0);

  auto poly = ReducedPoly<Real_t>(idx_vec);

  // arrays of global vertices
  const auto vertx = vertices_poly.x();
  const auto verty = vertices_poly.y();

  LogicExpressionCPU logic;
  int isurf;

  vecgeom::Transformation3D transformation;
  std::vector<Vector3> vertices;
  std::vector<Vector3> triangle_var;
  Vector3 vert1 = {0., 0., 0.};
  Vector3 vert2 = {0., 0., 0.};

  Vector3 section_origin1(0, 0, shell.GetLowerZ());
  Vector3 section_origin2(0, 0, shell.GetUpperZ());
  Real_t section_scale1 = 1;
  Real_t section_scale2 = 1;

  // recursive function to generate side surfaces of the extruded from a polygon
  std::function<void(const ReducedPoly<Real_t> &, int)> GenerateSurfaces = [&](ReducedPoly<Real_t> input_poly,
                                                                               int level = 0) {
    logic.push_back(lplus); // use parenthesis around each polygon
    if (input_poly.IsConvex(vertx, verty)) {

      // if polygon is convex, the convex index array equals the full index array
      input_poly.ind_arr_conv = new int[input_poly.Nvert];
      for (int i = 0; i < input_poly.Nvert; ++i) {
        input_poly.ind_arr_conv[i] = input_poly.ind_arr[i];
      }
      input_poly.Nconvex = input_poly.Nvert;

      // For convex polygon, the surfaces can be directly generated
      for (int i = 0; i < input_poly.Nvert; ++i) {
        vert1.Set(vertx[input_poly.ind_arr_conv[i]], verty[input_poly.ind_arr_conv[i]], 0);
        vert2.Set(vertx[input_poly.ind_arr_conv[(i + 1) % input_poly.Nconvex]],
                  verty[input_poly.ind_arr_conv[(i + 1) % input_poly.Nconvex]], 0);

        // for nested polygons:
        // Since real surfaces must point in the outwards direction of the original polygon,
        // their vertices are reversed for odd levels (so the they are negated).
        // However, since in the logic the full higher level is already negated, the surface must be
        // additionally negated to revert the negation by the index reversion.
        if (level % 2 == 0) {
          vertices = {section_origin1 + section_scale1 * vert1, section_origin2 + section_scale2 * vert1,
                      section_origin2 + section_scale2 * vert2, section_origin1 + section_scale1 * vert2};
        } else {
          vertices = {section_origin1 + section_scale1 * vert2, section_origin2 + section_scale2 * vert2,
                      section_origin2 + section_scale2 * vert1, section_origin1 + section_scale1 * vert1};
          logic.push_back(lnot);
        }

        if (input_poly.IsRealSurface(i, n_vertices)) {
          // create real surface with frame
          isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id);
        } else {
          // create virtual surface without frame
          transformation = builder::TransformationFromPlanarPoints<Real_t>(vertices);
          isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
                                                      Frame{FrameType::kNoFrame},
                                                      builder::CreateLocalTransformation<Real_t>(transformation));
          builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        }

        logic.push_back(isurf);
        if (i < input_poly.Nconvex - 1) {
          logic.push_back(land);
        }
      }

    } else { // else: polygon is concave

      // get convex indices, list of gaps with indices of all gaps
      input_poly.GetConvIndices(vertx, verty);

      // creating the outscribed convex polygone surfaces
      for (int i = 0; i < input_poly.Nconvex; ++i) {

        vert1.Set(vertx[input_poly.ind_arr_conv[i]], verty[input_poly.ind_arr_conv[i]], 0);
        vert2.Set(vertx[input_poly.ind_arr_conv[(i + 1) % input_poly.Nconvex]],
                  verty[input_poly.ind_arr_conv[(i + 1) % input_poly.Nconvex]], 0);

        // for nested polygons:
        // Since real surfaces must point in the outwards direction of the original polygon,
        // their vertices are reversed for odd levels (so the they are negated).
        // However, since in the logic the full higher level is already negated, the surface must be
        // additionally negated to revert the negation by the index reversion.
        if (level % 2 == 0) {
          vertices = {section_origin1 + section_scale1 * vert1, section_origin2 + section_scale2 * vert1,
                      section_origin2 + section_scale2 * vert2, section_origin1 + section_scale1 * vert2};
        } else {
          vertices = {section_origin1 + section_scale1 * vert2, section_origin2 + section_scale2 * vert2,
                      section_origin2 + section_scale2 * vert1, section_origin1 + section_scale1 * vert1};
          logic.push_back(lnot);
        }

        if (input_poly.IsRealSurface(i, n_vertices)) {
          // create real surface with frame
          isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id);
        } else {
          // create virtual surface without frame
          transformation = builder::TransformationFromPlanarPoints<Real_t>(vertices);
          isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
                                                      Frame{FrameType::kNoFrame},
                                                      builder::CreateLocalTransformation<Real_t>(transformation));
          builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        }
        logic.push_back(isurf);
        if (i < input_poly.Nconvex - 1) {
          logic.push_back(land);
        }
      }

      // LOGIC: polygon = conv surf 1 & conv surf 2 & ... & (!(gap1 | gap2 | ... ) )
      // note, gaps can also be concave and call the function recursively
      logic.push_back(land); // last & before subtraction of gaps

      logic.push_back(lplus);
      logic.push_back(lnot);  // negation for next level
      logic.push_back(lplus); // bracket around gaps

      for (auto j = 0u; j < input_poly.list_of_gaps.size(); ++j) {
        // creating the concave polygone that needs to be subtracted
        auto subpoly = ReducedPoly<Real_t>(input_poly.list_of_gaps[j]);
        GenerateSurfaces(subpoly, level + 1);
        // the addition of gaps is their union
        if (j < input_poly.list_of_gaps.size() - 1) logic.push_back(lor);
      }
      logic.push_back(lminus);
      logic.push_back(lminus);
    }
    logic.push_back(lminus);
  }; // end of defintion GenerateSurfaces

  // generate side surfaces of polygon
  GenerateSurfaces(poly, 0);
  logic.push_back(land); // logical and between the side surfaces and the top and bottom surfaces
  // Preparation for bottom and top surfaces
  // Triangualion by ear clipping

  auto sign = [=](Vector3 v1, Vector3 v2, Vector3 v3) {
    return (v1[0] - v3[0]) * (v2[1] - v3[1]) - (v2[0] - v3[0]) * (v1[1] - v3[1]);
  };

  auto IsInsideTriangle = [=](Vector3 const &vert_a, Vector3 const &vert_b, Vector3 const &vert_c,
                              Vector3 const &vert_p) {
    auto d1 = sign(vert_p, vert_a, vert_b);
    auto d2 = sign(vert_p, vert_b, vert_c);
    auto d3 = sign(vert_p, vert_c, vert_a);

    auto has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    auto has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
  };

  auto IsEar = [&](int a, int b, int c, Vector3 const &vert_a, Vector3 const &vert_b, Vector3 const &vert_c) {
    for (int k = 0; k < n_vertices; k++) {
      if (k != a && k != b && k != c) {
        Vector3 vert_p(vertx[k], verty[k], 0);
        if (IsInsideTriangle(vert_a, vert_b, vert_c, vert_p)) {
          return false;
        }
      }
    }
    // if vertex itself is concave return false
    if (sign(vert_a, vert_b, vert_c) >= 0) return false;
    return true;
  };

  std::vector<std::vector<Vector3>> triangles;
  while (idx_vec.size() > 2) {
    for (unsigned i = 0; i < idx_vec.size(); ++i) {
      int a = idx_vec[(i + idx_vec.size() - 1) % idx_vec.size()];
      int b = idx_vec[i];
      int c = idx_vec[(i + 1) % idx_vec.size()];
      Vector3 vert_a(vertx[a], verty[a], 0);
      Vector3 vert_b(vertx[b], verty[b], 0);
      Vector3 vert_c(vertx[c], verty[c], 0);

      if (IsEar(a, b, c, vert_a, vert_b, vert_c)) {
        triangles.push_back({vert_c, vert_b, vert_a});
        idx_vec.erase(idx_vec.begin() + i);
      }
    }
  }

  for (unsigned j = 0; j < triangles.size(); j++) {
    triangle_var = triangles[j];

    // bottom triangles
    vertices = {section_origin1 + section_scale1 * triangle_var[2], section_origin1 + section_scale1 * triangle_var[1],
                section_origin1 + section_scale1 * triangle_var[0]};
    isurf    = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id);

    // top triangles
    vertices = {section_origin2 + section_scale2 * triangle_var[0], section_origin2 + section_scale2 * triangle_var[1],
                section_origin2 + section_scale2 * triangle_var[2]};
    isurf    = builder::CreateLocalSurfaceFromVertices<Real_t>(vertices, logical_id);
  }

  // bottom virtual surface
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar), Frame{FrameType::kNoFrame},
      builder::CreateLocalTransformation<Real_t>({0, 0, shell.GetLowerZ(), 0, 180, 0}));
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);
  logic.push_back(land);

  // top virtual surface
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar), Frame{FrameType::kNoFrame},
      builder::CreateLocalTransformation<Real_t>({0, 0, shell.GetUpperZ(), 0, 0, 0}));
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);

  builder::AddLogicToShell<Real_t>(logical_id, logic);

  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
