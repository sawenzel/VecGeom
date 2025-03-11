///
/// \file Utils3D.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#ifndef VECGEOM_BASE_UTILS3D_H_
#define VECGEOM_BASE_UTILS3D_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Transformation3D.h"

#ifndef VECCORE_CUDA
#include <vector>
#else
#include "VecGeom/base/Vector.h"
#endif

/// A set of 3D geometry utilities
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

namespace Utils3D {

using Vec_t = Vector3D<Precision>;

template <typename T>
#ifndef VECCORE_CUDA
using vector_t = std::vector<T>;
#else
using vector_t = vecgeom::Vector<T>; // this has problems, like in the initializers Vector v = {...};
#endif

enum EPlaneXing_t { kParallel = 0, kIdentical, kIntersecting };

enum EBodyXing_t { kDisjoint = 0, kTouching, kOverlapping }; // do not change order

/// @brief A basic plane in Hessian normal form
struct Plane {
  Vector3D<Precision> fNorm; ///< Unit normal vector to plane
  Precision fDist = 0.;      ///< Distance to plane (positive if origin on same side as normal)

  Plane() : fNorm() {}

  Plane(Vector3D<Precision> const &norm, Precision dist)
  {
    fNorm = norm;
    fDist = dist;
  }

  /// @brief Transform the plane by a general transformation
  void Transform(Transformation3D const &tr);
};

/// Structure to hold line-line intersection results
///
/// In the case of fIntersect:
///                                    ^ lA.fPts[1]
///                                   /
///                               lA /
///                                 /
///                                /              lB
///         <---------------------X----------------------->
///  lB.fPts[0]                  /                          lB.fPts[1]
///                             /
///                            /    X = lA.fPts[0] + fA * (lA.fPts[1] - lA.fPts[0])
///                           /     X = lB.fPts[0] + fB * (lB.fPts[1] - lB.fPts[0])
///               lA.fPts[0] v
///
///
///
/// In the case of fOverlap:
///
///                     lB.fPts[0]                       lB.fPts[1]
///         <------------------<------------>------------->
///  lA.fPts[0]                              lA.fPts[1]
///
///
///                lB.fPts[0] = lA.fPts[0] + fA * (lA.fPts[1] - lA.fPts[0])
///                lB.fPts[1] = lA.fPts[0] + fB * (lA.fPts[1] - lA.fPts[0])
///
/// In the case of fNoIntersect and fParallel, fA and fB have no meaning.

struct LineIntersection {
  enum { fParallel, fOverlap, fIntersect, fNoIntersect } fType; ///< Intersection type
  Precision fA; ///< Used in calculating the intersection point (see the structure explanation)
  Precision fB; ///< Used in calculating the intersection point (see the structure explanation)
};

/// @brief A line segment
struct Line {
  Vector3D<Precision> fPts[2]; ///< Points defining the line

  /// Computes line-line intersection.
  LineIntersection *Intersect(const Line &lB);

  /// Indicates whether a point lies on the line segment defined by fPts
  bool IsPointOnLine(const Vec_t &p);
};

/// @brief A polygon defined by vertices and normal
/* The list of vertices is a reference to an external array. The used vertex indices have to be defined such
   that consecutive segments cross product is on the same side as the normal. */
struct Polygon {
  size_t fN       = 0;             ///< Number of vertices
  bool fConvex    = false;         ///< Convexity
  bool fHasNorm   = false;         ///< Normal is already supplied
  bool fValid     = false;         ///< Polygon is not degenerate
  Precision fDist = 0.;            ///< Distance to plane in the Hessian form
  Vec_t fNorm;                     ///< Unit normal vector to plane
  vector_t<Vec_t> *fVert{nullptr}; ///< Global list of vertices shared with other polygons
  vector_t<size_t> fInd;           ///< [fN] Indices of vertices
  vector_t<Vec_t> fSides;          ///< [fN] Side vectors

  Polygon() = default;
  /// @brief Constructor taking the number of vertices, a reference to a vector of vertices and the convexity
  VECCORE_ATT_HOST_DEVICE
  Polygon(size_t n, vector_t<Vec_t> &vertices, bool convex = false);

  // @brief Fast constructor with no checks in case of convex polygons, providing the normal vector (normalized)
  VECCORE_ATT_HOST_DEVICE
  Polygon(size_t n, vector_t<Vec_t> &vertices, Vec_t const &norm);

  /// Contructor with vertices and indices. Calls CheckAndFixDegenerate() on the polygon.
  Polygon(size_t n, vector_t<Vec_t> &vertices, vector_t<size_t> const &indices, bool convex);

  /// @brief Copy constructor
  Polygon(const Polygon &other) = default;

  /// @brief Assignment operator

  Polygon &operator=(const Polygon &other) = default;

  /// @brief Setter for a vertex index
  VECGEOM_FORCE_INLINE
  void SetVertex(size_t ind, size_t ivert) { fInd[ind] = ivert; }

  /// @brief Getter for a vertex
  VECGEOM_FORCE_INLINE
  Vec_t const &GetVertex(size_t i) const { return (*fVert)[fInd[i]]; }

  /// @brief Setter from an array of vertex indices
  template <typename T>
  void SetVertices(vector_t<T> indices)
  {
    for (size_t i = 0; i < fN; ++i)
      fInd[i] = size_t(indices[i]);
  }

  /// @brief Initialization is mandatory before first use
  void Init();

  /// @brief Transform the polygon by a general transformation
  void Transform(Transformation3D const &tr);

  /// Makes best effort to fix a degenerate polygon. In case it cannot be fixed, marks polygon as invalid.
  /// Only repeated vertices, and not collinear vertices are handled.
  void CheckAndFixDegenerate();

  /// Decomposes a polygon into triangles.
  /// @param[out] polys The resulting triangles.
  void TriangulatePolygon(vector_t<Polygon> &polys) const;

  /// Checks if fVert[i1] is a convex vertex in
  /// the polygon given its next neighbor fVert[i2] and previous neighbor fVert[i0].
  bool isConvexVertex(size_t i0, size_t i1, size_t i2) const;

  /// Checks if a point is inside the triangle formed by fVert[i0]-fVert[i1]-fVert[i2]
  /// Point should already be on the plane.
  bool isPointInsideTriangle(const Vec_t &point, size_t i0, size_t i1, size_t i2) const;

  /// Checks if a point is inside the polygon
  /// Point should already be on the plane.
  bool IsPointInside(const Vec_t &p) const;

#ifndef VECCORE_CUDA
  /// Computes the overlapping part of two polygons
  struct PolygonIntersection *Intersect(const Polygon &clipper);
#endif

  /// Retrieves the extent of the polygon
  /// @param[out] x Polygon boundaries in x axis, x[0] holding the smallest x value.
  /// @param[out] y Polygon boundaries in y axis, y[0] holding the smallest y value.
  /// @param[out] z Polygon boundaries in z axis, z[0] holding the smallest z value.
  void Extent(Precision x[2], Precision y[2], Precision z[2]);
};

/// @brief A simple polyhedron defined by vertices and polygons
struct Polyhedron {
  vector_t<Vec_t> fVert;    ///< Vector of vertices
  vector_t<Polygon> fPolys; ///< Vector of polygons

  /// @brief Constructors
  Polyhedron() = default;
  Polyhedron(size_t nvert, size_t npolys)
  {
    fVert.reserve(nvert);
    fPolys.reserve(npolys);
  }

  /// @brief Polyhedrons can be re-used
  VECGEOM_FORCE_INLINE
  void Reset(size_t nvert, size_t npolys)
  {
    fVert.reserve(nvert);
    fVert.clear();
    fPolys.reserve(npolys);
    fPolys.clear();
  }

  /// @brief Get number of vertices
  VECGEOM_FORCE_INLINE
  size_t GetNvertices() const { return fVert.size(); }

  /// @brief Get number of polygons
  VECGEOM_FORCE_INLINE
  size_t GetNpolygons() const { return fPolys.size(); }

  /// @brief Get a reference to a specific vertex
  VECGEOM_FORCE_INLINE
  Vec_t const &GetVertex(size_t i) const { return fVert[i]; }

  /// @brief Get a reference to a specific polygon
  VECGEOM_FORCE_INLINE
  Polygon const &GetPolygon(size_t i) const { return fPolys[i]; }

  /// @brief Transform the polygon by a general transformation
  void Transform(Transformation3D const &tr);

  /// Adds given polygon to this polyhedron
  void AddPolygon(Polygon &poly, bool triangulate);
};

/// Structure to hold the result of Polygon-Polygon intersection.
///
///
///              +------+         +------+   +-------+
///              |      |         |      |   |       |
///              |      |         |      |   |    xxxx----+
///              +------x------+  xxxxxxxx   |    x  x    |
///                     |      |  |      |   |    x  x    |
///                     |      |  |      |   |    xxxx----+
///                     +------+  +------+   +-------+
///                 point           line          polygon

struct PolygonIntersection {
  vector_t<Vec_t> fPoints;     ///< Resulting points from the intersection
  vector_t<Line> fLines;       ///< Resulting lines from the intersection
  vector_t<Polygon> fPolygons; ///< Resulting polygons from the intersection
  vector_t<Vec_t> fVertices;   ///< Resulting vertices in case there are polygons
};

#ifndef VECCORE_CUDA
/// @brief Streamers
std::ostream &operator<<(std::ostream &os, Plane const &hpl);
std::ostream &operator<<(std::ostream &os, Polygon const &poly);
std::ostream &operator<<(std::ostream &os, Polyhedron const &polyh);
#endif

/// @brief Function to find the crossing line between two planes.
EPlaneXing_t PlaneXing(Plane const &pl1, Plane const &pl2, Vector3D<Precision> &point, Vector3D<Precision> &direction);

// @brief Function to find if 2 arbitrary polygons cross each other.
EBodyXing_t PolygonXing(Polygon const &poly1, Polygon const &poly2, Line *line = nullptr);

// @brief Function to find if 2 arbitrary polyhedrons cross each other.
EBodyXing_t PolyhedronXing(Polyhedron const &poly1, Polyhedron const &poly2, vector_t<Line> &lines);

/// @brief Function to determine crossing of two arbitrary placed boxes
/** The function takes the box parameters and their transformations in a common frame.
    A fast check is performed if both transformations are identity. */
EBodyXing_t BoxCollision(Vector3D<Precision> const &box1, Transformation3D const &tr1, Vector3D<Precision> const &box2,
                         Transformation3D const &tr2);

/// @brief Function filling a polyhedron as a box
void FillBoxPolyhedron(Vec_t const &dimensions, Polyhedron &polyh);

} // namespace Utils3D
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
#endif
