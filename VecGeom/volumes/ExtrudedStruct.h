/// @file ExtrudedStruct.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_EXTRUDED_STRUCT_H
#define VECGEOM_EXTRUDED_STRUCT_H

#include "VecGeom/base/Config.h"

#include "VecGeom/volumes/PolygonalShell.h"
#include "VecGeom/volumes/TessellatedStruct.h"

#ifndef VECGEOM_ENABLE_CUDA
#include "VecGeom/volumes/TessellatedSection.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class ExtrudedStruct;);
VECGEOM_DEVICE_DECLARE_CONV(class, ExtrudedStruct);
VECGEOM_DEVICE_DECLARE_CONV(struct, XtruVertex2);
VECGEOM_DEVICE_DECLARE_CONV(struct, XtruSection);

inline namespace VECGEOM_IMPL_NAMESPACE {

// Structure wrapping either a polygonal shell helper in case of two
// extruded sections or a tessellated structure in case of more

struct XtruVertex2 {
  Precision x;
  Precision y;
};

struct XtruSection {
  Vector3D<Precision> fOrigin; // Origin of the section
  Precision fScale;
};

class ExtrudedStruct {

  // template <typename U>
  // using vector_t = vecgeom::Vector<U>;
  template <typename U>
  using vector_t = vecgeom::Vector<U>;

public:
  bool fIsSxtru                  = false;     ///< Flag for sxtru representation
  bool fInitialized              = false;     ///< Flag for initialization
  Precision *fZPlanes            = nullptr;   ///< Z position of planes
  mutable Precision fCubicVolume = 0.;        ///< Cubic volume
  mutable Precision fSurfaceArea = 0.;        ///< Surface area
  PolygonalShell fSxtruHelper;                ///< Sxtru helper
  TessellatedStruct<3, Precision> fTslHelper; ///< Tessellated helper
#ifndef VECGEOM_ENABLE_CUDA
  bool fUseTslSections = false;                           ///< Use tessellated section helper
  vector_t<TessellatedSection<Precision> *> fTslSections; ///< Tessellated sections
#endif
  vector_t<XtruVertex2> fVertices; ///< Polygone vertices
  vector_t<XtruSection> fSections; ///< Vector of sections
  PlanarPolygon fPolygon;          ///< Planar polygon

public:
  /** @brief Dummy constructor */
  VECCORE_ATT_HOST_DEVICE
  ExtrudedStruct() {}

  /** @brief Constructor providing polygone vertices and sections */
  VECCORE_ATT_HOST_DEVICE
  ExtrudedStruct(int nvertices, XtruVertex2 const *vertices, int nsections, XtruSection const *sections)
  {
    Initialize(nvertices, vertices, nsections, sections);
  }

  // Constructor used during Specialization for nsections == 2
  VECCORE_ATT_HOST_DEVICE
  ExtrudedStruct(size_t nvertices, const Precision *x, const Precision *y, Precision zmin, Precision zmax)
  {
    XtruVertex2 *vertices = new XtruVertex2[nvertices];
    XtruSection *sections = new XtruSection[2];
    for (size_t i = 0; i < nvertices; ++i) {
      vertices[i].x = x[i];
      vertices[i].y = y[i];
    }

    sections[0].fScale = 1.;
    sections[0].fOrigin.Set(0., 0., zmin);

    sections[1].fScale = 1.;
    sections[1].fOrigin.Set(0., 0., zmax);

    Initialize(nvertices, vertices, 2, sections);
    delete[] vertices;
    delete[] sections;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  int FindZSegment(Precision const &pointZ) const
  {
    int index              = -1;
    Precision const *begin = fZPlanes;
    Precision const *end   = fZPlanes + fSections.size() + 1;
    while (begin < end - 1 && pointZ - kTolerance > *begin) {
      ++index;
      ++begin;
    }
    if (pointZ + kTolerance > *begin) return (index + 1);
    return index;
  }

  /** @brief Initialize based on vertices and sections */
  void Initialize(int nvertices, XtruVertex2 const *vertices, int nsections, XtruSection const *sections)
  {
    if (fInitialized) return;
    assert(nsections > 1 && nvertices > 2);
    fZPlanes         = new Precision[nsections];
    fZPlanes[0]      = sections[0].fOrigin.z();
    bool degenerated = false;
    for (size_t i = 1; i < (size_t)nsections; ++i) {
      fZPlanes[i] = sections[i].fOrigin.z();
      // Make sure sections are defined in increasing order
      assert(fZPlanes[i] >= fZPlanes[i - 1] && "Extruded sections not defined in increasing Z order");
      if (fZPlanes[i] - fZPlanes[i - 1] < kTolerance) degenerated = true;
    }
#ifndef VECGEOM_ENABLE_CUDA
    if (!degenerated) fUseTslSections = true;
#endif
    (void)degenerated; // silence the compiler
    // Check if this is an SXtru
    if (nsections == 2 && (sections[0].fOrigin - sections[1].fOrigin).Perp2() < kTolerance &&
        vecCore::math::Abs(sections[0].fScale - sections[1].fScale) < kTolerance)
      fIsSxtru = true;
    if (fIsSxtru) {
      // Put vertices in arrays
      Precision *x = new Precision[nvertices];
      Precision *y = new Precision[nvertices];
      for (size_t i = 0; i < (size_t)nvertices; ++i) {
        x[i] = sections[0].fOrigin.x() + sections[0].fScale * vertices[i].x;
        y[i] = sections[0].fOrigin.y() + sections[0].fScale * vertices[i].y;
      }
      fSxtruHelper.Init(nvertices, x, y, sections[0].fOrigin[2], sections[1].fOrigin[2]);
      delete[] x;
      delete[] y;
    }
    // Create the tessellated structure in all cases
    CreateTessellated(nvertices, vertices, nsections, sections);
    fInitialized = true;
  }

  /** @brief Construct facets based on vertices and sections */
  VECCORE_ATT_HOST_DEVICE
  void CreateTessellated(size_t nvertices, XtruVertex2 const *vertices, size_t nsections, XtruSection const *sections)
  {
    struct FacetInd {
      size_t ind1{0}, ind2{0}, ind3{0};

      FacetInd() = default;
      FacetInd(int i1, int i2, int i3)
      {
        ind1 = i1;
        ind2 = i2;
        ind3 = i3;
      }
    };

    // Store sections
    for (size_t isect = 0; isect < nsections; ++isect)
      fSections.push_back(sections[isect]);

    // Create the polygon
    Precision *vx = new Precision[nvertices];
    Precision *vy = new Precision[nvertices];
    for (size_t i = 0; i < nvertices; ++i) {
      vx[i] = vertices[i].x;
      vy[i] = vertices[i].y;
    }
    fPolygon.Init(nvertices, vx, vy);
#ifndef VECGEOM_ENABLE_CUDA
    fUseTslSections &= fPolygon.IsConvex();
    if (fUseTslSections) {
      // Create tessellated sections
      fTslSections.reserve(nsections);
      for (size_t i = 0; i < nsections - 1; ++i) {
        fTslSections[i] =
            new TessellatedSection<Precision>(nvertices, sections[i].fOrigin.z(), sections[i + 1].fOrigin.z());
      }
    }
#endif
    // TRIANGULATE POLYGON

    VectorBase<FacetInd> facets(nvertices);
    // Fill a vector of vertex indices
    vector_t<size_t> vtx;
    for (size_t i = 0; i < nvertices; ++i)
      vtx.push_back(i);

    size_t i1 = 0;
    size_t i2 = 1;
    size_t i3 = 2;

    while (vtx.size() > 2) {
      // Find convex parts of the polygon (ears)
      size_t counter = 0;
      while (!IsConvexSide(vtx[i1], vtx[i2], vtx[i3])) {
        i1++;
        i2++;
        i3 = (i3 + 1) % vtx.size();
        counter++;
        assert(counter < nvertices && "Triangulation failed");
        (void)counter; // silence unused variable warnings in release builds
      }
      bool good = true;
      // Check if any of the remaining vertices are in the ear
      for (auto i : vtx) {
        if (i == vtx[i1] || i == vtx[i2] || i == vtx[i3]) continue;
        if (IsPointInside(vtx[i1], vtx[i2], vtx[i3], i)) {
          good = false;
          i1++;
          i2++;
          i3 = (i3 + 1) % vtx.size();
          break;
        }
      }

      if (good) {
        // Make triangle
        facets.push_back(FacetInd(vtx[i1], vtx[i2], vtx[i3]));
        // Remove the middle vertex of the ear and restart
        vtx.erase(vtx.begin() + i2);
        i1 = 0;
        i2 = 1;
        i3 = 2;
      }
    }
    // We have all index facets, create now the real facets
    // Bottom (normals pointing down)
    for (size_t i = 0; i < facets.size(); ++i) {
      i1 = facets[i].ind1;
      i2 = facets[i].ind2;
      i3 = facets[i].ind3;
      fTslHelper.AddTriangularFacet(VertexToSection(i1, 0), VertexToSection(i2, 0), VertexToSection(i3, 0));
    }
    // Sections
    for (size_t isect = 0; isect < nsections - 1; ++isect) {
      for (size_t i = 0; i < (size_t)nvertices; ++i) {
        size_t j = (i + 1) % nvertices;
        // Quadrilateral isect:(j, i)  isect+1: (i, j)
        fTslHelper.AddQuadrilateralFacet(VertexToSection(j, isect), VertexToSection(i, isect),
                                         VertexToSection(i, isect + 1), VertexToSection(j, isect + 1));
#ifndef VECGEOM_ENABLE_CUDA
        if (fUseTslSections)
          fTslSections[isect]->AddQuadrilateralFacet(VertexToSection(j, isect), VertexToSection(i, isect),
                                                     VertexToSection(i, isect + 1), VertexToSection(j, isect + 1));
#endif
      }
    }
    // Top (normals pointing up)
    for (size_t i = 0; i < facets.size(); ++i) {
      i1 = facets[i].ind1;
      i2 = facets[i].ind2;
      i3 = facets[i].ind3;
      fTslHelper.AddTriangularFacet(VertexToSection(i1, nsections - 1), VertexToSection(i3, nsections - 1),
                                    VertexToSection(i2, nsections - 1));
    }
    // Now close the tessellated structure
    fTslHelper.Close();
  }

  /** @brief Get the number of sections */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetNSections() const { return fSections.size(); }

  /** @brief Get the number of planes */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetNSegments() const { return (fSections.size() - 1); }

  /** @brief Get section i */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  XtruSection GetSection(int i) const { return fSections[i]; }

  /** @brief Get the number of vertices */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetNVertices() const { return fPolygon.GetNVertices(); }

  /** @brief Get the polygone vertex i */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void GetVertex(int i, Precision &x, Precision &y) const
  {
    x = fPolygon.GetVertices().x()[i];
    y = fPolygon.GetVertices().y()[i];
  }

  /** Return true if i is on the line through i1, i2 */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsSameLine(size_t i, size_t i1, size_t i2) const
  {
    const Precision *x = fPolygon.GetVertices().x();
    const Precision *y = fPolygon.GetVertices().y();
    if (x[i1] == x[i2]) return std::fabs(x[i] - x[i1]) < kTolerance * 0.5;

    Precision slope = (y[i2] - y[i1]) / (x[i2] - x[i1]);
    Precision predy = y[i1] + slope * (x[i] - x[i1]);
    Precision dy    = y[i] - predy;

    // Check perpendicular distance vs tolerance 'directly'
    const Precision tol = 0.5 * kTolerance;
    bool squareComp     = (dy * dy < (1 + slope * slope) * tol * tol);
    return squareComp;
  }

  /** @brief Return true if point i is on the line through i1, i2 and lies between i1 and i2 */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsSameLineSegment(size_t i, size_t i1, size_t i2) const
  {
    const Precision *x = fPolygon.GetVertices().x();
    const Precision *y = fPolygon.GetVertices().y();
    if (x[i] < std::min(x[i1], x[i2]) - kTolerance * 0.5 || x[i] > std::max(x[i1], x[i2]) + kTolerance * 0.5 ||
        y[i] < std::min(y[i1], y[i2]) - kTolerance * 0.5 || y[i] > std::max(y[i1], y[i2]) + kTolerance * 0.5)
      return false;

    return IsSameLine(i, i1, i2);
  }

  /** @brief Return true if i and j are on the same side of the line through i1, i2 */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsSameSide(size_t i, size_t j, size_t i1, size_t i2) const
  {
    const Precision *x = fPolygon.GetVertices().x();
    const Precision *y = fPolygon.GetVertices().y();

    return ((x[i] - x[i1]) * (y[i2] - y[i1]) - (x[i2] - x[i1]) * (y[i] - y[i1])) *
               ((x[j] - x[i1]) * (y[i2] - y[i1]) - (x[i2] - x[i1]) * (y[j] - y[i1])) >
           0;
  }

  /** Return true if i is inside of triangle (i1, i2, i3) or on its edges, else returns false */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsPointInside(size_t i1, size_t i2, size_t i3, size_t i) const
  {
    const Precision *x = fPolygon.GetVertices().x();
    const Precision *y = fPolygon.GetVertices().y();

    // Check extent first
    if ((x[i] < x[i1] && x[i] < x[i2] && x[i] < x[i3]) || (x[i] > x[i1] && x[i] > x[i2] && x[i] > x[i3]) ||
        (y[i] < y[i1] && y[i] < y[i2] && y[i] < y[i3]) || (y[i] > y[i1] && y[i] > y[i2] && y[i] > y[i3]))
      return false;

    bool inside = IsSameSide(i, i1, i2, i3) && IsSameSide(i, i2, i1, i3) && IsSameSide(i, i3, i1, i2);

    bool onEdge = IsSameLineSegment(i, i1, i2) || IsSameLineSegment(i, i2, i3) || IsSameLineSegment(i, i3, i1);

    return inside || onEdge;
  }

  /** @brief Check if the polygone segments (i0, i1) and (i1, i2) make a convex side */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsConvexSide(size_t i0, size_t i1, size_t i2)
  {
    const Precision *x = fPolygon.GetVertices().x();
    const Precision *y = fPolygon.GetVertices().y();
    Precision cross    = (x[i1] - x[i0]) * (y[i2] - y[i1]) - (x[i2] - x[i1]) * (y[i1] - y[i0]);
    return cross < 0.;
  }

  /** @brief Returns convexity of polygon */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsConvexPolygon() const { return fPolygon.IsConvex(); }

  /** @brief Returns the coordinates for a given vertex index at a given section */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> VertexToSection(size_t ivert, size_t isect) const
  {
    const Precision *x = fPolygon.GetVertices().x();
    const Precision *y = fPolygon.GetVertices().y();
    Vector3D<Precision> vert(fSections[isect].fOrigin[0] + fSections[isect].fScale * x[ivert],
                             fSections[isect].fOrigin[1] + fSections[isect].fScale * y[ivert],
                             fSections[isect].fOrigin[2]);
    return vert;
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
