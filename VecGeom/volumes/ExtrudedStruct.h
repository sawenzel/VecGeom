/// @file ExtrudedStruct.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_EXTRUDED_STRUCT_H
#define VECGEOM_EXTRUDED_STRUCT_H

#include "VecGeom/base/Config.h"

#include "VecGeom/volumes/PolygonalShell.h"
#include "VecGeom/volumes/TessellatedStruct.h"
#include <vector>

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

  struct FacetInd {
    size_t ind1{0}, ind2{0}, ind3{0};

    FacetInd() = default;
    VECCORE_ATT_HOST_DEVICE
    FacetInd(size_t i1, size_t i2, size_t i3) : ind1(i1), ind2(i2), ind3(i3) {}
  };

  template <typename Facets_t>
  VECCORE_ATT_HOST_DEVICE
  void TriangulatePolygon(Facets_t &facets) const
  {
    const size_t nvertices = GetNVertices();
    vector_t<size_t> vtx;
    for (size_t i = 0; i < nvertices; ++i)
      vtx.push_back(i);

    size_t i1 = 0;
    size_t i2 = 1;
    size_t i3 = 2;

    while (vtx.size() > 2) {
      size_t counter = 0;
      while (!IsConvexSide(vtx[i1], vtx[i2], vtx[i3])) {
        i1 = (i1 + 1) % vtx.size();
        i2 = (i2 + 1) % vtx.size();
        i3 = (i3 + 1) % vtx.size();
        counter++;
        VECGEOM_VALIDATE(counter < nvertices, << "Triangulation failed");
        (void)counter; // silence unused variable warnings in release builds
      }

      bool good = true;
      for (auto i : vtx) {
        if (i == vtx[i1] || i == vtx[i2] || i == vtx[i3]) continue;
        if (IsPointInside(vtx[i1], vtx[i2], vtx[i3], i)) {
          good = false;
          i1   = (i1 + 1) % vtx.size();
          i2   = (i2 + 1) % vtx.size();
          i3   = (i3 + 1) % vtx.size();
          break;
        }
      }

      if (good) {
        facets.push_back(FacetInd(vtx[i1], vtx[i2], vtx[i3]));
        vtx.erase(vtx.begin() + i2);
        i1 = 0;
        i2 = 1;
        i3 = 2;
      }
    }
  }

public:
  struct DeprecatedTslHelper {
    // TODO(VecGeom-release-transition): remove these public compatibility fields
    // after a VecGeom release has given Geant4 time to switch to GetMeshHelper().
    // They emulate the old ExtrudedStruct::fTslHelper.fVertices/fFacets[*]->fIndices
    // export layout only; navigation must use fTslRuntimeHelper instead.
    vector_t<Vector3D<Precision>> fVertices;
    vector_t<TriangleFacet<Precision>> fFacetStorage;
    vector_t<TriangleFacet<Precision> *> fFacets;

    // TODO(VecGeom-release-transition): remove together with the compatibility
    // fields above. New visualization/export code should use GetMeshHelper().
    VECCORE_ATT_HOST_DEVICE
    void FillFrom(TessellatedStruct<3, Precision> const &tsl)
    {
      fVertices.clear();
      fFacetStorage.clear();
      fFacets.clear();

      fVertices.reserve(tsl.fVertices.size());
      for (size_t i = 0; i < tsl.fVertices.size(); ++i)
        fVertices.push_back(tsl.fVertices[i]);

      fFacetStorage.reserve(tsl.fFacets.size());
      fFacets.reserve(tsl.fFacets.size());
      for (size_t i = 0; i < tsl.fFacets.size(); ++i) {
        const auto *facet = tsl.fFacets[i];
        fFacetStorage.push_back(*facet);
        fFacets.push_back(&fFacetStorage[fFacetStorage.size() - 1]);
      }
    }
  };

  class ExtrudedMeshHelper {
  private:
    ExtrudedStruct const &fXtru;
    std::vector<FacetInd> fCapFacets;

    size_t MeshVertexIndex(size_t ivert, size_t isect) const;

  public:
    // Temporary API view for visualization/export code; navigation keeps using fTslRuntimeHelper.
    explicit ExtrudedMeshHelper(ExtrudedStruct const &xtru);

    size_t GetNvertices() const;

    Vector3D<Precision> GetVertex(size_t index) const;

    size_t GetNfacets() const;

    void GetFacetVertices(size_t ifacet, size_t (&indices)[3]) const;
  };

  bool fIsSxtru                  = false;     ///< Flag for sxtru representation
  bool fInitialized              = false;     ///< Flag for initialization
  Precision *fZPlanes            = nullptr;   ///< Z position of planes
  mutable Precision fCubicVolume = 0.;        ///< Cubic volume
  mutable Precision fSurfaceArea = 0.;        ///< Surface area
  PolygonalShell fSxtruHelper;                ///< Sxtru helper
  TessellatedRuntimeStruct<Precision> fTslRuntimeHelper; ///< The tessellated helper for navigation
  // TODO(VecGeom-release-transition): remove this old-layout export shim after
  // a VecGeom release. Geant4 and other visualization/export users should use
  // GetMeshHelper(), not the internal tessellated navigation representation.
  DeprecatedTslHelper fTslHelper;
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
    VECGEOM_ASSERT(nsections > 1 && nvertices > 2);
    fZPlanes         = new Precision[nsections];
    fZPlanes[0]      = sections[0].fOrigin.z();
    bool degenerated = false;
    for (size_t i = 1; i < (size_t)nsections; ++i) {
      fZPlanes[i] = sections[i].fOrigin.z();
      // Make sure sections are defined in increasing order
      VECGEOM_VALIDATE(fZPlanes[i] >= fZPlanes[i - 1], << "Extruded sections not defined in increasing Z order");
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
    TessellatedStruct<3, Precision> tsl_builder_struct;

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
    VectorBase<FacetInd> facets(nvertices);
    // TRIANGULATE POLYGON
    TriangulatePolygon(facets);
    // We have all index facets, create now the real facets
    // Bottom (normals pointing down)
    for (size_t i = 0; i < facets.size(); ++i) {
      size_t i1 = facets[i].ind1;
      size_t i2 = facets[i].ind2;
      size_t i3 = facets[i].ind3;
      tsl_builder_struct.AddTriangularFacet(VertexToSection(i1, 0), VertexToSection(i2, 0), VertexToSection(i3, 0));
    }
    // Sections
    for (size_t isect = 0; isect < nsections - 1; ++isect) {
      for (size_t i = 0; i < (size_t)nvertices; ++i) {
        size_t j = (i + 1) % nvertices;
        // Quadrilateral isect:(j, i)  isect+1: (i, j)
        tsl_builder_struct.AddQuadrilateralFacet(VertexToSection(j, isect), VertexToSection(i, isect),
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
      size_t i1 = facets[i].ind1;
      size_t i2 = facets[i].ind2;
      size_t i3 = facets[i].ind3;
      tsl_builder_struct.AddTriangularFacet(VertexToSection(i1, nsections - 1), VertexToSection(i3, nsections - 1),
                                            VertexToSection(i2, nsections - 1));
    }
    // Now close the tessellated structure
    tsl_builder_struct.Close();
    fTslHelper.FillFrom(tsl_builder_struct);
    fTslRuntimeHelper.InitFrom(tsl_builder_struct);
#ifndef VECGEOM_ENABLE_CUDA
    if (getenv("NOTSLSECTIONS")) {
      // convenience mode to compare tsl sections against pure tessellated
      fUseTslSections = false;
    }
#endif
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

  ExtrudedMeshHelper GetMeshHelper() const { return ExtrudedMeshHelper(*this); }

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
  bool IsConvexSide(size_t i0, size_t i1, size_t i2) const
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

inline size_t ExtrudedStruct::ExtrudedMeshHelper::MeshVertexIndex(size_t ivert, size_t isect) const
{
  return isect * fXtru.GetNVertices() + ivert;
}

inline ExtrudedStruct::ExtrudedMeshHelper::ExtrudedMeshHelper(ExtrudedStruct const &xtru) : fXtru(xtru)
{
  const size_t nvertices = fXtru.GetNVertices();
  // Rebuild only the 2D cap triangulation; 3D vertices are redirected through VertexToSection().
  VectorBase<FacetInd> capFacets(nvertices);
  fXtru.TriangulatePolygon(capFacets);
  fCapFacets.reserve(capFacets.size());
  for (size_t i = 0; i < capFacets.size(); ++i)
    fCapFacets.push_back(capFacets[i]);
}

inline size_t ExtrudedStruct::ExtrudedMeshHelper::GetNvertices() const
{
  return fXtru.GetNVertices() * fXtru.GetNSections();
}

inline Vector3D<Precision> ExtrudedStruct::ExtrudedMeshHelper::GetVertex(size_t index) const
{
  const size_t nvertices = fXtru.GetNVertices();
  return fXtru.VertexToSection(index % nvertices, index / nvertices);
}

inline size_t ExtrudedStruct::ExtrudedMeshHelper::GetNfacets() const
{
  const size_t nsideFacets = 2 * fXtru.GetNVertices() * (fXtru.GetNSections() - 1);
  return 2 * fCapFacets.size() + nsideFacets;
}

inline void ExtrudedStruct::ExtrudedMeshHelper::GetFacetVertices(size_t ifacet, size_t (&indices)[3]) const
{
  const size_t nvertices     = fXtru.GetNVertices();
  const size_t nsections     = fXtru.GetNSections();
  const size_t ncapFacets    = fCapFacets.size();
  const size_t nsideFacets   = 2 * nvertices * (nsections - 1);
  const size_t firstTopFacet = ncapFacets + nsideFacets;

  if (ifacet < ncapFacets) {
    const auto &facet = fCapFacets[ifacet];
    indices[0]        = MeshVertexIndex(facet.ind1, 0);
    indices[1]        = MeshVertexIndex(facet.ind2, 0);
    indices[2]        = MeshVertexIndex(facet.ind3, 0);
    return;
  }

  if (ifacet < firstTopFacet) {
    // Match the side-face split used by the navigation tessellation: (j,i,i+1) and (j,i+1,j+1).
    const size_t sideFacet = ifacet - ncapFacets;
    const size_t isect     = sideFacet / (2 * nvertices);
    const size_t rem       = sideFacet % (2 * nvertices);
    const size_t i         = rem / 2;
    const size_t j         = (i + 1) % nvertices;
    const bool secondTri   = (rem % 2) != 0;

    indices[0] = MeshVertexIndex(j, isect);
    indices[1] = secondTri ? MeshVertexIndex(i, isect + 1) : MeshVertexIndex(i, isect);
    indices[2] = secondTri ? MeshVertexIndex(j, isect + 1) : MeshVertexIndex(i, isect + 1);
    return;
  }

  const auto &facet = fCapFacets[ifacet - firstTopFacet];
  indices[0]        = MeshVertexIndex(facet.ind1, nsections - 1);
  indices[1]        = MeshVertexIndex(facet.ind3, nsections - 1);
  indices[2]        = MeshVertexIndex(facet.ind2, nsections - 1);
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
