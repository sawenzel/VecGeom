// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the data structure for the tessellated shape.
/// \file volumes/TessellatedStruct.h
/// \author First version created by Mihaela Gheata (CERN/ISS)

#ifndef VECGEOM_VOLUMES_TESSELLATEDSTRUCT_H_
#define VECGEOM_VOLUMES_TESSELLATEDSTRUCT_H_

#include "TessellatedCluster.h"
#include "VecGeom/base/BitSet.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Config.h"
#include <vector>

#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/management/HybridManager2.h"
#include "VecGeom/navigation/HybridNavigator2.h"

#ifdef VECGEOM_EMBREE
#include "VecGeom/management/EmbreeManager.h"
#include "VecGeom/navigation/EmbreeNavigator.h"
#endif

#include "VecGeom/management/ABBoxManager.h"

namespace vecgeom {

// VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, TessellatedStruct, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

// Structure used for vectorizing queries on groups of triangles

#ifdef VECGEOM_EMBREE
#define USEEMBREE 1
#endif

/** Templated class holding the data structures for the tessellated solid.

  The class is templated on the number of edges for the composing facets and on the floating precision type
  used to represent the data and perform all calculations. It provides API for:
  - Adding triangular or quadrilateral facets
  - Initializing internally all data structures after adding all the facets that compose a closed tessellated
  surface. This creates a temporary helper data structure of the type GridHelper, used to clusterize facets in
  groups having as many elements as the vector size. The clusters are then used to construct a special navigation
  acceleration structure.
  - retrieving the hit clusters and the hit facets selected during navigation
*/
template <size_t NVERT, typename T = double>
class TessellatedStruct {

#ifndef VECGEOM_ENABLE_CUDA
  using Real_v = vecgeom::VectorBackend::Real_v;
#else
  using Real_v        = vecgeom::ScalarBackend::Real_v;
#endif
  using Facet_t = Tile<NVERT, T>;

  // Here we should be able to use vecgeom::Vector
  template <typename U>
  using vector_t = vecgeom::Vector<U>;

  using BVHStructure = HybridManager2::HybridBoxAccelerationStructure;

#ifdef USEEMBREE
  using BVHStructure2 = EmbreeManager::EmbreeAccelerationStructure;
#else
  using BVHStructure2 = HybridManager2::HybridBoxAccelerationStructure; // EmbreeManager::EmbreeAccelerationStructure;
#endif

  /** Structure representing a cell of a uniform grid embedding the tessellated solid bounding volume.

  The cell just stores an array of indices for the facets intersecting it. The cell coordinates are known by the
  GridHelper data structure owning it.
  */
  struct GridCell {
    vector_t<int> fArray; ///< Array of facet indices
    bool fUsed = false;   ///< Flag for cell usage

    /// Default constructor for a grid cell.
    VECCORE_ATT_HOST_DEVICE
    GridCell()
    { /* fArray.reserve(4); */
    }
  };

  /** Helper structure representing a grid of equal cells dividing the bounding box of a tessellated solid.

  The grid helper is defined by the number of cells in X/Y/Z. It provides methods to retrieve the cell
  containing a space point, or associated to a triplet of indices on (x, y, z). The helper is used by the
  TessellatedStruct for the facet clusterization.
  */
  //__________________________________________________________________________
  struct GridHelper {
    int fNgrid       = 0;           ///< Grid size
    int fNcells      = 0;           ///< Number of cells in the grid
    int fNcached     = 0;           ///< number of cached cells
    GridCell **fGrid = nullptr;     ///< Grid for clustering facets
    Vector3D<T> fMinExtent;         ///< Minimum extent
    Vector3D<T> fMaxExtent;         ///< Maximum extent
    Vector3D<T> fInvExtSize;        ///< Inverse extent size
    vector_t<Vector3D<T>> fAllVert; ///< Full list of vertices

    /// Default constructor for the grid helper structure.
    GridHelper() {}

    /// Destructor of the grid helper, deleting the cells and their content.
    ~GridHelper()
    {
      if (fGrid) {
        for (int i = 0; i < fNcells; ++i)
          delete fGrid[i];
        delete[] fGrid;
      }
    }

    /// Create all the cells corresponding to a given grid division number.
    /** @param ngrid Number of cells on each axis.*/
    void CreateCells(int ngrid)
    {
      if (fNgrid) return;
      fNgrid  = ngrid;
      fNcells = ngrid * ngrid * ngrid;
      fGrid   = new GridCell *[fNcells];
      for (int i = 0; i < fNcells; ++i)
        fGrid[i] = new GridCell();
    }

    /// Clears the content of all cells.
    VECCORE_ATT_HOST_DEVICE
    VECGEOM_FORCE_INLINE
    void ClearCells()
    {
      for (int icell = 0; icell < fNcells; ++icell)
        fGrid[icell]->fArray.clear();
    }

    /// Retrive a cell by its index triplet on (x,y,z).
    /** @param ind array of cell indices on (x,y,z)
        @return GridCell corresponding to the given indices
    */
    VECCORE_ATT_HOST_DEVICE
    VECGEOM_FORCE_INLINE
    GridCell *GetCell(int ind[3]) { return fGrid[fNgrid * fNgrid * ind[0] + fNgrid * ind[1] + ind[2]]; }

    /// Retreive a cell containing a space point and fill its indices triplet.
    /** @param[in]  point Space point
        @param[out] ind   Triplet of indices of the cell containing the point
        @return           Grid cell containing the space point
    */
    VECCORE_ATT_HOST_DEVICE
    VECGEOM_FORCE_INLINE
    GridCell *GetCell(Vector3D<T> const &point, int ind[3])
    {
      Vector3D<T> ratios = (point - fMinExtent) * fInvExtSize;
      for (int i = 0; i < 3; ++i) {
        ind[i] = ratios[i] * fNgrid;
        ind[i] = vecCore::math::Max(ind[i], 0);
        ind[i] = vecCore::math::Min(ind[i], fNgrid - 1);
      }
      return (GetCell(ind));
    }
  };

public:
  bool fSolidClosed   = false;          ///< Closure of the solid
  T fCubicVolume      = 0;              ///< Cubic volume
  T fSurfaceArea      = 0;              ///< Surface area
  GridHelper *fHelper = nullptr;        ///< Grid helper
  Vector3D<T> fMinExtent;               ///< Minimum extent
  Vector3D<T> fMaxExtent;               ///< Maximum extent
  Vector3D<T> fInvExtSize;              ///< Inverse extent size
  Vector3D<T> fTestDir;                 ///< Test direction for Inside function
  BVHStructure *fNavHelper   = nullptr; ///< Navigation helper using bounding boxes
  BVHStructure2 *fNavHelper2 = nullptr; ///< Navigation helper using bounding boxes

  // Here we have a pointer to the aligned bbox structure
  // ABBoxanager *fABBoxManager;

  vector_t<int> fCluster;                                  ///< Cluster of facets storing just the indices
  vector_t<int> fCandidates;                               ///< Candidates for the current cluster
  vector_t<Vector3D<T>> fVertices;                         ///< Vector of unique vertices
  vector_t<Facet_t *> fFacets;                             ///< Vector of triangular facets
  vector_t<TessellatedCluster<NVERT, Real_v> *> fClusters; ///< Vector of facet clusters
  BitSet *fSelected          = nullptr;                    ///< Facets already in clusters
  int fNcldist[kVecSize + 1] = {0};                        ///< Distribution of number of cluster size

private:
  /// Creates the navigation acceleration structure based ob the pre-computed clusters of facets.
  void CreateABBoxes()
  {
    using Boxes_t           = ABBoxManager<Precision>::ABBoxContainer_t;
    using BoxCorner_t       = ABBoxManager<Precision>::ABBox_s;
    int nclusters           = fClusters.size();
    BoxCorner_t *boxcorners = new BoxCorner_t[2 * nclusters];
    for (int i = 0; i < nclusters; ++i) {
      boxcorners[2 * i]     = fClusters[i]->fMinExtent;
      boxcorners[2 * i + 1] = fClusters[i]->fMaxExtent;
    }
    Boxes_t boxes = &boxcorners[0];
    fNavHelper    = HybridManager2::Instance().BuildStructure(boxes, nclusters);
#ifdef USEEMBREE
    fNavHelper2 = EmbreeManager::Instance().BuildStructureFromBoundingBoxes(boxes, nclusters);
#else
    fNavHelper2 = fNavHelper;
#endif
  }

public:
  /// Default constructor.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  TessellatedStruct()
  {
    fMinExtent = InfinityLength<T>();
    fMaxExtent = -InfinityLength<T>();
    fHelper    = new GridHelper();
  }

  /// Destructor.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  ~TessellatedStruct()
  {
    delete fHelper;
    if (fSelected) BitSet::ReleaseInstance(fSelected);
  }

  /// Adds a pre-defined facet and re-computes the extent.
  /** The vertices are added to the list of all vertices (including duplications) and the extent
    is re-adjusted.
    @param facet Pre-computed facet to be added
  */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void AddFacet(Facet_t *facet)
  {
    using vecCore::math::Max;
    using vecCore::math::Min;

    fFacets.push_back(facet);
    int ind = fHelper->fAllVert.size();
    // Add the vertices
    for (size_t i = 0; i < NVERT; ++i) {
      fHelper->fAllVert.push_back(facet->fVertices[i]);
      facet->fIndices[i] = ind + i;
      fMinExtent[0]      = Min(fMinExtent[0], facet->fVertices[i].x());
      fMinExtent[1]      = Min(fMinExtent[1], facet->fVertices[i].y());
      fMinExtent[2]      = Min(fMinExtent[2], facet->fVertices[i].z());
      fMaxExtent[0]      = Max(fMaxExtent[0], facet->fVertices[i].x());
      fMaxExtent[1]      = Max(fMaxExtent[1], facet->fVertices[i].y());
      fMaxExtent[2]      = Max(fMaxExtent[2], facet->fVertices[i].z());
    }
  }

  /// Add a non-duplicated vertex to the solid.
  /** Duplications are only checked in the grid cell containing the vertex. An index to the unique vertex is
    added to the cell, while the vertex position is added to the list fVertices.
    @param  vertex Space point to be added as vertex.
    @return Unique vertex id after removing duplicates.
  */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int AddVertex(Vector3D<T> const &vertex)
  {
    // Get the cell in which to add the vertex
    int ind[3];
    GridCell *cell = fHelper->GetCell(vertex, ind);
    // Loop existing vertices in the cell and check for a duplicate
    constexpr Precision tolerancesq = kTolerance * kTolerance;
    for (int ivert : cell->fArray) {
      // existing vertex?
      if ((fVertices[ivert] - vertex).Mag2() < tolerancesq) return ivert;
    }
    // Push new vertex into the tessellated structure
    int ivertnew = fVertices.size();
    fVertices.push_back(vertex);
    // Update the cell with the new vertex index
    cell->fArray.push_back(ivertnew);
    return ivertnew;
  }

  /// Retrieval of the extent of the tessellated structure.
  /** @param[out] amin Box corner having minimum coordinates
      @param[out] amax Box corner having maximum coordinates
  */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void Extent(Vector3D<T> &amin, Vector3D<T> &amax) const
  {
    amin = fMinExtent;
    amax = fMaxExtent;
  }

  /// Method performing all tasks for initializing and closing the data structures.

  /** The following sequence of operations is executed:
    - Creation of the grid of cells
    - Removal of duplicate facet indices and update of facets with  the unique vertex indices
    - Filling the cells with facet indices crossing them and creation of clusters
    - Creation of the navigation acceleration structures
  */
  void Close()
  {
    int ind[3];
    fInvExtSize = fMaxExtent - fMinExtent;
    if (fInvExtSize[0] * fInvExtSize[1] * fInvExtSize[2] < kTolerance) {
      std::cerr << "Tessellated structure is flat - not allowed\n";
      return;
    }
    fInvExtSize          = 1. / fInvExtSize;
    fHelper->fMinExtent  = fMinExtent;
    fHelper->fMaxExtent  = fMaxExtent;
    fHelper->fInvExtSize = fInvExtSize;
    // Make a grid with ~kVecSize facets per cell
    int ngrid = 1 + size_t(vecCore::math::Pow<T>(T(fFacets.size()), 1. / 3.));
    // Stopwatch timer;
    // timer.Start();
    fHelper->CreateCells(ngrid);
    // auto time = timer.Stop();
    // std::cerr << "CreateCells: " << time << " sec\n";

    // Loop over facets and their vertices, fill list of vertices free of
    // duplications.
    // timer.Start();
    for (auto facet : fFacets) {
      for (int ivert = 0; ivert < 3; ++ivert) {
        facet->fIndices[ivert] = AddVertex(facet->fVertices[ivert]);
      }
    }
    // time = timer.Stop();
    // std::cerr << "Remove duplicates: " << time << " sec\n";

    // Clear vertices and store facet indices in the grid helper
    // timer.Start();
    fHelper->ClearCells();
    unsigned ifacet = 0;
    for (auto facet : fFacets) {
      fHelper->GetCell(facet->fCenter, ind)->fArray.push_back(ifacet);
      //      for (int ivert = 0; ivert < 3; ++ivert) {
      //        fHelper->GetCell(facet->fVertices[ivert], ind)->fArray.push_back(ifacet);
      //      }
      ifacet++;
    }
    // time = timer.Stop();
    // std::cerr << "Store facets into grid: " << time << " sec\n";

    // Make clusters
    // timer.Start();
    //    std::cerr << "=== Using dummy clusters\n";
    //    CreateDummyClusters();

    const int nfacets = fFacets.size();
    fSelected         = BitSet::MakeInstance(nfacets);
    fSelected->ResetAllBits();
    fCandidates.clear();
    ifacet = 0;
    fCandidates.push_back(ifacet);
    fSelected->SetBitNumber(ifacet);
    TessellatedCluster<NVERT, Real_v> *cluster;
    while (fCandidates.size()) {
      // Use existing candidates in fCandidates to create the cluster
      cluster = CreateCluster();
      if (!cluster) cluster = MakePartialCluster();
      fClusters.push_back(cluster);
      // Fill cluster from the same cell or from a neighbor cell
      if (!fCandidates.size()) {
        ifacet = fSelected->FirstNullBit();
        if (ifacet == fFacets.size()) break;
        fCandidates.push_back(ifacet);
        fSelected->SetBitNumber(ifacet);
      }
    }

    // time = timer.Stop();
    // std::cerr << "Clusterizer: " << time << " sec\n";

    // Create navigation helper to be used in TessellatedImplementation
    // timer.Start();
    CreateABBoxes(); // to navigate, see: TestHybridBVH.cpp/HybridNavigator2.h/HybridSafetyEstimator.h
    // time = timer.Stop();
    // std::cerr << "Create AABoxes: " << time << " sec\n";
    // Generate random direction non-parallel to any of the surfaces
    constexpr T tolerance(1.e-8);
    while (1) {
      RandomDirection(fTestDir);
      // Loop over triangles and check that dot product is not close to 0
      for (auto facet : fFacets) {
        if (vecCore::math::Abs(facet->fNormal.Dot(fTestDir)) < tolerance) break;
      }
      break;
    }
    fSolidClosed = true;
  }

  /// Generate and store a random direction used for Contains and Inside navigation queries.
  /** @param[out] direction Random direction generated */
  void RandomDirection(Vector3D<Precision> &direction)
  {
    Precision phi   = RNG::Instance().uniform(0., 2. * kPi);
    Precision theta = std::acos(1. - 2. * RNG::Instance().uniform(0, 1));
    direction.x()   = std::sin(theta) * std::cos(phi);
    direction.y()   = std::sin(theta) * std::sin(phi);
    direction.z()   = std::cos(theta);
  }

  /// Loop over facets and group them in clusters in the order of definition.
  void CreateDummyClusters()
  {
    TessellatedCluster<NVERT, Real_v> *tcl = nullptr;
    int i                                  = 0;
    int j                                  = 0;
    for (auto facet : fFacets) {
      i = i % kVecSize;
      if (i == 0) {
        if (tcl) fClusters.push_back(tcl);
        tcl = new TessellatedCluster<NVERT, Real_v>();
      }
      tcl->AddFacet(i++, facet, j++);
    }
    // The last cluster may not be yet full
    for (; i < kVecSize; ++i)
      tcl->AddFacet(i, tcl->fFacets[0], tcl->fIfacets[0]);
  }

  /// Create the next cluster of closest neighbour facets using the GridHelper structure.
  /** @return Created cluster of neighbour facets */
  TessellatedCluster<NVERT, Real_v> *CreateCluster()
  {
    // Create cluster starting from fCandidates list
    unsigned nfacets = 0;
    VECGEOM_ASSERT(fCandidates.size() > 0); // call the method with at least one candidate in the list
    constexpr int rankmax = 3;      // ??? how to determine an appropriate value ???
    int rank              = 0;
    fCluster.clear();
    int ifacet = fCandidates[0];
    fCluster.push_back(ifacet);
    while (fCandidates.size() < kVecSize && rank < rankmax) {
      GatherNeighborCandidates(fCandidates[0], rank++);
    }
    if (fCandidates.size() < kVecSize) return nullptr;
    fCandidates.erase(fCandidates.begin());
    // Add facets with maximum neighborhood weight to existing cluster
    int iref = 0;
    while (nfacets < kVecSize) {
      nfacets = AddCandidatesToCluster(4 >> iref++); // 4 common vertices, 2, 1 or none
    }

    // The cluster is now complete, create a tessellated cluster object
    TessellatedCluster<NVERT, Real_v> *tcl = new TessellatedCluster<NVERT, Real_v>();
    int i                                  = 0;
    for (auto ifct : fCluster) {
      Facet_t *facet = fFacets[ifct];
      tcl->AddFacet(i++, facet, ifct);
    }
    fNcldist[fCluster.size()]++;
    return tcl;
  }

  /// Create partial cluster starting from fCandidates list
  /** @return Created cluster*/
  TessellatedCluster<NVERT, Real_v> *MakePartialCluster()
  {
    for (auto ifacet : fCandidates) {
      fCluster.push_back(ifacet);
    }
    fNcldist[fCluster.size()]++;
    fCandidates.clear();
    while (fCluster.size() < kVecSize)
      fCluster.push_back(fCluster[0]);

    // The cluster is now complete, create a tessellated cluster object
    TessellatedCluster<NVERT, Real_v> *tcl = new TessellatedCluster<NVERT, Real_v>();
    int i                                  = 0;
    for (auto ifacet : fCluster) {
      Facet_t *facet = fFacets[ifacet];
      tcl->AddFacet(i++, facet, ifacet);
    }
    return tcl;
  }

  /// Add candidates to and existing cluster.
  /** A neighbourhood weight is computed for each facet candidate according the number of vertices
    contained in grid cells already occupied by the cluster.
    @param  weightmin Minimum accepted weight
    @return New size of the cluster
  */
  int AddCandidatesToCluster(int weightmin)
  {
    // Add all candidate having the required weight to the cluster, until cluster
    // is complete
    for (auto it = fCandidates.begin(); it != fCandidates.end() && fCluster.size() < kVecSize;) {
      int weight = NeighborToCluster(*it);
      if (weight >= weightmin) {
        fCluster.push_back(*it);
        it = fCandidates.erase(it);
      } else {
        ++it;
      }
    }
    return fCluster.size();
  }

  /// Compute the weight of neighbourhood of a facet with respect to a cluster.
  /** @param ifacet Facet index
      @return Number of facet vertices contained by grid cells occupied by the cluster
  */
  int NeighborToCluster(int ifacet)
  {
    Facet_t *facet = fFacets[ifacet];
    int weight     = 0;
    for (auto icand : fCluster) {
      Facet_t *other = fFacets[icand];
      weight += facet->IsNeighbor(*other);
    }
    return weight;
  }

  /// Add all facets touching a cell to the list of candidates to be added to the next cluster.
  /** @param ind Triplet of cell indices on (x,y,z) */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void AddCandidatesFromCell(int ind[3])
  {
    GridCell *cell = fHelper->GetCell(ind);
    if (cell->fUsed) return;
    for (auto neighbor : cell->fArray) {
      if (!fSelected->TestBitNumber(neighbor)) {
        fSelected->SetBitNumber(neighbor);
        fCandidates.push_back(neighbor);
      }
    }
    cell->fUsed = true;
  }

  /// Gather candidates from cells neighboring the cell containing the center of the facet
  /** @param  ifacet Facet index for which neighbours are searched
      @param  rank   Maximum distance between the reference cell containing the center of the facet
        and the other cells where neighbours are looked for
      @return Size of the list of candidates.
  */
  int GatherNeighborCandidates(int ifacet, int rank)
  {
    int ind0[3], ind[3];
    const Facet_t *facet = fFacets[ifacet];
    fHelper->GetCell(facet->fCenter, ind0);
    if (rank == 0) {
      AddCandidatesFromCell(ind0);
      return (fCandidates.size());
    }

    // Establish cell index limits for the requested rank
    int limits[6];
    bool limited[6] = {false};
    for (int i = 0; i < 3; ++i) {
      limits[2 * i] = ind0[i] - rank;
      if (limits[2 * i] < 0) {
        limits[2 * i]  = 0;
        limited[2 * i] = true;
      }
      limits[2 * i + 1] = ind0[i] + rank;
      if (limits[2 * i + 1] > fHelper->fNgrid - 1) {
        limits[2 * i + 1]  = fHelper->fNgrid - 1;
        limited[2 * i + 1] = true;
      }
    }
    // Gather all cells for the given rank
    for (int iax1 = 0; iax1 < 3; ++iax1) {
      int iax2 = (iax1 + 1) % 3;
      int iax3 = (iax1 + 2) % 3;
      for (int iside = 0; iside < 2; ++iside) {
        if (limited[2 * iax1 + iside]) continue;
        ind[iax1] = limits[2 * iax1 + iside];
        for (ind[iax2] = limits[2 * iax2]; ind[iax2] <= limits[2 * iax2 + 1]; ind[iax2]++) {
          for (ind[iax3] = limits[2 * iax3]; ind[iax3] <= limits[2 * iax3 + 1]; ind[iax3]++) {
            AddCandidatesFromCell(ind);
          }
        }
      }
    }
    return (fCandidates.size());
  }

  /// Method for adding a new triangular facet
  /** @param vt0      First vertex
      @param vt1      Second vertex
      @param vt2      Third vertex
      @param absolute If true then vt0, vt1 and vt2 are the vertices to be added in
        anti-clockwise order looking from the outsider. If false the vertices are relative
        to the first: vt0, vt0+vt1, vt0+vt2, in anti-clockwise order when looking from the outsider.
  */
  VECCORE_ATT_HOST_DEVICE
  bool AddTriangularFacet(Vector3D<T> const &vt0, Vector3D<T> const &vt1, Vector3D<T> const &vt2, bool absolute = true)
  {
    VECGEOM_ASSERT(NVERT == 3);
    Facet_t *facet = new Facet_t;
    bool added     = false;
    if (absolute)
      added = facet->SetVertices(vt0, vt1, vt2);
    else
      added = facet->SetVertices(vt0, vt0 + vt1, vt0 + vt1 + vt2);
    if (!added) {
      delete facet;
      return false;
    }
    AddFacet(facet);
    return true;
  }

  /// Method for adding a new quadrilateral facet
  /** @param vt0      First vertex
      @param vt1      Second vertex
      @param vt2      Third vertex
      @param vt3      Fourth vertex
      @param absolute If true then vt0, vt1, vt2 and vt3 are the vertices to be added in
        anti-clockwise order looking from the outsider. If false the vertices are relative
        to the first: vt0, vt0+vt1, vt0+vt2, vt0+vt3 in anti-clockwise order when looking from the
        outsider.
  */
  VECCORE_ATT_HOST_DEVICE
  bool AddQuadrilateralFacet(Vector3D<T> const &vt0, Vector3D<T> const &vt1, Vector3D<T> const &vt2,
                             Vector3D<T> const &vt3, bool absolute = true)
  {
    // We should check the quadrilateral convexity to correctly define the
    // triangle facets
    // CheckConvexity()vt0, vt1, vt2, vt3, absolute);
    VECGEOM_ASSERT(NVERT <= 4);
    Facet_t *facet = new Facet_t;
    if (NVERT == 3) {
      if (absolute) {
        if (!facet->SetVertices(vt0, vt1, vt2)) {
          delete facet;
          return false;
        }
      } else {
        if (!facet->SetVertices(vt0, vt0 + vt1, vt0 + vt1 + vt2)) {
          delete facet;
          return false;
        }
      }
      AddFacet(facet);
      facet = new Facet_t;
      if (absolute) {
        if (!facet->SetVertices(vt0, vt2, vt3)) {
          delete facet;
          return false;
        }
      } else {
        if (!facet->SetVertices(vt0, vt0 + vt2, vt0 + vt2 + vt3)) {
          delete facet;
          return false;
        }
      }
      AddFacet(facet);
      return true;
    } else if (NVERT == 4) {
      // Add a single facet
      if (absolute) {
        if (!facet->SetVertices(vt0, vt1, vt2, vt3)) {
          delete facet;
          return false;
        }
      } else {
        if (!facet->SetVertices(vt0, vt0 + vt1, vt0 + vt1 + vt2, vt0 + vt1 + vt2 + vt3)) {
          delete facet;
          return false;
        }
      }
      AddFacet(facet);
      return true;
    }
    return false;
  }

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
