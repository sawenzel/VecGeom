#ifndef VECGEOM_SURFACE_BREPHELPER_H_
#define VECGEOM_SURFACE_BREPHELPER_H_

#include <cassert>
#include <functional>
#include <map>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/SurfData.h>
#include <VecGeom/surfaces/conv/SolidConverter.h>
#include <VecGeom/surfaces/bvh/BVHsurfCreator.h>
#include <VecGeom/management/Logger.h>

// Check if math necessary
#include <VecGeom/base/Math.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/Box.h>
#include <VecGeom/volumes/Trd.h>
#include <VecGeom/volumes/Tube.h>
#include <VecGeom/volumes/BooleanVolume.h>
#include <VecGeom/management/GeoManager.h>
//#include <VecGeom/management/BVHManager.h>

namespace vgbrep {

template <typename Real_t>
class BrepHelper {
  using SurfData_t     = SurfData<Real_t>;
  using CylData_t      = CylData<Real_t>;
  using ConeData_t     = ConeData<Real_t>;
  using EllipData_t    = EllipData<Real_t>;
  using SphData_t      = SphData<Real_t>;
  using TorusData_t    = TorusData<Real_t>;
  using Arb4Data_t     = Arb4Data<Real_t>;
  using WindowMask_t   = WindowMask<Real_t>;
  using RingMask_t     = RingMask<Real_t>;
  using ZPhiMask_t     = ZPhiMask<Real_t>;
  using TriangleMask_t = TriangleMask<Real_t>;
  using QuadMask_t     = QuadrilateralMask<Real_t>;
  using CPUsurfData_t  = CPUsurfData<vecgeom::Precision>;

private:
  int fVerbose{0};                ///< verbosity level
  SurfData_t *fSurfData{nullptr}; ///< Surface data
  CPUsurfData_t &fCPUdata;        ///< Transient CPU surface data used during conversion

  BrepHelper();

public:
  ~BrepHelper();
  /// Returns the singleton instance (CPU only)
  static BrepHelper &Instance();

  template <typename Real_i>
  bool ApproxEqualTransformation(const vecgeom::Transformation3DMP<Real_i> &t1,
                                 const vecgeom::Transformation3DMP<Real_i> &t2);
  void ClearData();
  void ComputeDefaultStates(int common_id);
  // Computes bounding extent on a side of cylindrical surface
  bool ComputeCylinderExtent(Side &side);
  // Computes the bounding extent on a planar side.
  void ComputePlaneExtent(Side &side);
  bool ComputeExtents();
  /// @brief Top-level conversion from a closed GeoManager to the surface model
  /// @return Successful conversion
  bool Convert();
  ///< This method uses the transformation T1 of the first placed surface on the left side (which always exists)
  ///< as transformation for the common surface, then recalculates the transformations of all placed
  ///< surfaces as T' = T1.Inverse() * T. If identity this will get the index 0.
  void ConvertTransformations(int idsurf);
  ///< This method creates helper lists of candidate surfaces for each navigation state
  void CreateCandidateLists();
  /// @brief Iterate the geometry tree and flatten surfaces at scene level
  /// @return Success of operation
  bool CreateCommonSurfacesScenes();
  bool CreateLocalSurfaces();
  void DumpBVH(uint ivol);
  SurfData_t const &GetSurfData() const { return *fSurfData; }
  /// @brief Iterates over all logical volumes and initializes the data needed for BVH construction and navigation
  void InitBVHData();
  /// @brief Print the list of common surface candidates for a given state
  /// @param state Full state (not just local scene state)
  void PrintCandidates(vecgeom::NavigationState const &state);
  void PrintCandidateLists();
  void PrintCommonSurface(int common_id);
  void PrintFramedSurface(FramedSurface const &surf);
  void PrintSurfData();
  void SetNvolumes(int nvolumes);
  void SetVerbosity(int verbose) { fVerbose = verbose; }
  void SortSides(int common_id);

private:
  ///< @brief Create a common surface in scene_id
  int CreateCommonSurface(int idglob, int volId, int scene_id, int &iframe, char &iside);
  ///< This function evaluates if the frames of two placed surfaces on the same side
  ///< of a common surface are matching
  bool EqualFrames(Side const &side, int i1, int i2);
  // A function to update all mask containers. Needs to be called
  // when updating masks after creating both frames and extents.
  void UpdateMaskData();
  ///< The method updates the SurfData storage
  void UpdateSurfData();
};

} // namespace vgbrep
#endif
