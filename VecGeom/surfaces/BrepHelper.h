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
// #include <VecGeom/management/BVHManager.h>

namespace vgbrep {

template <typename Real_t>
class BrepHelper {
  using SurfData_t     = SurfData<Real_t>;
  using EllipData_t    = EllipData<Real_t>;
  using TorusData_t    = TorusData<Real_t>;
  using Arb4Data_t     = Arb4Data<Real_t>;
  using WindowMask_t   = WindowMask<Real_t>;
  using RingMask_t     = RingMask<Real_t>;
  using ZPhiMask_t     = ZPhiMask<Real_t>;
  using TriangleMask_t = TriangleMask<Real_t>;
  using QuadMask_t     = QuadrilateralMask<Real_t>;
  using CPUsurfData_t  = CPUsurfData<vecgeom::Precision>;
  using SideDivision_t = SideDivision<Real_t>;

private:
  int fVerbose{0};                ///< verbosity level
  SurfData_t *fSurfData{nullptr}; ///< Surface data
  CPUsurfData_t &fCPUdata;        ///< Transient CPU surface data used during conversion

public:
  ~BrepHelper();

  /// @brief Returns the singleton instance (CPU only)
  /// @return Singleton instance
  static BrepHelper &Instance();

  /// @brief Clear all content of the arrays
  void ClearData();

  /// @brief Top-level conversion from a closed GeoManager to the surface model
  /// @return Successful conversion
  bool Convert();

  /// @brief Surface data getter
  SurfData_t const &GetSurfData() const { return *fSurfData; }

  /// @brief Print a summary on the numbers and sizes for the surface data
  void PrintSurfData();

  /// @brief Construction time verbosity
  /// @param verbose Verbosity level
  void SetVerbosity(int verbose) { fVerbose = verbose; }

  // methods
private:
  /// @brief Private ctor used by the singleton creation
  BrepHelper();

  /// @brief Computes default states for both sides of a common surface
  /// @param common_id Comon surface id
  void ComputeDefaultStates(int common_id);

  /// @brief  Computes the division helper for a cylindrical side.
  /// @param side Side to which the extent is computed
  /// @param extent_full extent of the side
  /// @return Index of the helper
  int ComputeCylinderDivision(Side &side, ZPhiMask<double> extent_full);

  // Computes bounding extent on a side of cylindrical surface
  ZPhiMask<double> ComputeCylinderExtent(const Side &side);

  /// @brief  Computes the division helper for a planar side.
  /// @param side Side to which the extent is computed
  /// @param extent_full extent of the side
  /// @return Index of the helper
  int ComputePlaneDivision(Side &side, WindowMask<double> extent_full);

  /// @brief  Computes the bounding extent on a planar side.
  /// @param side Side to which the extent is computed
  WindowMask<double> ComputePlaneExtent(const Side &side);

  /// @brief Computes division helpers for all sides
  void ComputeSideDivisions();

  // /// @brief Computes extents for the sides of all common surfaces
  // /// @return Operation success
  // bool ComputeExtents();

  ///< This method uses the transformation T1 of the first placed surface on the left side (which always exists)
  ///< as transformation for the common surface, then recalculates the transformations of all placed
  ///< surfaces as T' = T1.Inverse() * T. If identity this will get the index 0.
  void ConvertTransformations(int idsurf);

  ///< This method creates helper lists of candidate surfaces for each navigation state
  void CreateCandidateLists();

  ///< @brief Create a common surface in scene_id
  int CreateCommonSurface(int idglob, int volId, int scene_id, int &iframe, char &iside);

  /// @brief Method creating all common surfaces
  bool CreateCommonSurfacesScenes();

  /// @brief Create local surfaces by convering all defined solids
  bool CreateLocalSurfaces();

  /// @brief Iterate the geometry tree and flatten surfaces at scene level
  /// @return Success of operation
  void DumpBVH(uint ivol);

  /// @brief This function evaluates if the frames of two placed surfaces on the same side of a common surface are matching
  /// @param side Side containing the framed surfaces
  /// @param i1 Index of the first framed surface
  /// @param i2 Index of the second framed surface
  /// @return Equality
  bool EqualFrames(Side const &side, int i1, int i2);

  /// @brief Iterates over all logical volumes and initializes the data needed for BVH construction and navigation
  void InitBVHData();

  /// @brief Iterates over all logical volumes and identifies convex surfaces in booleans, which can be treated as normal surfaces
  void FindConvexBooleanSurfaces();

  /// @brief Print the list of common surface candidates for a given state
  /// @param state Full state (not just local scene state)
  void PrintCandidates(vecgeom::NavigationState const &state);
  void PrintCandidateLists();

  /// @brief Print verbose info for a common surface
  /// @param common_id Common surface index
  void PrintCommonSurface(int common_id);

  /// @brief Print verbose info for a framed surface
  /// @param surf Framed surface
  template <typename Real_i, typename Transformation_t>
  void PrintFramedSurface(FramedSurface<Real_i, Transformation_t> const &surf);

  /// @brief Reserve container slots based on the number of registered volumes
  /// @param nvolumes Number of registered volumes
  void SetNvolumes(int nvolumes);

  /// @brief Sort framed surfaces on sides, daughters first (see: FramedSurface::operator< )
  /// @param common_id Common surface id for which to sort sides
  void SortSides(int common_id);

  /// @brief A function to update all mask containers.
  /// @details Needs to be called when updating masks after creating both frames and extents.
  void UpdateMaskData();

  /// @brief The method updates the SurfData storage
  void UpdateSurfData();
};

} // namespace vgbrep
#endif
