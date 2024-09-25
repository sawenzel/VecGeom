#ifndef VECGEOM_SURFDATA_H_
#define VECGEOM_SURFDATA_H_

#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/bvh/BVHsurf.h>
#include <VecGeom/base/BVH.h>

namespace vgbrep {

/*
 * The main surface storage utility, providing access by index to:
 *    * global and local transformations applied to surfaces
 *    * surface data per surface type
 *    * mask data per mask type
 *    * imprint data (list of masks)
 */
template <typename Real_t>
struct SurfData {

  // BVH internal precision
#ifdef VECGEOM_BVH_SINGLE
  using Real_b = float;
#else
  using Real_b = double;
#endif
  using EllipData_t    = EllipData<Real_t>;
  using TorusData_t    = TorusData<Real_t>;
  using Arb4Data_t     = Arb4Data<Real_t>;
  using WindowMask_t   = WindowMask<Real_t>;
  using RingMask_t     = RingMask<Real_t>;
  using ZPhiMask_t     = ZPhiMask<Real_t>;
  using TriangleMask_t = TriangleMask<Real_t>;
  using QuadMask_t     = QuadrilateralMask<Real_t>;
  using SideDivision_t = SideDivision<Real_t>;

  int fNscenes{0};

  int fNvolTrans{0};
  int fNlocalSurf{0};
  int fNExitingSurfaces{0};
  int fNEnteringSurfaces{0};
  int fNglobalSurf{0};
  int fNellip{0};
  int fNtorus{0};
  int fNarb4{0};
  int fNcommonSurf{0};
  int fNsides{0};
  int fNStates{0};
  int fSizeCandList{0};
  int fNshells{0};
  int fNlogic{0};
  int fNrange{0};
  int fNwindows{0};
  int fNrings{0};
  int fNzphis{0};
  int fNtriangs{0};
  int fNquads{0};
  int fNsideDivisions{0};
  int fNslices{0};
  int fNsliceCandidates{0};
  int fNPlacedVolumes{0};

  int *fSceneStartIndex{nullptr}; ///< Start indices for data indexed by state id (per scene)
  int *fSceneTouchables{nullptr}; ///< Number of touchables (per scene)

  /// Transformations.
  TransformationMP<Real_t> *fPVolTrans{nullptr}; ///< Transformations to placed volumes

  /// Volume shells, indexed by the logical volume id
  VolumeShell *fShells{nullptr}; ///< volume shells

  // Local and global framed surfaces
  FramedSurface<Real_t, TransformationMP<Real_t>> *fLocalSurf{nullptr};             ///< local surfaces
  FramedSurface<Real_t, vecgeom::Transformation2DMP<Real_t>> *fFramedSurf{nullptr}; ///< global surfaces

  EllipData_t *fEllipData{nullptr}; ///< Elliptical data
  TorusData_t *fTorusData{nullptr}; ///< Torus data
  Arb4Data_t *fArb4Data{nullptr};   ///< Arb4 data

  // Frame data
  WindowMask_t *fWindowMasks{nullptr};             ///< rectangular masks
  RingMask_t *fRingMasks{nullptr};                 ///< ring masks
  ZPhiMask_t *fZPhiMasks{nullptr};                 ///< cylindrical masks
  TriangleMask_t *fTriangleMasks{nullptr};         ///< triangular masks
  QuadMask_t *fQuadMasks{nullptr};                 ///< quadrilateral masks
  CommonSurface<Real_t> *fCommonSurfaces{nullptr}; ///< common surfaces
  Candidates *fCandidates;                         ///< candidate surfaces per navigation state
  int *fSides{nullptr};                            ///< side surface indices
  SideDivision_t *fSideDivisions{nullptr};         ///< [fNsideDivisions] side division helpers
  SliceCand *fSlices{nullptr};                     ///< [fNslices] slice candidates
  int *fSliceCandidates{nullptr};                  ///< [fNsliceCandidates] frame indices for all slice candidates
  int *fSurfShellList{nullptr};                    ///< indices of local surfaces used in shells
  logic_int *fLogicList{nullptr};                  ///< list of logic expressions per volume
  int *fCandList{nullptr};                         ///< global list of candidate indices
  int *fShellExitingSurfaceList{nullptr};          ///< List of surfaces of a volume, having a frame
  int *fShellEnteringSurfaceList{nullptr};         ///< List of surfaces of the daughters of a volume, having a frame
  int *fShellEnteringSurfacePvolList{nullptr};     ///< Id of the PlacedVolume each entering surface belongs to
  int *fShellEnteringSurfacePvolTransList{
      nullptr};                                  ///< Id of the volume transformation each entering surface belongs to
  int *fShellEnteringSurfaceLvolIdList{nullptr}; ///< Id of the logical volume each entering surface belongs to
  int *fShellDaughterPvolIdList{nullptr};        ///< Global PV Ids of the daughter PVs of each Volume
  int *fShellDaughterPvolTransList{nullptr};     ///< Transformations of the daughter PVs of each Volume
  bvh::BVHsurf<Real_b> *fBVH{
      nullptr}; ///< BVH per volume shell, built from the AABBs of its entering and exiting surfaces
  bvh::BVHsurf<Real_b> *fBVHSolids{nullptr}; ///< BVH per volume shell, built from the AABBs of the daughter volumes

  SurfData() = default;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static SurfData<Real_t> &Instance()
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return *globaldevicesurfdata::gSurfDataDevice<Real_t>;
#else
    static SurfData<Real_t> gSurfData;
    return gSurfData;
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  void Clear()
  {
    delete[] fWindowMasks;
    fWindowMasks = nullptr;
    delete[] fRingMasks;
    fRingMasks = nullptr;
    delete[] fZPhiMasks;
    fZPhiMasks = nullptr;
    delete[] fQuadMasks;
    fQuadMasks = nullptr;
    delete[] fTriangleMasks;
    fTriangleMasks = nullptr;
    delete[] fEllipData;
    fEllipData = nullptr;
    delete[] fTorusData;
    fTorusData = nullptr;
    delete[] fArb4Data;
    fArb4Data = nullptr;
    delete[] fFramedSurf;
    fFramedSurf = nullptr;
    delete[] fSides;
    fSides = nullptr;
    delete[] fSideDivisions;
    fSideDivisions = nullptr;
    delete[] fSlices;
    fSlices = nullptr;
    delete[] fSliceCandidates;
    fSliceCandidates = nullptr;
    delete[] fCommonSurfaces;
    fCommonSurfaces = nullptr;
    delete[] fCandList;
    fCandList = nullptr;
    delete[] fCandidates;
    fCandidates = nullptr;
    delete[] fLocalSurf;
    fLocalSurf = nullptr;
    delete[] fShells;
    fShells = nullptr;
    delete[] fSurfShellList;
    fSurfShellList = nullptr;
    delete[] fLogicList;
    fLogicList = nullptr;
    delete[] fShellExitingSurfaceList;
    fShellExitingSurfaceList = nullptr;
    delete[] fShellEnteringSurfaceList;
    fShellEnteringSurfaceList = nullptr;
    delete[] fShellEnteringSurfacePvolList;
    fShellEnteringSurfacePvolList = nullptr;
    delete[] fShellEnteringSurfacePvolTransList;
    fShellEnteringSurfacePvolTransList = nullptr;
    delete[] fShellEnteringSurfaceLvolIdList;
    fShellEnteringSurfaceLvolIdList = nullptr;
    delete[] fShellDaughterPvolIdList;
    fShellDaughterPvolIdList = nullptr;
    delete[] fBVH;
    fBVH = nullptr;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Candidates const &GetCandidates(int scene_id, int state_id) const
  {
    return fCandidates[fSceneStartIndex[scene_id] + state_id];
  }

  /// Surface data accessors by component id
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  EllipData_t const &GetEllipData(int id) const { return fEllipData[id]; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  TorusData_t const &GetTorusData(int id) const { return fTorusData[id]; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Arb4Data_t const &GetArb4Data(int id) const { return fArb4Data[id]; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  WindowMask_t const &GetWindowMask(int id) const { return fWindowMasks[id]; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  RingMask_t const &GetRingMask(int id) const { return fRingMasks[id]; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  ZPhiMask_t const &GetZPhiMask(int id) const { return fZPhiMasks[id]; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  TriangleMask_t const &GetTriangleMask(int id) const { return fTriangleMasks[id]; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  QuadMask_t const &GetQuadMask(int id) const { return fQuadMasks[id]; }

  // Accessors by common surface id
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  UnplacedSurface<Real_t> const &GetUnplaced(int isurf, bool &flipped) const
  {
    FramedSurface<Real_t, vecgeom::Transformation2DMP<Real_t>> const &framedsurf =
        fFramedSurf[fCommonSurfaces[isurf].fLeftSide.fSurfaces[0]];
    flipped = fCommonSurfaces[isurf].fFlipped;
    return framedsurf.fSurface;
  }

  // Accessors by FSlocator

  /// @brief Get framed surface pointed by a locator
  /// @param locator Frame locator
  /// @return Frame pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  FramedSurface<Real_t, vecgeom::Transformation2DMP<Real_t>> const &GetFramedSurface(FSlocator const &locator) const
  {
    auto const &surf = fCommonSurfaces[locator.GetCSindex()];
    auto const &side = locator.IsLeftSide() ? surf.fLeftSide : surf.fRightSide;
    assert((locator.frame_id >= 0 && locator.frame_id < side.fNsurf && "Wrong locator") || locator.GetCSindex() == 0);
    return fFramedSurf[side.fSurfaces[locator.frame_id]];
  }

  /// @brief Deduce overlap of framed surface from a locator
  /// @param locator Frame locator
  /// @return Frame pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFSOverlapping(FSlocator const &locator) const
  {
    // frame surf id -1 indicates an extruding overlap, nonetheless, we must return false here,
    // so the overlap can be detected properly and the true position can be found. // fixme maybe we can improve the
    // logic in testRaytracing.cpp
    if (locator.GetFSindex() == -1) return false;
    auto const &framedsurf = GetFramedSurface(locator);
    return framedsurf.fOverlapping;
  }

  /// @brief Deduce state of framed surface from a locator
  /// @param locator Frame locator
  /// @return Frame pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int FramedSurfaceState(FSlocator const &locator) const
  {
    auto const &framedsurf = GetFramedSurface(locator);
    return framedsurf.fState;
  }

  /// @brief Deduce LogicId of framed surface from a locator
  /// @param locator Frame locator
  /// @return Frame pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int FramedSurfaceLogicId(FSlocator const &locator) const
  {
    if (locator.GetFSindex() == -1) return 0;
    auto const &framedsurf = GetFramedSurface(locator);
    return framedsurf.fLogicId;
  }

  /// @brief Check if framed surface from a locator is embedded
  /// @param locator Frame locator
  /// @return Frame pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsFramedSurfaceEmbedded(FSlocator const &locator) const
  {
    auto const &framedsurf = GetFramedSurface(locator);
    return framedsurf.fEmbedded;
  }

  /// @brief Check if framed surface from a locator is embedded
  /// @param locator Frame locator
  /// @return Frame pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int FramedSurfaceParentInd(FSlocator const &locator) const
  {
    auto const &framedsurf = GetFramedSurface(locator);
    return framedsurf.fParent;
  }

  /// @brief Get common surface pointed by a locator
  /// @param locator Frame locator
  /// @return CS pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  CommonSurface<Real_t> const &GetCommonSurface(FSlocator const &locator) const
  {
    return fCommonSurfaces[locator.GetCSindex()];
  }

  /// @brief Get side on common surface pointed by a locator
  /// @param locator Frame locator
  /// @return Side pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Side const &GetSide(FSlocator const &locator) const
  {
    auto const &surf = fCommonSurfaces[locator.GetCSindex()];
    return locator.IsLeftSide() ? surf.fLeftSide : surf.fRightSide;
  }

  /// @brief Get opposite side on common surface pointed by a locator
  /// @param locator Frame locator
  /// @return Side opposite to the one pointed by the locator
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Side const &GetOppositeSide(FSlocator const &locator) const
  {
    auto const &surf = fCommonSurfaces[locator.GetCSindex()];
    return locator.IsLeftSide() ? surf.fRightSide : surf.fLeftSide;
  }

  /// @brief Get scene FSlocator from a touchable one
  /// @param ltouchable Locator for the touchable framed surface
  /// @param lscene Locator for the scene framed surface
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void TouchableToSceneLocator(FSlocator const &ltouchable, FSlocator &lscene) const
  {
    auto const &framedsurf = GetFramedSurface(ltouchable);
    assert(framedsurf.fSceneCS > 0 && "Touchable locator not on a scene surface");
    lscene.common_id = framedsurf.fSceneCS * (1 - 2 * int(framedsurf.fSceneCSind < 0));
    lscene.frame_id  = vecCore::math::Abs(framedsurf.fSceneCSind) - 1;
  }

  /// @brief Get touchable FSlocator from a scene one
  /// @param scene_state Global state for the scene surface in the parent scene
  /// @param surf_index Local surface index for the touchable
  /// @param ltouchable Locator for the touchable framed surface
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SceneToTouchableLocator(vecgeom::NavigationState const &scene_state, int surf_index, FSlocator &ltouchable) const
  {
    constexpr char kLside          = 1;
    unsigned short parent_scene_id = 0, dummy_id = 0;
    scene_state.GetSceneId(parent_scene_id, dummy_id);
    auto parent_state_id = scene_state.GetId();
    // Get parent scene common surface and side from the exiting candidates (ordered by the surf_index of
    // the scene volume surface)
    auto const &cand_scene = GetCandidates(parent_scene_id, parent_state_id);
    ltouchable.common_id   = cand_scene[surf_index] * (1 - 2 * int((cand_scene.fSides[surf_index] & kLside) == 0));
    ltouchable.frame_id    = cand_scene.fFrameInd[surf_index];
  }
};
} // namespace vgbrep
#endif
