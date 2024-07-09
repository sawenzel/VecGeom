#ifndef VECGEOM_SURFACE_CPUTYPES_H
#define VECGEOM_SURFACE_CPUTYPES_H

#include <VecGeom/surfaces/base/CommonTypes.h>
#include <VecGeom/surfaces/Model.h>

namespace vgbrep {

template <typename T>
VECGEOM_FORCE_INLINE char const *to_cstring(T type)
{
  return nullptr;
}

template <>
VECGEOM_FORCE_INLINE char const *to_cstring<SurfaceType>(SurfaceType type)
{
  static const char *const data[] = {"planar", "cylindrical", "conical", "spherical", "torus", "elliptical", "arb4"};
  assert(size_t(type) * sizeof(const char *) < sizeof(data));
  return data[static_cast<int>(type)];
}

template <>
VECGEOM_FORCE_INLINE char const *to_cstring<bool>(bool type)
{
  if (type) return "true";
  return "false";
}

template <>
VECGEOM_FORCE_INLINE char const *to_cstring<FrameType>(FrameType type)
{
  static const char *const data[] = {"no_frame", "rangeZ", "ring", "z_phi", "rangeSph", "window", "triangle", "quad"};
  assert(size_t(type) * sizeof(const char *) < sizeof(data));
  return data[static_cast<int>(type)];
}

using LogicExpressionCPU = std::vector<logic_int>;

// Placeholder (on host) for all surfaces belonging to a volume. An array of those will be indexed
// by the logical volume id. Also an intermediate helper for building portals.
// Note: the local surfaces defined by solids will have local references that will be changed by
// the flattening process, depending on the scene on which the parent volume will be flattened
struct VolumeShellCPU {
  std::vector<int> fSurfaces;         ///< Local surface id's for this volume
  std::vector<int> fExitingSurfaces;  ///< Local surface id's for this volume, excluding virtual ones
  std::vector<int> fEnteringSurfaces; ///< Local surface id's for all surfaces of daughters, excluding virtual ones
  std::vector<int> fEnteringSurfacesPvol;

  LogicExpressionCPU fLogic; ///< Logic expression for the solid
  bool fSimplified{false};   ///< The logic was simplified
  int fBVH{0};
};

// Surface data used only on CPU during the conversion process
template <typename Real_t>
struct CPUsurfData {
  using VecInt_t       = std::vector<int>;
  using VecChar_t      = std::vector<char>;
  using MultimapInt_t  = std::multimap<long, int>;
  using SurfData_t     = SurfData<Real_t>;
  using CylData_t      = CylData<Real_t>;
  using ConeData_t     = ConeData<Real_t>;
  using EllipData_t    = EllipData<Real_t>;
  using TorusData_t    = TorusData<Real_t>;
  using SphData_t      = SphData<Real_t>;
  using Arb4Data_t     = Arb4Data<Real_t>;
  using WindowMask_t   = WindowMask<Real_t>;
  using RingMask_t     = RingMask<Real_t>;
  using ZPhiMask_t     = ZPhiMask<Real_t>;
  using TriangleMask_t = TriangleMask<Real_t>;
  using QuadMask_t     = QuadrilateralMask<Real_t>;

  std::vector<WindowMask_t> fWindowMasks; ///< rectangular masks
  std::vector<RingMask_t> fRingMasks;     ///< ring masks
  std::vector<ZPhiMask_t> fZPhiMasks;     ///< cylindrical masks
  std::vector<TriangleMask_t> fTriangleMasks;
  std::vector<QuadMask_t> fQuadMasks;                 ///< quadrilateral masks
  std::vector<CylData_t> fCylSphData;                 ///< data for cyl surfaces
  std::vector<ConeData_t> fConeData;                  ///< data for conical surfaces
  std::vector<EllipData_t> fEllipData;                ///< data for elliptical surfaces
  std::vector<TorusData_t> fTorusData;                ///< data for torus surfaces
  std::vector<Arb4Data_t> fArb4Data;                  ///< data for Arb4 surfaces
  std::vector<TransformationMP<Real_t>> fLocalTrans;  ///< local transformations
  std::vector<TransformationMP<Real_t>> fGlobalTrans; ///< global transformations for surfaces in the scene
  std::vector<FramedSurface> fLocalSurfaces;          ///< local surfaces per logical volume
  std::vector<FramedSurface> fFramedSurf;             ///< global surfaces
  std::vector<CommonSurface> fCommonSurfaces;         ///< common surfaces
  std::vector<VolumeShellCPU> fShells;                ///< vector of local volume surfaces
  std::vector<VolumeShellCPU> fSceneShells;           ///< vector of scene volume surfaces

  VecInt_t fSceneStartIndex;            ///< Start indices for data indexed by state id (per scene)
  VecInt_t fSceneTouchables;            ///< Number of touchables (per scene)
  std::vector<MultimapInt_t> fSurfHash; ///< maps rotation hash index to a list of common surface id's (per scene)

  std::vector<VecInt_t> fCandidatesEntering; ///< list of entering candidates: scene0...,scene1...
  std::vector<VecInt_t> fCandidatesExiting;  ///< list of exiting candidates: scene0...,scene1...
  std::vector<VecInt_t> fFrameIndEntering; ///< list of start frame indices for entering candidates: scene0...,scene1...
  std::vector<VecInt_t> fFrameIndExiting;  ///< list of start frame indices for exiting candidates: scene0...,scene1...
  std::vector<VecChar_t> fSidesEntering;   ///< list of relevant sides for entering candidates: scene0...,scene1...
  std::vector<VecChar_t> fSidesExiting;    ///< list of relevant sides for exiting candidates: scene0...,scene1...

private:
  CPUsurfData() = default;

public:
  static VECGEOM_FORCE_INLINE CPUsurfData<Real_t> &Instance()
  {
    static CPUsurfData<Real_t> gCPUsurfdata;
    return gCPUsurfdata;
  }

  void Clear()
  {
    // Dispose of surface data and shrink the container
    std::vector<WindowMask_t>().swap(fWindowMasks);
    std::vector<RingMask_t>().swap(fRingMasks);
    std::vector<ZPhiMask_t>().swap(fZPhiMasks);
    std::vector<TriangleMask_t>().swap(fTriangleMasks);
    std::vector<QuadMask_t>().swap(fQuadMasks);
    std::vector<CylData_t>().swap(fCylSphData);
    std::vector<ConeData_t>().swap(fConeData);
    std::vector<EllipData_t>().swap(fEllipData);
    std::vector<TorusData_t>().swap(fTorusData);
    std::vector<TransformationMP<Real_t>>().swap(fLocalTrans);
    std::vector<TransformationMP<Real_t>>().swap(fGlobalTrans);
    std::vector<FramedSurface>().swap(fLocalSurfaces);
    std::vector<FramedSurface>().swap(fFramedSurf);
    std::vector<CommonSurface>().swap(fCommonSurfaces);
    std::vector<VolumeShellCPU>().swap(fShells);
    std::vector<VolumeShellCPU>().swap(fSceneShells);
    VecInt_t().swap(fSceneStartIndex);
    VecInt_t().swap(fSceneTouchables);
    std::vector<MultimapInt_t>().swap(fSurfHash);
    std::vector<VecInt_t>().swap(fCandidatesEntering);
    std::vector<VecInt_t>().swap(fCandidatesExiting);
    std::vector<VecInt_t>().swap(fFrameIndEntering);
    std::vector<VecInt_t>().swap(fFrameIndExiting);
    std::vector<VecChar_t>().swap(fSidesEntering);
    std::vector<VecChar_t>().swap(fSidesExiting);
  }

  VecInt_t &GetCandidatesEntering(int scene_id, int state_id)
  {
    return fCandidatesEntering[fSceneStartIndex[scene_id] + state_id];
  }

  VecInt_t &GetCandidatesExiting(int scene_id, int state_id)
  {
    return fCandidatesExiting[fSceneStartIndex[scene_id] + state_id];
  }

  VecInt_t &GetFrameIndEntering(int scene_id, int state_id)
  {
    return fFrameIndEntering[fSceneStartIndex[scene_id] + state_id];
  }

  VecInt_t &GetFrameIndExiting(int scene_id, int state_id)
  {
    return fFrameIndExiting[fSceneStartIndex[scene_id] + state_id];
  }

  VecChar_t &GetSidesEntering(int scene_id, int state_id)
  {
    return fSidesEntering[fSceneStartIndex[scene_id] + state_id];
  }

  VecChar_t &GetSidesExiting(int scene_id, int state_id)
  {
    return fSidesExiting[fSceneStartIndex[scene_id] + state_id];
  }

  void GetMask(int id, WindowMask_t const *&mask) { mask = &fWindowMasks[id]; }
  void GetMask(int id, RingMask_t const *&mask) { mask = &fRingMasks[id]; }
  void GetMask(int id, ZPhiMask_t const *&mask) { mask = &fZPhiMasks[id]; }
  void GetMask(int id, TriangleMask_t const *&mask) { mask = &fTriangleMasks[id]; }
  void GetMask(int id, QuadMask_t const *&mask) { mask = &fQuadMasks[id]; }

  /// @brief Trampoline function to to the frame embedding checker
  /// @param f1 Parent framed surface
  /// @param f2 Child framed surface
  /// @return Child is embedded in parent
  bool IsEmbedding(FramedSurface const &f1, FramedSurface const &f2)
  {
    auto log_not_supported = [&]() {
      VECGEOM_LOG(error) << "Embedding check " << to_cstring(f1.fFrame.type) << " - " << to_cstring(f2.fFrame.type)
                         << " not supported";
    };
    TransformationMP<Real_t> const &t1 = fGlobalTrans[f1.fTrans];
    TransformationMP<Real_t> const &t2 = fGlobalTrans[f2.fTrans];
    TransformationMP<Real_t> trans     = t2 * t1.Inverse();
    trans.SetProperties();

    switch (f1.fFrame.type) {
    case FrameType::kRing: {
      RingMask_t const *mask1 = nullptr;
      GetMask(f1.fFrame.id, mask1);
      switch (f2.fFrame.type) {
      case FrameType::kRing: {
        RingMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, RingMask_t, RingMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kWindow: {
        WindowMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, RingMask_t, WindowMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kTriangle: {
        TriangleMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, RingMask_t, TriangleMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kQuadrilateral: {
        QuadMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, RingMask_t, QuadMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    case FrameType::kZPhi: {
      ZPhiMask_t const *mask1 = nullptr;
      GetMask(f1.fFrame.id, mask1);
      switch (f2.fFrame.type) {
      case FrameType::kZPhi: {
        ZPhiMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, ZPhiMask_t, ZPhiMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    case FrameType::kWindow: {
      WindowMask_t const *mask1 = nullptr;
      GetMask(f1.fFrame.id, mask1);
      switch (f2.fFrame.type) {
      case FrameType::kRing: {
        RingMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, WindowMask_t, RingMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kWindow: {
        WindowMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, WindowMask_t, WindowMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kTriangle: {
        TriangleMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, WindowMask_t, TriangleMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kQuadrilateral: {
        QuadMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, WindowMask_t, QuadMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    case FrameType::kTriangle: {
      TriangleMask_t const *mask1 = nullptr;
      GetMask(f1.fFrame.id, mask1);
      switch (f2.fFrame.type) {
      case FrameType::kRing: {
        RingMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, TriangleMask_t, RingMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kWindow: {
        WindowMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, TriangleMask_t, WindowMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kTriangle: {
        TriangleMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, TriangleMask_t, TriangleMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kQuadrilateral: {
        QuadMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, TriangleMask_t, QuadMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    case FrameType::kQuadrilateral: {
      QuadMask_t const *mask1 = nullptr;
      GetMask(f1.fFrame.id, mask1);
      switch (f2.fFrame.type) {
      case FrameType::kRing: {
        RingMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, QuadMask_t, RingMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kWindow: {
        WindowMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, QuadMask_t, WindowMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kTriangle: {
        TriangleMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, QuadMask_t, TriangleMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      case FrameType::kQuadrilateral: {
        QuadMask_t const *mask2 = nullptr;
        GetMask(f2.fFrame.id, mask2);
        return FrameChecker<Real_t, QuadMask_t, QuadMask_t>::IsEmbedding(*mask1, *mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    default:
      log_not_supported();
    };
    return false;
  }
};

} // namespace vgbrep

#endif
