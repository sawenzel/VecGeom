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
  static const char *const data[] = {"no_surf",   "planar", "cylindrical", "conical",
                                     "spherical", "torus",  "elliptical",  "arb4"};
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

template <>
VECGEOM_FORCE_INLINE char const *to_cstring<AxisType>(AxisType type)
{
  static const char *const data[] = {"x-axis", "y-axis", "z-axis", "r-axis", "phi-axis", "xy-grid", "no-axis"};
  assert(size_t(type) * sizeof(const char *) < sizeof(data));
  return data[static_cast<int>(type)];
}

using LogicExpressionCPU = std::vector<logic_int>;

// Placeholder (on host) for all surfaces belonging to a volume. An array of those will be indexed
// by the logical volume id. Also an intermediate helper for building portals.
// Note: the local surfaces defined by solids will have local references that will be changed by
// the flattening process, depending on the scene on which the parent volume will be flattened
struct VolumeShellCPU {
  std::vector<int> fSurfaces;             ///< Local surface id's for this volume
  std::vector<int> fExitingSurfaces;      ///< Local surface id's for this volume, excluding virtual ones
  std::vector<int> fEnteringSurfaces;     ///< Local surface id's for all surfaces of daughters, excluding virtual ones
  std::vector<int> fEnteringSurfacesPvol; ///< Global PVol ids for the daughter surfaces of this volume
  std::vector<int> fEnteringSurfacesPvolTrans; ///< Array of ids to daughter placed volume transformations
  std::vector<int> fEnteringSurfacesLvolIds;   ///< Array of ids to daughter logical volumes
  std::vector<int> fDaughterPvolIds;           ///< Global PV Ids of the daughter PVs of this Volume
  std::vector<int> fDaughterPvolTrans;         ///< Transformations of the daughter PVs of this Volume

  LogicExpressionCPU fLogic; ///< Logic expression for the solid
  bool fSimplified{false};   ///< The logic was simplified
  int fBVH{0};
};

/// @brief Helper for dividing a side in equal slices along one axis, keeping frame candidates in each slice
/// @details The range given by the side extent is divided in equal slices along an axis. Each
///          slice intersects a number of frames. The slice is found besed on the crossing point, then the full
///          frame loops are reduced to the list of candidates in that slice.
struct SideDivisionCPU {
  using VecInt_t = std::vector<int>;
  double fStartU{0.};                ///< Division start on primary division axis
  double fStepU{0.};                 ///< Division step on primary division axis
  double fStartV{0.};                ///< Division start on secondary axis (if any)
  double fStepV{0.};                 ///< Division step on secondary axis (if any)
  unsigned short fNslices{0};        ///< Total number of slices
  unsigned short fNslicesU{0};       ///< Number of slices on primary division axis
  unsigned short fNslicesV{0};       ///< Number of slices on secondary division axis
  AxisType fAxis{AxisType::kNoAxis}; ///< Division axis
  std::vector<VecInt_t> fSlices;     ///< Array of slices

  // Methods
  SideDivisionCPU() = default;
  SideDivisionCPU(AxisType axis, double coord_min, double coord_max, int ncand)
      : fStartU(coord_min), fStepU((coord_max - coord_min) / ncand), fNslices(ncand), fAxis(axis)
  {
    fSlices.insert(fSlices.end(), ncand, {});
  }

  SideDivisionCPU(AxisType axis, double coord_minU, double coord_maxU, double coord_minV, double coord_maxV, int ncand)
      : fAxis(axis)
  {
    fNslicesU = 1 + std::sqrt(static_cast<double>(ncand)); // rounding happens
    fNslicesV = ncand / fNslicesU;                         // rounding happens
    fNslices  = fNslicesU * fNslicesV;
    fStartU   = coord_minU;
    fStepU    = (coord_maxU - coord_minU) / fNslicesU;
    fStartV   = coord_minV;
    fStepV    = (coord_maxV - coord_minV) / fNslicesV;
    fSlices.insert(fSlices.end(), fNslices, {});
  }

  void AddCandidate(int icand, double coord_min, double coord_max)
  {
    int istart = (coord_min - fStartU - vecgeom::kToleranceDist<double>) / fStepU;
    int iend   = (coord_max - fStartU + vecgeom::kToleranceDist<double>) / fStepU;
    for (int i = istart; i <= iend && i < fNslices; ++i) {
      if (i < 0) continue;
      fSlices[i].push_back(icand);
    }
  }

  void AddCandidate(int icand, double umin, double umax, double vmin, double vmax)
  {
    int istartU = (umin - fStartU - vecgeom::kToleranceDist<double>) / fStepU;
    int iendU   = (umax - fStartU + vecgeom::kToleranceDist<double>) / fStepU;
    int istartV = (vmin - fStartV - vecgeom::kToleranceDist<double>) / fStepV;
    int iendV   = (vmax - fStartV + vecgeom::kToleranceDist<double>) / fStepV;
    for (int i = istartU; i <= iendU && i < fNslicesU; ++i) {
      for (int j = istartV; j <= iendV && j < fNslicesV; ++j) {
        if (i < 0 || j < 0) continue;
        fSlices[i * fNslicesV + j].push_back(icand);
      }
    }
  }

  template <typename Real_t>
  void CopyTo(SideDivision<Real_t> &div, SliceCand *slices, int *candidates)
  {
    div.fStartU   = fStartU;
    div.fStepU    = fStepU;
    div.fStartV   = fStartV;
    div.fStepV    = fStepV;
    div.fNslices  = fNslices;
    div.fNslicesU = fNslicesU;
    div.fNslicesV = fNslicesV;
    div.fAxis     = fAxis;
    div.fSlices   = slices;
    auto cand     = candidates;
    for (size_t i = 0; i < fSlices.size(); ++i) {
      slices[i].fNcand      = fSlices[i].size();
      slices[i].fCandidates = cand;
      for (int j = 0; j < slices[i].fNcand; ++j)
        cand[j] = fSlices[i][j];
      cand += slices[i].fNcand;
    }
  }

  template <typename Real_t>
  size_t GetSizeObject() const
  {
    return sizeof(SideDivision<Real_t>);
  }

  template <typename Real_t>
  size_t GetSizeSlices() const
  {
    return fNslices * sizeof(SliceCand);
  }

  size_t GetNcandidates() const
  {
    size_t size = 0;
    for (auto i = 0; i < fNslices; ++i)
      size += fSlices[i].size();
    return size;
  }

  size_t GetSizeCandidates() const { return GetNcandidates() * sizeof(int); }

  template <typename Real_t>
  size_t GetSize() const
  {
    size_t size = GetSizeObject<Real_t>() + GetSizeSlices<Real_t>() + GetSizeCandidates();
    return size;
  }

  void Print() const
  {
    if (fAxis == AxisType::kXY) {
      std::cout << to_cstring(fAxis) << " division: fNslices = " << fNslices << ", fNslicesU = " << fNslicesU
                << ", fNslicesV = " << fNslicesV << ", fStartU = " << fStartU << ", fStepU = " << fStepU
                << ", fStartV = " << fStartV << ", fStepV = " << fStepV << ", efficiency = " << 100 * Efficiency()
                << " %\n";
    } else {
      std::cout << to_cstring(fAxis) << " division: fNslices = " << fNslices << ", fStart = " << fStartU
                << ", fStep = " << fStepU << ", efficiency = " << 100 * Efficiency() << " %\n";
    }
    for (auto i = 0; i < fNslices; ++i)
      std::cout << fSlices[i].size() << " ";
    std::cout << "\n";
  }

  double Efficiency() const
  {
    // assert(fSlices[0].size() * fSlices[fNslices - 1].size() > 0); // except kXY
    double average = 0;
    for (auto i = 0; i < fNslices; ++i)
      average += fSlices[i].size();
    average /= fNslices;
    double eff = (fNslices - average) / (average * (fNslices - 1));
    return eff;
  }
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
  std::vector<QuadMask_t> fQuadMasks;  ///< quadrilateral masks
  std::vector<CylData_t> fCylSphData;  ///< data for cyl surfaces
  std::vector<ConeData_t> fConeData;   ///< data for conical surfaces
  std::vector<EllipData_t> fEllipData; ///< data for elliptical surfaces
  std::vector<TorusData_t> fTorusData; ///< data for torus surfaces
  std::vector<Arb4Data_t> fArb4Data;   ///< data for Arb4 surfaces
  std::vector<TransformationMP<Real_t>> fPVolTrans;   ///< Transformations to placed volumes
  std::vector<FramedSurface<Real_t>> fLocalSurfaces;  ///< local surfaces per logical volume
  std::vector<FramedSurface<Real_t>> fFramedSurf;     ///< global surfaces
  std::vector<CommonSurface<Real_t>> fCommonSurfaces; ///< common surfaces
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

  std::vector<SideDivisionCPU> fSideDivisions; ///< list of divisions for all sides

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
    std::vector<TransformationMP<Real_t>>().swap(fPVolTrans);
    std::vector<FramedSurface<Real_t>>().swap(fLocalSurfaces);
    std::vector<FramedSurface<Real_t>>().swap(fFramedSurf);
    std::vector<CommonSurface<Real_t>>().swap(fCommonSurfaces);
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
    std::vector<SideDivisionCPU>().swap(fSideDivisions);
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

  WindowMask_t const &GetWindowMask(int id) const { return fWindowMasks[id]; }
  RingMask_t const &GetRingMask(int id) const { return fRingMasks[id]; }
  ZPhiMask_t const &GetZPhiMask(int id) const { return fZPhiMasks[id]; }
  TriangleMask_t const &GetTriangleMask(int id) const { return fTriangleMasks[id]; }
  QuadMask_t const &GetQuadMask(int id) const { return fQuadMasks[id]; }

  CylData_t const &GetCylData(int id) const { return fCylSphData[id]; }
  SphData_t const &GetSphData(int id) const { return fCylSphData[id]; }
  ConeData_t const &GetConeData(int id) const { return fConeData[id]; }
  EllipData_t const &GetEllipData(int id) const { return fEllipData[id]; }
  TorusData_t const &GetTorusData(int id) const { return fTorusData[id]; }
  Arb4Data_t const &GetArb4Data(int id) const { return fArb4Data[id]; }

  /// @brief Trampoline function to to the frame embedding checker
  /// @param f1 Parent framed surface
  /// @param f2 Child framed surface
  /// @return Child is embedded in parent
  bool IsEmbedding(FramedSurface<Real_t> const &f1, FramedSurface<Real_t> const &f2)
  {
    auto log_not_supported = [&]() {
      VECGEOM_LOG(error) << "Embedding check " << to_cstring(f1.fFrame.type) << " - " << to_cstring(f2.fFrame.type)
                         << " not supported";
    };
    TransformationMP<Real_t> const &t1 = f1.fTrans;
    TransformationMP<Real_t> const &t2 = f2.fTrans;
    TransformationMP<Real_t> trans     = t2 * t1.Inverse();
    trans.SetProperties();

    switch (f1.fFrame.type) {
    case FrameType::kRing: {
      RingMask_t const mask1 = GetRingMask(f1.fFrame.id);
      switch (f2.fFrame.type) {
      case FrameType::kRing: {
        RingMask_t const mask2 = GetRingMask(f2.fFrame.id);
        return FrameChecker<Real_t, RingMask_t, RingMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kWindow: {
        WindowMask_t const mask2 = GetWindowMask(f2.fFrame.id);
        return FrameChecker<Real_t, RingMask_t, WindowMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kTriangle: {
        TriangleMask_t const mask2 = GetTriangleMask(f2.fFrame.id);
        return FrameChecker<Real_t, RingMask_t, TriangleMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kQuadrilateral: {
        QuadMask_t const mask2 = GetQuadMask(f2.fFrame.id);
        return FrameChecker<Real_t, RingMask_t, QuadMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    case FrameType::kZPhi: {
      ZPhiMask_t const mask1 = GetZPhiMask(f1.fFrame.id);
      switch (f2.fFrame.type) {
      case FrameType::kZPhi: {
        ZPhiMask_t const mask2 = GetZPhiMask(f2.fFrame.id);
        return FrameChecker<Real_t, ZPhiMask_t, ZPhiMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    case FrameType::kWindow: {
      WindowMask_t const mask1 = GetWindowMask(f1.fFrame.id);
      switch (f2.fFrame.type) {
      case FrameType::kRing: {
        RingMask_t const mask2 = GetRingMask(f2.fFrame.id);
        return FrameChecker<Real_t, WindowMask_t, RingMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kWindow: {
        WindowMask_t const mask2 = GetWindowMask(f2.fFrame.id);
        return FrameChecker<Real_t, WindowMask_t, WindowMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kTriangle: {
        TriangleMask_t const mask2 = GetTriangleMask(f2.fFrame.id);
        return FrameChecker<Real_t, WindowMask_t, TriangleMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kQuadrilateral: {
        QuadMask_t const mask2 = GetQuadMask(f2.fFrame.id);
        return FrameChecker<Real_t, WindowMask_t, QuadMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    case FrameType::kTriangle: {
      TriangleMask_t const mask1 = GetTriangleMask(f1.fFrame.id);
      switch (f2.fFrame.type) {
      case FrameType::kRing: {
        RingMask_t const mask2 = GetRingMask(f2.fFrame.id);
        return FrameChecker<Real_t, TriangleMask_t, RingMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kWindow: {
        WindowMask_t const mask2 = GetWindowMask(f2.fFrame.id);
        return FrameChecker<Real_t, TriangleMask_t, WindowMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kTriangle: {
        TriangleMask_t const mask2 = GetTriangleMask(f2.fFrame.id);
        return FrameChecker<Real_t, TriangleMask_t, TriangleMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kQuadrilateral: {
        QuadMask_t const mask2 = GetQuadMask(f2.fFrame.id);
        return FrameChecker<Real_t, TriangleMask_t, QuadMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      default:
        log_not_supported();
      };
      break;
    }
    case FrameType::kQuadrilateral: {
      QuadMask_t const mask1 = GetQuadMask(f1.fFrame.id);
      switch (f2.fFrame.type) {
      case FrameType::kRing: {
        RingMask_t const mask2 = GetRingMask(f2.fFrame.id);
        return FrameChecker<Real_t, QuadMask_t, RingMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kWindow: {
        WindowMask_t const mask2 = GetWindowMask(f2.fFrame.id);
        return FrameChecker<Real_t, QuadMask_t, WindowMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kTriangle: {
        TriangleMask_t const mask2 = GetTriangleMask(f2.fFrame.id);
        return FrameChecker<Real_t, QuadMask_t, TriangleMask_t>::IsEmbedding(mask1, mask2, trans);
      }
      case FrameType::kQuadrilateral: {
        QuadMask_t const mask2 = GetQuadMask(f2.fFrame.id);
        return FrameChecker<Real_t, QuadMask_t, QuadMask_t>::IsEmbedding(mask1, mask2, trans);
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
