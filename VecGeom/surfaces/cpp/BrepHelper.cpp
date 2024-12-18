#include <VecGeom/surfaces/BrepHelper.h>
#include <VecGeom/surfaces/base/CpuTypes.h>

namespace vgbrep {

template <typename Real_i>
struct SideIteratorSlices {
  using VecInt_t = std::vector<int>;
  std::vector<const VecInt_t *> fSlices; ///< Division slices to be iterated
  int fNslices{0};                       ///< Number of slices to iterate
  int fNsurf{0};                         ///< Number of surfaces on the side
  int fCrtSlice{0};                      ///< Current slice
  int fCrt{0};                           ///< Current index in slice
  bool fDone{true};                      ///< Iteration is done

  SideIteratorSlices(const Side &side, Vector3D<Real_i> const &amin, Vector3D<Real_i> const &amax,
                     CPUsurfData<vecgeom::Precision> const &cpudata)
  {
    if (side.fDivision < 0) {
      // If no division on the side, initiate loop mode
      fNsurf = side.fNsurf;
      fDone  = false;
      return;
    }
    auto const &div = cpudata.fSideDivisions[side.fDivision];
    int u1, u2, v1, v2;
    auto nslices = div.GetNslices(amin, amax, u1, u2, v1, v2);
    if (nslices == 0) return;
    fDone = false;
    for (auto indU = u1; indU <= u2; ++indU) {
      for (auto indV = v1; indV <= v2; ++indV) {
        auto slice = div.GetSlice(indU, indV);
        if (slice->size()) fSlices.push_back(slice);
      }
    }
    fNslices = fSlices.size();
    if (fNslices == 0) fDone = true;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool Done() const { return fDone; }

  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE int operator()()
  {
    if (fNsurf > 0) return fCrt; // loop mode
    return (*fSlices[fCrtSlice])[fCrt];
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SideIteratorSlices &operator++()
  {
    if (fNsurf > 0) {
      // loop mode
      if (++fCrt > (fNsurf - 1)) fDone = true;
      return *this;
    }
    auto const &slice = *fSlices[fCrtSlice];
    fCrt              = (fCrt + 1) % slice.size();
    if (fCrt == 0) fCrtSlice++;
    if (fCrtSlice == fNslices) fDone = true;
    return *this;
  }
};

template <typename Real_t>
BrepHelper<Real_t>::BrepHelper()
    : fSurfData(&SurfData<Real_t>::Instance()), fCPUdata(CPUsurfData<vecgeom::Precision>::Instance())
{
  static_assert(std::is_same<CPUsurfData_t, CPUsurfData<vecgeom::Precision>>::value,
                "CPUsurfData_t must be CPUsurfData<vecgeom::Precision>");
  static_assert(std::is_same<Real_t, double>::value || std::is_same<Real_t, float>::value,
                "Real_t must be either double or float");
}

template <typename Real_t>
BrepHelper<Real_t> &BrepHelper<Real_t>::Instance()
{
  static BrepHelper<Real_t> instance;
  return instance;
}

template <typename Real_t>
BrepHelper<Real_t>::~BrepHelper()
{
  if (fSurfData) ClearData();
}

template <typename Real_t>
void BrepHelper<Real_t>::ClearData()
{
  // Dispose of surface data and shrink the container
  fCPUdata.Clear();
  // delete fSurfData;
  fSurfData->Clear();
  fSurfData = nullptr;
}

template <typename Real_t>
void BrepHelper<Real_t>::SortSides(int common_id)
{
  // lambda to detect surfaces on a side that have identical frame
  auto sortFrames = [&](Side &side) {
    if (!side.fNsurf) return;
    std::sort(side.fSurfaces, side.fSurfaces + side.fNsurf,
              [&](int i, int j) { return fCPUdata.fFramedSurf[i] < fCPUdata.fFramedSurf[j]; });
  };

  // lambda to find all parent frames on one side
  auto findParentFramedSurf = [&](Side &side) {
    if (!side.fNsurf) return;
    int top_parent  = side.fNsurf - 1;
    int num_parents = 0;
    // if there is a parent, it can only be at the last position after sorting
    for (auto parent_ind = top_parent; parent_ind >= 0; --parent_ind) {
      auto &parent_frame = fCPUdata.fFramedSurf[side.fSurfaces[parent_ind]];
      if (parent_frame.fParent < 0) num_parents++;
      // Non-embedding frames may be several on the side
      auto parent_navind = parent_frame.fState;
      // re-parent frames before if they have descendent states
      for (int i = 0; i < parent_ind; ++i) {
        auto &child_frame = fCPUdata.fFramedSurf[side.fSurfaces[i]];
        auto navind       = child_frame.fState;
        if (vecgeom::NavigationState::IsDescendentImpl(navind, parent_navind)) {
          // Check if the frame is embedded in ANY of the parents
          if (!child_frame.fEmbedded) {
            auto intersect_type = fCPUdata.CheckFrames(parent_frame, child_frame);
            child_frame.fEmbedded =
                (intersect_type == FrameIntersect::kEmbedding) || (intersect_type == FrameIntersect::kEqual);
          }
          child_frame.fParent = parent_ind;
        }
      }
    }

    // rerun loop to check if non-embedded children have embedding parents
    for (auto parent_ind = top_parent; parent_ind >= 0; --parent_ind) {
      auto &parent_frame = fCPUdata.fFramedSurf[side.fSurfaces[parent_ind]];
      // Non-embedding frames may be several on the side
      auto parent_navind = parent_frame.fState;
      for (int i = 0; i < parent_ind; ++i) {
        auto &child_frame = fCPUdata.fFramedSurf[side.fSurfaces[i]];
        auto navind       = child_frame.fState;
        if (vecgeom::NavigationState::IsDescendentImpl(navind, parent_navind)) {
          // Check if the frame is embedded in ANY of the parents
          if (!child_frame.fEmbedded) {

            // if there are multiple parents on the same surface (e.g., on a polycone) we should not print the warning
            // of non-embedded surfaces having embedded parents since the parents are in fact embedding, but we would
            // need to check the daughters against the union of their parents. since parents are next to each other, we
            // just check if the previous or next surface to the parent has the same state as the checked surface to see
            // if there are multiple parents Note that this might suppress real overlaps for all surfaces with multiple
            // parents
            bool multi_parent = false;
            if (parent_ind != top_parent)
              multi_parent = fCPUdata.fFramedSurf[side.fSurfaces[parent_ind + 1]].fState == parent_navind;
            if (parent_ind != 0)
              multi_parent |= fCPUdata.fFramedSurf[side.fSurfaces[parent_ind - 1]].fState == parent_navind;
            if (multi_parent) continue;
            if (parent_frame.fEmbedding && child_frame.fLogicId == 0) {
              VECGEOM_LOG(warning) << "Non-embedded frame " << i << " having embedding parent " << parent_ind
                                   << " on CS " << common_id;
              if (fVerbose > 0) {
                PrintFramedSurface(child_frame);
                PrintFramedSurface(parent_frame);
                // Debugging only:
                fCPUdata.CheckFrames(parent_frame, child_frame);
              }
            }
          }
        }
      }
    }
    side.fNumParents = num_parents;
    assert(num_parents > 0);
  };

  sortFrames(fCPUdata.fCommonSurfaces[common_id].fLeftSide);
  sortFrames(fCPUdata.fCommonSurfaces[common_id].fRightSide);
  findParentFramedSurf(fCPUdata.fCommonSurfaces[common_id].fLeftSide);
  findParentFramedSurf(fCPUdata.fCommonSurfaces[common_id].fRightSide);
}

template <typename Real_t>
void BrepHelper<Real_t>::ComputeDefaultStates(int common_id)
{
  using vecgeom::NavigationState;
  // Computes the default states for each side of a common surface
  Side &left  = fCPUdata.fCommonSurfaces[common_id].fLeftSide;
  Side &right = fCPUdata.fCommonSurfaces[common_id].fRightSide;
  assert(left.fNsurf > 0 || right.fNsurf > 0);

  NavIndex_t default_ind = 0;

  // A lambda that finds the deepest common ancestor between 2 states
  auto getCommonState = [&](NavIndex_t const &s1, NavIndex_t const &s2) {
    NavIndex_t a1(s1), a2(s2);
    // Bring both states at the same level
    while (NavigationState::GetLevelImpl(a1) > NavigationState::GetLevelImpl(a2))
      NavigationState::PopImpl(a1);
    while (NavigationState::GetLevelImpl(a2) > NavigationState::GetLevelImpl(a1))
      NavigationState::PopImpl(a2);

    // Pop until we reach the same state
    while (a1 != a2) {
      NavigationState::PopImpl(a1);
      NavigationState::PopImpl(a2);
    }
    return a1;
  };

  int minlevel = 10000; // this is a big-enough number as level
  for (int isurf = 0; isurf < left.fNsurf; ++isurf) {
    auto navind = fCPUdata.fFramedSurf[left.fSurfaces[isurf]].fState;
    minlevel    = std::min(minlevel, (int)NavigationState::GetLevelImpl(navind));
  }
  for (int isurf = 0; isurf < right.fNsurf; ++isurf) {
    auto navind = fCPUdata.fFramedSurf[right.fSurfaces[isurf]].fState;
    minlevel    = std::min(minlevel, (int)NavigationState::GetLevelImpl(navind));
  }

  // initialize the default state
  if (left.fNsurf > 0)
    default_ind = fCPUdata.fFramedSurf[left.fSurfaces[0]].fState;
  else if (right.fNsurf > 0)
    default_ind = fCPUdata.fFramedSurf[right.fSurfaces[0]].fState;

  for (int isurf = 0; isurf < left.fNsurf; ++isurf)
    default_ind = getCommonState(default_ind, fCPUdata.fFramedSurf[left.fSurfaces[isurf]].fState);
  for (int isurf = 0; isurf < right.fNsurf; ++isurf)
    default_ind = getCommonState(default_ind, fCPUdata.fFramedSurf[right.fSurfaces[isurf]].fState);

  if (NavigationState::GetLevelImpl(default_ind) == minlevel) NavigationState::PopImpl(default_ind);
  fCPUdata.fCommonSurfaces[common_id].fDefaultState = default_ind;
}

template <typename Real_t>
WindowMask<double> BrepHelper<Real_t>::GetPlanarFrameExtent(
    FramedSurface<Real_t, TransformationMP<Real_t>> const &framed_surf, TransformationMP<Real_t> const &trans)
{
  // This is a helper-lambda that updates the extent based on corner coordinates
  auto updateExtent = [](WindowMask<double> &e, Vector3D<double> const &pt) {
    e.rangeU[0] = std::min(e.rangeU[0], pt[0]);
    e.rangeU[1] = std::max(e.rangeU[1], pt[0]);
    e.rangeV[0] = std::min(e.rangeV[0], pt[1]);
    e.rangeV[1] = std::max(e.rangeV[1], pt[1]);
  };
  // Setting initial mask for an extent.
  WindowMask<double> ext{vecgeom::kInfLength, -vecgeom::kInfLength, vecgeom::kInfLength, -vecgeom::kInfLength};
  FrameType frame_type = framed_surf.fFrame.type;
  Vector3D<double> local;

  WindowMask<double> extentL;
  // Calculating the limits
  switch (frame_type) {
  case FrameType::kWindow: {
    auto const &maskLocal = fCPUdata.fWindowMasks[framed_surf.fFrame.id];
    maskLocal.GetExtent(extentL);
    break;
  }
  case FrameType::kRing: {
    auto const &maskLocal = fCPUdata.fRingMasks[framed_surf.fFrame.id];
    maskLocal.GetExtent(extentL);
    break;
  }
  case FrameType::kQuadrilateral: {
    WindowMask_t extLocal;
    auto const &quad = fCPUdata.fQuadMasks[framed_surf.fFrame.id];
    quad.GetExtent(extentL);
    break;
  }
  case FrameType::kTriangle: {
    TriangleMask_t extLocal;
    auto const &maskLocal = fCPUdata.fTriangleMasks[framed_surf.fFrame.id];
    maskLocal.GetExtent(extentL);
    break;
  }
  default:
    assert(0 && "Not implemented");
  }

  // This part updates extent
  local = trans.InverseTransform(Vector3D<double>{extentL.rangeU[0], extentL.rangeV[0], 0});
  updateExtent(ext, local);
  local = trans.InverseTransform(Vector3D<double>{extentL.rangeU[0], extentL.rangeV[1], 0});
  updateExtent(ext, local);
  local = trans.InverseTransform(Vector3D<double>{extentL.rangeU[1], extentL.rangeV[1], 0});
  updateExtent(ext, local);
  local = trans.InverseTransform(Vector3D<double>{extentL.rangeU[1], extentL.rangeV[0], 0});
  updateExtent(ext, local);
  return ext;
}

template <typename Real_t>
WindowMask<double> BrepHelper<Real_t>::ComputePlaneExtent(const Side &side)
{
  // This is a helper-lambda that updates extents
  // for all sides of common plane surfaces
  auto updatePlaneExtent = [](WindowMask<double> &e, WindowMask<double> const &elocal) {
    e.rangeU[0] = std::min(e.rangeU[0], elocal.rangeU[0]);
    e.rangeU[1] = std::max(e.rangeU[1], elocal.rangeU[1]);
    e.rangeV[0] = std::min(e.rangeV[0], elocal.rangeV[0]);
    e.rangeV[1] = std::max(e.rangeV[1], elocal.rangeV[1]);
  };

  // Setting initial mask for an extent.
  WindowMask<double> ext{vecgeom::kInfLength, -vecgeom::kInfLength, vecgeom::kInfLength, -vecgeom::kInfLength};

  // loop through all extents on a side:
  for (int i = 0; i < side.fNsurf; ++i) {
    // Compute extent for the frame
    auto &framed_surf = fCPUdata.fFramedSurf[side.fSurfaces[i]];
    auto extentL      = GetPlanarFrameExtent(framed_surf, framed_surf.fTrans);
    // This part updates extent
    updatePlaneExtent(ext, extentL);
  }

  return ext;
}

template <typename Real_t>
ZPhiMask<double> BrepHelper<Real_t>::ComputeCylinderExtent(const Side &side)
{
  auto const &sideext = fCPUdata.GetZPhiMask(fCPUdata.fFramedSurf[side.fSurfaces[0]].fFrame.id);
  auto sideext_local  = sideext.InverseTransform(fCPUdata.fFramedSurf[side.fSurfaces[0]].fTrans);

  // loop over remaining frames on the side
  for (int i = 1; i < side.fNsurf; ++i) {
    // convert extent of current frame to local coordinates
    auto &framed_surf = fCPUdata.fFramedSurf[side.fSurfaces[i]];
    // Transform the ZPhi mask to the local system
    auto const &extLocal = fCPUdata.GetZPhiMask(framed_surf.fFrame.id);
    auto extFrame        = extLocal.InverseTransform(framed_surf.fTrans);
    // Combine with current extent
    bool success = sideext_local.CombineWith(extFrame);
    if (!success) {
      VECGEOM_LOG(critical) << "ComputeCylinderExtent error";
    }
  }

  return sideext_local;
}

template <typename Real_t>
int BrepHelper<Real_t>::ComputeCylinderDivision(Side &side, ZPhiMask<double> extent_full)
{
  // return -1;
  SideDivisionCPU divisionZ(AxisType::kZ, extent_full.rangeZ[0], extent_full.rangeZ[1], side.fNsurf);
  auto updateRangeZ = [](double z, double &zmin, double &zmax) {
    zmin = std::min(zmin, z);
    zmax = std::max(zmax, z);
  };
  // loop through all extents on a side:
  for (int i = 0; i < side.fNsurf; ++i) {
    auto &framed_surf = fCPUdata.fFramedSurf[side.fSurfaces[i]];
    assert(framed_surf.fFrame.type == FrameType::kZPhi);

    Vector3D<double> local;
    double zmin{vecgeom::InfinityLength<Real_t>()}, zmax{-vecgeom::InfinityLength<Real_t>()};
    auto const &maskLocal = fCPUdata.fZPhiMasks[framed_surf.fFrame.id];
    local                 = framed_surf.fTrans.InverseTransform(Vector3D<double>{0, 0, maskLocal.rangeZ[0]});
    updateRangeZ(local[2], zmin, zmax);
    local = framed_surf.fTrans.InverseTransform(Vector3D<double>{0, 0, maskLocal.rangeZ[1]});
    updateRangeZ(local[2], zmin, zmax);
    divisionZ.AddCandidate(i, zmin, zmax);
  }
  if (divisionZ.Efficiency() > 0.01) {
    // divisionZ.Print();
    side.fDivision = fCPUdata.fSideDivisions.size();
    fCPUdata.fSideDivisions.push_back(divisionZ);
  }
  return side.fDivision;
}

template <typename Real_t>
void BrepHelper<Real_t>::CountTraversals(const CommonSurface<Real_t> &surf, int &nfound, int &ntotal) const
{
  auto count_per_side = [&](Side const &side) {
    ntotal += side.fNsurf;
    for (int i = 0; i < side.fNsurf; ++i) {
      auto const &frame = fCPUdata.fFramedSurf[side.fSurfaces[i]];
      if (frame.fTraversal > -2) nfound++;
    }
  };
  count_per_side(surf.fLeftSide);
  count_per_side(surf.fRightSide);
}

template <typename Real_t>
void BrepHelper<Real_t>::ComputeTraversalForFrame(int common_id, bool left, int iframe)
{
  using Vector3D         = Vector3D<vecgeom::Precision>;
  auto &surf             = fCPUdata.fCommonSurfaces[common_id];
  Side const &side       = left ? surf.fLeftSide : surf.fRightSide;
  auto &thisside_frame   = fCPUdata.fFramedSurf[side.fSurfaces[iframe]];
  Side const &other_side = left ? surf.fRightSide : surf.fLeftSide;
  auto setTraversal      = [](FramedSurface<double, TransformationMP<double>> &frame, int iframe) {
    // if not set just set it
    if (frame.fTraversal == -2) {
      frame.fTraversal = iframe;
    } else {
      // otherwise set it to the minimum frame index (deepest history)
      if (iframe < frame.fTraversal) frame.fTraversal = iframe;
    }
  };

  // Compute extent of the frame in its local reference frame

  // Loop frames on the other_side and find the candidates
  bool has_div = other_side.fDivision >= 0;
  Vector3D amin, amax;
  if (has_div) {
    if (surf.fType == SurfaceType::kPlanar) {
      if (fCPUdata.fSideDivisions[other_side.fDivision].fAxis == AxisType::kR) {
        auto const &ringMask = fCPUdata.fRingMasks[thisside_frame.fFrame.id];
        amin[0]              = ringMask.rangeR[0];
        amax[0]              = ringMask.rangeR[1];
      } else {
        // Calculate the aligned bounding box of the frame
        auto extentL = GetPlanarFrameExtent(thisside_frame, thisside_frame.fTrans);
        amin.Set(extentL.rangeU[0], extentL.rangeV[0], 0.);
        amax.Set(extentL.rangeU[1], extentL.rangeV[1], 0.);
      }
    } else if (surf.fType == SurfaceType::kCylindrical || surf.fType == SurfaceType::kConical) {
      Vector3D local;
      amin.Set(0., 0., vecgeom::InfinityLength<Real_t>());
      amax.Set(0., 0., -vecgeom::InfinityLength<Real_t>());
      auto const &maskLocal = fCPUdata.fZPhiMasks[thisside_frame.fFrame.id];
      local                 = thisside_frame.fTrans.InverseTransform(Vector3D{0, 0, maskLocal.rangeZ[0]});
      amin[2]               = std::min(amin[2], local[2]);
      amax[2]               = std::max(amax[2], local[2]);
      local                 = thisside_frame.fTrans.InverseTransform(Vector3D{0, 0, maskLocal.rangeZ[1]});
      amin[2]               = std::min(amin[2], local[2]);
      amax[2]               = std::max(amax[2], local[2]);
    }
  }

  bool disjoint = true;
  std::set<int> candidates;
  // check with the frames on the other side
  for (SideIteratorSlices it(other_side, amin, amax, fCPUdata); !it.Done(); ++it) {
    auto j = it();
    if (!candidates.insert(j).second) continue;
    auto &otherside_frame = fCPUdata.fFramedSurf[other_side.fSurfaces[j]];

    // frame on the other side already marked "disjoint", don't check
    if (otherside_frame.fTraversal == -1) continue;

    // check intersection type
    auto intersect_type = fCPUdata.CheckFrames(thisside_frame, otherside_frame);
    if (intersect_type == FrameIntersect::kNoIntersect) continue;

    // Other than kNoIntersect is not disjoint
    disjoint = false;
    if (intersect_type == FrameIntersect::kIntersect) {
      // If frames intersect, we cannot have direct traversals on either of them
      thisside_frame.fTraversal  = -3;
      otherside_frame.fTraversal = -3;
      break;
    }
    if (intersect_type == FrameIntersect::kEmbedding) {
      // crossing from otherside_frame always to thisside_frame
      if (thisside_frame.fLogicId)
        otherside_frame.fTraversal = -3;
      else
        setTraversal(otherside_frame, iframe);
      // thisside_frame is overlapping
      thisside_frame.fTraversal = -3;
      break;
    }
    if (intersect_type == FrameIntersect::kEqual) {
      // frames transition from one to another
      if (otherside_frame.fLogicId)
        thisside_frame.fTraversal = -3;
      else
        setTraversal(thisside_frame, j);

      if (thisside_frame.fLogicId)
        otherside_frame.fTraversal = -3;
      else
        setTraversal(otherside_frame, iframe);
    }
    if (intersect_type == FrameIntersect::kEmbedded) {
      // crossing from thisside_frame always to otherside_frame
      if (otherside_frame.fLogicId)
        thisside_frame.fTraversal = -3;
      else
        setTraversal(thisside_frame, j);
      otherside_frame.fTraversal = -3;
    }
  }
  if (disjoint) thisside_frame.fTraversal = -1;
}

template <typename Real_t>
void BrepHelper<Real_t>::ComputeTraversalFrames(int common_id, bool left)
{
  auto &surf             = fCPUdata.fCommonSurfaces[common_id];
  Side const &side       = left ? surf.fLeftSide : surf.fRightSide;
  Side const &other_side = left ? surf.fRightSide : surf.fLeftSide;
  if (other_side.fNsurf == 0) {
    // nothing on the other side, mark all transitions as non-entering
    for (int i = 0; i < side.fNsurf; ++i)
      fCPUdata.fFramedSurf[side.fSurfaces[i]].fTraversal = -1;
    return;
  }

  // Loop frames on the side and find the candidates on the other side
  for (int i = 0; i < side.fNsurf; ++i) {
    auto &thisside_frame = fCPUdata.fFramedSurf[side.fSurfaces[i]];
    // skip logical frames, and checked frames which are already marked disjoint or overlapping
    if (thisside_frame.fTraversal == -3 || thisside_frame.fTraversal == -1) continue;
    // check with the frames on the other side
    ComputeTraversalForFrame(common_id, left, i);
  }
}

template <typename Real_t>
int BrepHelper<Real_t>::ComputePlaneDivision(Side &side, WindowMask<double> extent_full)
{
  // Special case if all frames are rings placed with id transformation
  bool all_rings_id = true; // all frames are rings with id transformation
  double ring_max   = -vecgeom::kInfLength;
  double ring_min   = vecgeom::kInfLength;
  for (int i = 0; i < side.fNsurf; ++i) {
    auto &framed_surf    = fCPUdata.fFramedSurf[side.fSurfaces[i]];
    FrameType frame_type = framed_surf.fFrame.type;
    if (frame_type == FrameType::kRing && !framed_surf.fTrans.HasTranslation()) {
      auto const &maskRing = fCPUdata.fRingMasks[framed_surf.fFrame.id];
      ring_min             = std::min(maskRing.rangeR[0], ring_min);
      ring_max             = std::max(maskRing.rangeR[1], ring_max);
    } else {
      all_rings_id = false;
      break;
    }
  }

  // Create all supported division types. We could increase the number of slices potentially.
  SideDivisionCPU divisionX(AxisType::kX, extent_full.rangeU[0], extent_full.rangeU[1], side.fNsurf);
  SideDivisionCPU divisionY(AxisType::kY, extent_full.rangeV[0], extent_full.rangeV[1], side.fNsurf);
  SideDivisionCPU divisionR(AxisType::kR, ring_min, ring_max, side.fNsurf);
  SideDivisionCPU divisionXY(AxisType::kXY, extent_full.rangeU[0], extent_full.rangeU[1], extent_full.rangeV[0],
                             extent_full.rangeV[1], side.fNsurf);
  // loop through all extents on a side:
  for (int i = 0; i < side.fNsurf; ++i) {
    auto &framed_surf = fCPUdata.fFramedSurf[side.fSurfaces[i]];
    if (all_rings_id) {
      auto const &extentRing = fCPUdata.fRingMasks[framed_surf.fFrame.id];
      divisionR.AddCandidate(i, extentRing.rangeR[0], extentRing.rangeR[1]);
    }
    auto ext = GetPlanarFrameExtent(framed_surf, framed_surf.fTrans);
    divisionX.AddCandidate(i, ext.rangeU[0], ext.rangeU[1]);
    divisionY.AddCandidate(i, ext.rangeV[0], ext.rangeV[1]);
    if (side.fNsurf > 3) divisionXY.AddCandidate(i, ext.rangeU[0], ext.rangeU[1], ext.rangeV[0], ext.rangeV[1]);
  }

  auto bestEfficiency      = divisionX.Efficiency();
  SideDivisionCPU *bestDiv = &divisionX;
  auto efficiency          = divisionY.Efficiency();
  if (efficiency > bestEfficiency) {
    bestDiv        = &divisionY;
    bestEfficiency = efficiency;
  }
  efficiency = divisionXY.Efficiency();
  if (side.fNsurf >= 4 && efficiency > bestEfficiency) {
    bestDiv        = &divisionXY;
    bestEfficiency = efficiency;
  }
  if (all_rings_id) {
    efficiency = divisionR.Efficiency();
    if (efficiency > bestEfficiency) {
      bestDiv        = &divisionR;
      bestEfficiency = efficiency;
    }
  }
  if (bestEfficiency > 0.01) {
    // bestDiv->Print();
    side.fDivision = fCPUdata.fSideDivisions.size();
    fCPUdata.fSideDivisions.push_back(*bestDiv);
  }
  return side.fDivision;
}

template <typename Real_t>
void BrepHelper<Real_t>::ComputeSideDivisions()
{
  // Lambda for computing the division helper of a single side
  auto computeSingleSideDivision = [&](SurfaceType type, Side &side) {
    switch (type) {
    case SurfaceType::kPlanar: {
      auto extent_plane = ComputePlaneExtent(side);
      return ComputePlaneDivision(side, extent_plane);
    }
    case SurfaceType::kCylindrical:
    case SurfaceType::kConical: {
      auto extent_cyl = ComputeCylinderExtent(side);
      return ComputeCylinderDivision(side, extent_cyl);
    }
    default:
      return -1;
    }
  };

  // Compute division helpers for all sides having more than one frame on all surfaces
  int nfound{0}, ntotal{0};
  int nfoundall{0}, ntotalall{0};
  for (size_t common_id = 1; common_id < fCPUdata.fCommonSurfaces.size(); ++common_id) {
    auto &surf = fCPUdata.fCommonSurfaces[common_id];
    if (surf.fLeftSide.fNsurf > 1) {
      computeSingleSideDivision(surf.fType, surf.fLeftSide);
    }
    if (fCPUdata.fCommonSurfaces[common_id].fRightSide.fNsurf > 1) {
      computeSingleSideDivision(surf.fType, surf.fRightSide);
    }
    // In case the surface is NOT a scene surface, check for unique traversal frames
    if (surf.IsSceneSurface()) continue;
    ComputeTraversalFrames(common_id, true);
    ComputeTraversalFrames(common_id, false);
    CountTraversals(surf, nfoundall, ntotalall);
    if (surf.fLeftSide.fNsurf > 0 && surf.fRightSide.fNsurf > 0) {
      CountTraversals(surf, nfound, ntotal);
      //printf("surface %ld: %d/%d traversals\n", common_id, nfound, ntotal);
    }
  }
  printf("traversals_all: %d / %d [%f %%]\n", nfoundall, ntotalall, 100. * float(nfoundall) / ntotalall);
  printf("traversals    : %d / %d [%f %%]\n", nfound, ntotal, 100. * float(nfound) / ntotal);

  // Copy division helpers to surface data
  fSurfData->fNsideDivisions = fCPUdata.fSideDivisions.size();
  size_t size_slices         = 0;
  size_t size_slice_cand     = 0;
  for (auto const &div : fCPUdata.fSideDivisions) {
    size_slices += div.fNslices;
    size_slice_cand += div.GetNcandidates();
  }
  fSurfData->fNslices          = size_slices;
  fSurfData->fNsliceCandidates = size_slice_cand;
  fSurfData->fSideDivisions    = new SideDivision<Real_t>[fSurfData->fNsideDivisions];
  fSurfData->fSlices           = new SliceCand[size_slices];
  fSurfData->fSliceCandidates  = new int[size_slice_cand];

  SliceCand *slices     = fSurfData->fSlices;
  int *slice_candidates = fSurfData->fSliceCandidates;
  for (auto i = 0; i < fSurfData->fNsideDivisions; ++i) {
    fCPUdata.fSideDivisions[i].CopyTo(fSurfData->fSideDivisions[i], slices, slice_candidates);
    slices += fCPUdata.fSideDivisions[i].fSlices.size();
    slice_candidates += fCPUdata.fSideDivisions[i].GetNcandidates();
  }
}

template <typename Real_t>
bool BrepHelper<Real_t>::Convert()
{
  bool success = CreateLocalSurfaces();
  if (!success) return false;
  success = CreateCommonSurfacesScenes();
  return success;
}

template <typename Real_t>
void BrepHelper<Real_t>::ConvertTransformations(int idsurf)
{
  auto &surf = fCPUdata.fCommonSurfaces[idsurf];
  // Adopt the transformation of the first surface on left for the common surface
  surf.fTrans = fCPUdata.fFramedSurf[surf.fLeftSide.fSurfaces[0]].fTrans;
  TransformationMP<vecgeom::Precision> identity;

  // Set transformation of first surface on left to identity
  fCPUdata.fFramedSurf[surf.fLeftSide.fSurfaces[0]].fTrans = identity;

  // Set flip status of common surface based on first framed surface after sorting
  fCPUdata.fCommonSurfaces[idsurf].fFlipped = fCPUdata.fFramedSurf[surf.fLeftSide.fSurfaces[0]].fLogicId < 0 ? 1 : 0;

  TransformationMP<vecgeom::Precision> tsurfinv = surf.fTrans.Inverse();

  // Skip first surface on left side
  for (int i = 1; i < surf.fLeftSide.fNsurf; ++i) {
    int idglob       = surf.fLeftSide.fSurfaces[i];
    auto &framedsurf = fCPUdata.fFramedSurf[idglob];
    framedsurf.fTrans *= tsurfinv;
    framedsurf.fTrans.SetProperties();
  }

  // Convert right-side surfaces
  for (int i = 0; i < surf.fRightSide.fNsurf; ++i) {
    int idglob       = surf.fRightSide.fSurfaces[i];
    auto &framedsurf = fCPUdata.fFramedSurf[idglob];
    framedsurf.fTrans *= tsurfinv;
    framedsurf.fTrans.SetProperties();
  }
}

template <typename Real_t>
void BrepHelper<Real_t>::CreateCandidateLists()
{
  constexpr char kLside = 0x01;
  constexpr char kRside = 0x02;
  // Lambda adding the surface id as exiting candidate to all states from a side
  auto addSurfToSideStates = [&](int isurf, char iside) {
    auto const &surf = fCPUdata.fCommonSurfaces[isurf];
    Side const &side = (iside == kLside) ? surf.fLeftSide : surf.fRightSide;
    auto scene_id    = surf.GetSceneId();
    for (int i = 0; i < side.fNsurf; ++i) {
      int idglob             = side.fSurfaces[i];
      auto const &framedsurf = fCPUdata.fFramedSurf[idglob];
      vecgeom::NavigationState state(framedsurf.fState);
      auto state_id = state.GetId();
      auto surf_ind = framedsurf.fSurfIndex;

      auto &candidatesExiting = fCPUdata.GetCandidatesExiting(scene_id, state_id);
      auto &frameIndExiting   = fCPUdata.GetFrameIndExiting(scene_id, state_id);
      auto &sidesExiting      = fCPUdata.GetSidesExiting(scene_id, state_id);
      // We need to store the surface candidate and the frame index for the slot matching the local surface index
      candidatesExiting[surf_ind] = isurf;
      frameIndExiting[surf_ind]   = i;
      // Exiting frames may exist on both sides
      sidesExiting[surf_ind] |= iside;
      if (fVerbose > 0) {
        printf("  added to exiting of state on scene %d: ", scene_id);
        state.PrintTop();
        int j = 0;
        printf("candExiting:   ");
        for (auto candidate : candidatesExiting)
          printf("  %d: %d ", j++, candidate);
        printf("\n");
      }
    }
  };

  // Lambda adding the surface id as entering candidate to all parent states from a side
  auto addSurfToSideParents = [&](int isurf, char iside) {
    auto const &surf = fCPUdata.fCommonSurfaces[isurf];
    Side const &side = (iside == kLside) ? surf.fLeftSide : surf.fRightSide;
    for (int i = 0; i < side.fNsurf; ++i) {
      int idglob       = side.fSurfaces[i];
      auto &framedsurf = fCPUdata.fFramedSurf[idglob];
      // Skip parent frames, just make sure their parent state matches the default state
      NavIndex_t parent_state = 0;
      framedsurf.GetParentState(parent_state);
      if (framedsurf.fParent < 0) {
        assert(parent_state == surf.fDefaultState);
        continue;
      }

      // If parent is a boolean, it could be a virtual frame
      auto const &parent_framedsurf = fCPUdata.fFramedSurf[side.fSurfaces[framedsurf.fParent]];
      if (parent_framedsurf.fLogicId) framedsurf.fVirtualParent = true;

      // If the frame is embedded in the parent frame, skip the frame because it will always be entered through the
      // parent exception: if the parent is a virtual surface in a boolean, the embedded frame can only be entered
      // directly (not through the parent) and must be added to the entering candidates
      if (framedsurf.fEmbedded && !framedsurf.fVirtualParent) continue;
      // The frame is not embedded in the parent, so add the surface as candidate to the parent state
      // assert(parent_state != surf.fDefaultState);
      vecgeom::NavigationState state(parent_state);
      auto state_id            = state.GetId();
      auto &candidatesEntering = fCPUdata.GetCandidatesEntering(surf.GetSceneId(), state_id);
      auto &frameIndEntering   = fCPUdata.GetFrameIndEntering(surf.GetSceneId(), state_id);
      auto &sidesEntering      = fCPUdata.GetSidesEntering(surf.GetSceneId(), state_id);
      // the surface may already be a candidate for the parent state
      if (candidatesEntering.size() && std::abs(candidatesEntering.back()) == isurf) {
        candidatesEntering.back() = -isurf;
        sidesEntering.back() |= iside;
      } else {
        candidatesEntering.push_back(-isurf);
        frameIndEntering.push_back(i);
        sidesEntering.push_back(iside);
      }

      if (fVerbose > 0) {
        printf("  added %d to entering of non-embedding parent state on scene %d: ", candidatesEntering.back(),
               surf.GetSceneId());
        vecgeom::NavigationState::PrintTopImpl(parent_state);
      }
    }
  };

  // Lambda for adding the surface to the entering candidates of a state
  auto addSurfToDefaultEntering = [&](int isurf, NavIndex_t state) {
    auto const &surf = fCPUdata.fCommonSurfaces[std::abs(isurf)];
    if (fVerbose > 0) printf("===== CS %d:\n", isurf);
    //  Add surface as entering candidate for the provided state
    int state_id             = vecgeom::NavigationState::GetIdImpl(state);
    auto &candidatesEntering = fCPUdata.GetCandidatesEntering(surf.GetSceneId(), state_id);
    auto &frameIndEntering   = fCPUdata.GetFrameIndEntering(surf.GetSceneId(), state_id);
    auto &sidesEntering      = fCPUdata.GetSidesEntering(surf.GetSceneId(), state_id);
    // Check which sides contain surfaces of daughters
    char sides = 0;
    if (surf.fLeftSide.fNsurf) sides |= kLside;
    if (surf.fRightSide.fNsurf) sides |= kRside;
    candidatesEntering.push_back(isurf);
    frameIndEntering.push_back(-1); // means this state is the default for isurf
    sidesEntering.push_back(sides);
    if (fVerbose > 0) {
      printf("  added %d to entering of def state on scene %d: ", candidatesEntering.back(), surf.GetSceneId());
      vecgeom::NavigationState::PrintTopImpl(state);
    }
  };

  // loop over all common surfaces and add their index in the appropriate list
  for (int isurf = 1; isurf < int(fCPUdata.fCommonSurfaces.size()); ++isurf) {
    auto const &surf        = fCPUdata.fCommonSurfaces[isurf];
    auto const &topSurfLeft = fCPUdata.fFramedSurf[surf.fLeftSide.GetTopSurfaceIndex()];
    // The surface is an entering candidate for the default state, if the default state
    // is different than the top surface state
    if (topSurfLeft.fState != surf.fDefaultState) addSurfToDefaultEntering(isurf, surf.fDefaultState);
    // Check if the surface is an entering candidate for any of the parent states on sides
    addSurfToSideParents(isurf, kLside);
    addSurfToSideParents(isurf, kRside);

    // Add surface to side states as exiting candidate
    addSurfToSideStates(isurf, kLside);
    addSurfToSideStates(isurf, kRside);
  }
}

template <typename Real_t>
int BrepHelper<Real_t>::CreateCommonSurface(int idglob, int volId, int scene_id, int &iframe, char &iside)
{
  constexpr char kLside = 0x01;
  constexpr char kRside = 0x02;
  bool flip{false}, flip_bool{false};
  auto approxEqual = [&](int idglob1, int idglob2) {
    flip                                                            = false;
    flip_bool                                                       = false;
    FramedSurface<Precision, TransformationMP<Precision>> const &s1 = fCPUdata.fFramedSurf[idglob1];
    FramedSurface<Precision, TransformationMP<Precision>> const &s2 = fCPUdata.fFramedSurf[idglob2];
    // Surfaces may be in future "compatible" even if they are not the same, for now enforce equality
    if (s1.fSurface.type != s2.fSurface.type) return false;

    // Skip Arb4 for now
    static bool warning_printed = false;
    if (s1.fSurface.type == SurfaceType::kArb4 || s2.fSurface.type == SurfaceType::kArb4) {
      if (!warning_printed) {
        VECGEOM_LOG(warning) << "CreateCommonSurface: case " << to_cstring(s1.fSurface.type) << " not implemented";
        warning_printed = true;
      }
      return false;
    }

    // Check if the surfaces may be flipped because of Boolean negation
    if ((s1.fLogicId == 0) || (s2.fLogicId == 0)) {
      flip_bool = (s1.fLogicId < 0) || (s2.fLogicId < 0);
    } else {
      flip_bool = (s1.fLogicId < 0) != (s2.fLogicId < 0);
    }

    // Check if the 2 surfaces are parallel
    vecgeom::Transformation3DMP<vecgeom::Precision> const &t1 = s1.fTrans;
    vecgeom::Transformation3DMP<vecgeom::Precision> const &t2 = s2.fTrans;
    // Check if the rotations are matching. The z axis inverse-transformed
    // with the two rotations should end up as aligned vectors. This is
    // true for planes (Z is the normal) but also for tubes/cones where
    // Z is the axis of symmetry
    vecgeom::Vector3D<vecgeom::Precision> const zaxis(0, 0, 1);
    auto z1 = t1.InverseTransformDirection(zaxis);
    auto z2 = t2.InverseTransformDirection(zaxis);
    if (!ApproxEqualVector(z1.Cross(z2), {0, 0, 0})) return false;
    // Calculate normalized connection vector between the two transformations
    // Use double precision explicitly
    vecgeom::Vector3D<vecgeom::Precision> tdiff = t1.Translation() - t2.Translation();
    bool same_tr                                = ApproxEqualVector(tdiff, {0, 0, 0});
    vecgeom::Vector3D<vecgeom::Precision> ldir;
    switch (s1.fSurface.type) {
    case SurfaceType::kPlanar:
      flip = z1.Dot(z2) < 0;
      if (same_tr) break;
      // For planes to match, the connecting vector must be along the planes
      tdiff.Normalize();
      t1.TransformDirection(tdiff, ldir);
      if (std::abs(ldir[2]) > vecgeom::kTolerance) return false;
      break;
    case SurfaceType::kCylindrical:
      if (std::abs(s1.fSurface.Radius() - s2.fSurface.Radius()) > vecgeom::kTolerance) return false;
      // Check if the cylynders are flipped with respect to each other
      flip = (s1.fSurface.IsFlipped()) ^ (s2.fSurface.IsFlipped());
      if (same_tr) break;
      tdiff.Normalize();
      t1.TransformDirection(tdiff, ldir);
      // For connected cylinders, the connecting vector must be along the Z axis
      if (!ApproxEqualVector(ldir, {0, 0, ldir[2]})) return false;
      break;
    case SurfaceType::kConical:
      if (std::abs(s1.fSurface.RadiusZ(-Abs(t1.Translation()[2])) - s2.fSurface.RadiusZ(-Abs(t2.Translation()[2]))) >
          vecgeom::kTolerance) {
        return false;
      }
      if (std::abs(s1.fSurface.fSlope - s2.fSurface.fSlope) > vecgeom::kTolerance) {
        return false;
      }
      flip = s1.fSurface.IsFlipped() ^ s2.fSurface.IsFlipped();
      if (same_tr) break;
      tdiff.Normalize();
      t1.TransformDirection(tdiff, ldir);
      // For connected cones, the connecting vector must be along the Z axis
      if (!ApproxEqualVector(ldir, {0, 0, ldir[2]})) return false;
      break;
    case SurfaceType::kSpherical:
    case SurfaceType::kTorus:
    case SurfaceType::kArb4:
    default:
      if (!warning_printed) {
        VECGEOM_LOG(warning) << "CreateCommonSurface: case " << to_cstring(s1.fSurface.type) << " not implemented";
        warning_printed = true;
      }
      return false;
    };
    return true;
  };

  auto surfHash = [&](int idglobal, double tolerance = 100 * vecgeom::kTolerance) {
    // Compute hash for the surface rotation and translation
    auto const &surf                                             = fCPUdata.fFramedSurf[idglobal];
    vecgeom::Transformation3DMP<vecgeom::Precision> const &trans = surf.fTrans;

    // get normal vector of surface
    vecgeom::Vector3D<vecgeom::Precision> normal;
    vecgeom::Vector3D<vecgeom::Precision> scaled_norm_vector;
    const vecgeom::Vector3D<vecgeom::Precision> lnorm(0, 0, 1);
    trans.InverseTransformDirection(lnorm, normal);

    long hash = 0;
    // helper function to generate hash from integer numbers
    auto hash_combine = [](long seed, const long value) {
      return seed ^ (std::hash<long>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    };

    switch (surf.fSurface.type) {
    case SurfaceType::kPlanar:
      // use normal vector scaled by the distance to the origin for hashing
      scaled_norm_vector = trans.Translation().Dot(normal) * normal;
      for (int i = 0; i < 3; i++) {
        // use tolerance to generate int with the desired precision from a real number for hashing
        hash = hash_combine(hash, static_cast<long>(std::round(scaled_norm_vector[i] / tolerance)));
      }
      break;
    case SurfaceType::kCylindrical:
      // use radius and normal for hashing
      hash = hash_combine(hash, static_cast<long>(std::round(surf.fSurface.Radius() / tolerance)));

      for (int i = 0; i < 3; i++) {
        hash = hash_combine(hash, static_cast<long>(std::round(normal[i] / tolerance)));
      }
      break;
    case SurfaceType::kConical:
      // use radius at origin, slope, and normal for hashing
      hash = hash_combine(
          hash, static_cast<long>(std::round(surf.fSurface.RadiusZ(-Abs(trans.Translation()[2])) / tolerance)));
      hash = hash_combine(hash, static_cast<long>(std::round(surf.fSurface.fSlope / tolerance)));
      for (int i = 0; i < 3; i++) {
        hash = hash_combine(hash, static_cast<long>(std::round(normal[i] / tolerance)));
      }
      break;
    case SurfaceType::kSpherical:
    case SurfaceType::kTorus:
    case SurfaceType::kArb4:
    default:
      hash = 0;
      break;
    };

    return hash;
  };

  auto const &surf   = fCPUdata.fFramedSurf[idglob];
  bool is_scene_surf = (scene_id > 0) && (surf.fState == 0);
  auto hash          = surfHash(idglob, 1000 * vecgeom::kTolerance);
  // Get the compatible surfaces
  auto range          = fCPUdata.fSurfHash[scene_id].equal_range(hash);
  bool found_dup_surf = false;
  int id              = -1;
  // check duplicates only for valid hashes
  if (hash != 0) {
    for (auto it = range.first; it != range.second; ++it) {
      const auto &other_id = fCPUdata.fCommonSurfaces[it->second].fLeftSide.fSurfaces[0];

      if (approxEqual(other_id, idglob)) {
        flip ^= flip_bool;
        // Do not allow surfaces of the same volume on different sides of the same common surface, otherwise the
        // surface will be missed when coming from the entering side. if (flip && othersurf.VolumeId() == volId)
        // continue;
        found_dup_surf = true;
        id             = it->second;
        auto &crt_side = flip ? fCPUdata.fCommonSurfaces[id].fRightSide : fCPUdata.fCommonSurfaces[id].fLeftSide;
        // The common surface is compatible only if the parent state for the current framed surface
        // has a frame on the same side or it is already the default state.
        NavIndex_t parent_state_index = fCPUdata.fFramedSurf[idglob].fState;
        vecgeom::NavigationState::PopImpl(parent_state_index);
        NavIndex_t default_state = fCPUdata.fCommonSurfaces[id].fDefaultState;
        // if the default state is different (note that it can also be different if both are 0 but it is a scene) do
        // thorough check. If the common surface is a top scene surface (fSceneId < 0), don't do the state check if
        // the to-be-checked surface is also a top scene surface (is_scene_surf), because those need to be put on the
        // same common surface
        if ((default_state != parent_state_index) || (default_state == 0 && parent_state_index == 0 &&
                                                      fCPUdata.fCommonSurfaces[id].fSceneId < 0 && !is_scene_surf)) {
          // To be compatible, a surface of the parent state MUST exist on the same side
          bool has_parent = false;
          for (auto isurf = 0; isurf < crt_side.fNsurf; ++isurf) {
            has_parent = fCPUdata.fFramedSurf[crt_side.fSurfaces[isurf]].fState == parent_state_index;
            if (has_parent) break;
          }
          if (!has_parent) {
            found_dup_surf = false;
            continue;
          }
        }
        // Add the global surface to the appropriate side
        iframe = crt_side.AddSurface(idglob);
        iside  = flip ? kRside : kLside;
        break;
      }
    }
  }

  if (!found_dup_surf) {
    // Construct a new common surface from the current placed global surface
    // Set the common state to be the parent of the idglob surface state
    id = fCPUdata.fCommonSurfaces.size();
    fCPUdata.fCommonSurfaces.push_back({fCPUdata.fFramedSurf[idglob].fSurface.type, idglob, surf.fLogicId < 0});
    iside             = kLside;
    iframe            = 0;
    auto parent_state = fCPUdata.fFramedSurf[idglob].fState;
    vecgeom::NavigationState::PopImpl(parent_state);
    fCPUdata.fCommonSurfaces[id].fSceneId      = is_scene_surf ? -scene_id : scene_id;
    fCPUdata.fCommonSurfaces[id].fDefaultState = parent_state;
    // insert the common surface in the scene and in the scene multimap
    fCPUdata.fSurfHash[scene_id].insert(std::make_pair(hash, id));
  }

  return id;
}

template <typename Real_t>
bool BrepHelper<Real_t>::CreateCommonSurfacesScenes()
{
  using Real_b          = typename SurfData<Real_t>::Real_b;
  constexpr char kLside = 0x01;
  constexpr char kRside = 0x02;
  int nphysical         = 0;
  int maxscene          = -1;
  int numscenes         = 0;
  vecgeom::NavigationState state;
  int ntot = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  std::vector<bool> visited(ntot, false);
  auto &numStates = fCPUdata.fSceneTouchables;

  // recursive geometry visitor lambda counting the number of states per scene
  typedef std::function<void(vecgeom::VPlacedVolume const *)> funcCountStates_t;
  funcCountStates_t countStatesPerScene = [&](vecgeom::VPlacedVolume const *pvol) {
    state.Push(pvol);
    const auto vol          = pvol->GetLogicalVolume();
    auto ivol               = vol->id();
    auto daughters          = vol->GetDaughters();
    int nd                  = daughters.size();
    unsigned short scene_id = 0, newscene_id = 0;
    bool is_scene = state.GetSceneId(scene_id, newscene_id);
    if (scene_id > maxscene) {
      int ntoinsert = scene_id - maxscene + 1;
      numStates.insert(numStates.end(), ntoinsert, 0);
      maxscene  = scene_id;
      numscenes = maxscene + 1;
    }
    numStates[scene_id]++;
    bool do_daughters = is_scene ? (!visited[ivol]) : true;
    if (do_daughters) {
      for (int id = 0; id < nd; ++id) {
        countStatesPerScene(daughters[id]);
      }
    }
    visited[ivol] = is_scene;
    state.Pop();
  };

  auto allocateExitingCandidates = [&](unsigned short scene_id, int state_id, size_t nsurf) {
    auto &candidatesExiting = fCPUdata.GetCandidatesExiting(scene_id, state_id);
    auto &frameIndExiting   = fCPUdata.GetFrameIndExiting(scene_id, state_id);
    auto &sidesExiting      = fCPUdata.GetSidesExiting(scene_id, state_id);
    if (nsurf >= candidatesExiting.size()) {
      candidatesExiting.insert(candidatesExiting.end(), nsurf - candidatesExiting.size(), 0);
      frameIndExiting.insert(frameIndExiting.end(), nsurf - frameIndExiting.size(), 0);
      sidesExiting.insert(sidesExiting.end(), nsurf - sidesExiting.size(), 0);
    }
  };

  // recursive geometry visitor lambda creating the common surfaces for the current placed volume
  typedef std::function<void(vecgeom::VPlacedVolume const *)> func_t;
  func_t createCommonSurfaces = [&](vecgeom::VPlacedVolume const *pvol) {
    // TODO: In this funcstion, store the product of pvol.trans * surface.trans
    state.Push(pvol);
    const auto vol          = pvol->GetLogicalVolume();
    auto ivol               = vol->id();
    auto daughters          = vol->GetDaughters();
    int nd                  = daughters.size();
    NavIndex_t nav_ind      = state.GetNavIndex();
    unsigned short scene_id = 0, newscene_id = 0;
    bool is_scene   = state.GetSceneId(scene_id, newscene_id);
    auto state_id   = state.GetId();
    auto &nperscene = fCPUdata.fSceneTouchables;
    nphysical++;
    nperscene[scene_id]++;
    TransformationMP<vecgeom::Precision> trans;
    // only consider the surface transformation in its scene
    state.TopInSceneMatrix(trans);
    VolumeShellCPU const &shell = fCPUdata.fShells[ivol];
    // Allocate exit candidates arrays
    auto nsurf_local = shell.fSurfaces.size();
    allocateExitingCandidates(scene_id, state_id, nsurf_local);
    if (is_scene && !visited[ivol]) allocateExitingCandidates(newscene_id, 0, nsurf_local);

    for (int lsurf_id : shell.fSurfaces) {
      FramedSurface<Precision, TransformationMP<Precision>> &lsurf = fCPUdata.fLocalSurfaces[lsurf_id];

      // Ignore 'inside' helper surfaces having no frame
      if (lsurf.fFrame.type == FrameType::kNoFrame) continue;
      TransformationMP<vecgeom::Precision> surftrans = lsurf.fTrans;
      surftrans *= trans;

      // Create the surface in the current scene using the local navigation index in the scene
      int id_surf = fCPUdata.fFramedSurf.size();
      fCPUdata.fFramedSurf.push_back({lsurf.fSurface, lsurf.fFrame, surftrans, nav_ind, lsurf.fNeverCheck});
      auto &framed_surf      = fCPUdata.fFramedSurf[id_surf];
      framed_surf.fLogicId   = lsurf.fLogicId;
      framed_surf.fSurfIndex = lsurf.fSurfIndex;
      framed_surf.fEmbedding = (lsurf.fLogicId == 0) ? lsurf.fEmbedding : false;
      assert(lsurf.fSurfIndex < nsurf_local);

      char iside = 0;
      int iframe = 0;
      auto isurf = CreateCommonSurface(id_surf, ivol, scene_id, iframe, iside);

      if (fVerbose > 0) {
        std::cout << "scene " << scene_id << ": framed surface " << id_surf << " on CS " << isurf << std::endl;
        state.PrintTop();
        std::cout << "  " << surftrans << "\n";
      }

      if (is_scene) {
        // Create the surface once also in the new scene if the shell belongs to one
        if (!visited[ivol]) {
          // Make a new top framed surface in the new scene
          int id_surf_scene = fCPUdata.fFramedSurf.size();
          // Store the local transformation of the scene volume surface
          fCPUdata.fFramedSurf.push_back(
              {lsurf.fSurface, lsurf.fFrame, lsurf.fTrans, 0 /*top in scene*/, lsurf.fNeverCheck});
          // Watchout: evil bug: cannot use the framed_surf reference after this point, because after
          // inserting a new frame the array may be re-allocated internally by the vector
          fCPUdata.fFramedSurf[id_surf_scene].fLogicId   = lsurf.fLogicId;
          fCPUdata.fFramedSurf[id_surf_scene].fSurfIndex = lsurf.fSurfIndex;
          fCPUdata.fFramedSurf[id_surf_scene].fEmbedding = (lsurf.fLogicId == 0) ? lsurf.fEmbedding : false;
          auto isurf_scene = CreateCommonSurface(id_surf_scene, ivol, newscene_id, iframe, iside);

          // This assert was to ensure that the first surface of a new volume must be on the left side
          // For booleans, this is not strictly true, so we remove this as a consequence of !1075
          // constexpr char kLside                  = 0x01;
          // assert(iside == kLside);

          // Add the CS pointer to the frame in the parent scene. So if a track enters the frame it is relocated in
          // this frame, it checks the info on the scene CS
          fCPUdata.fFramedSurf[id_surf].fSceneCS = isurf_scene;
          // Frames on sides are sorted after, so store the global surface index for now
          // After sorting we change this index the the index of the frame on the side
          fCPUdata.fFramedSurf[id_surf].fSceneCSind = (iside == kLside) ? id_surf_scene : -id_surf_scene;
          fCPUdata.fSceneShells[ivol].fSurfaces[framed_surf.fSurfIndex] = id_surf;
          if (fVerbose > 0) {
            VECGEOM_LOG(info) << "scene " << newscene_id << ": top framed surface " << id_surf_scene << " on CS "
                              << isurf_scene;
            VECGEOM_LOG(info) << "  linked to CS " << isurf << " surf_index=" << lsurf.fSurfIndex;
          }
        } else {
          // We need to assign the scene CS pointer to the framed surface
          int id_surf_first       = fCPUdata.fSceneShells[ivol].fSurfaces[framed_surf.fSurfIndex];
          auto &framed_surf_first = fCPUdata.fFramedSurf[id_surf_first];
          framed_surf.fSceneCS    = framed_surf_first.fSceneCS;
          framed_surf.fSceneCSind = framed_surf_first.fSceneCSind;
        }
      }
    }

    bool do_daughters = is_scene ? (!visited[ivol]) : true;
    if (do_daughters) {
      for (int id = 0; id < nd; ++id) {
        createCommonSurfaces(daughters[id]);
      }
    }
    visited[ivol] = is_scene;
    state.Pop();
  };

  auto checkExitingCandidates = [&](unsigned short scene_id, int state_id, size_t nsurf) {
    auto &candidatesExiting = fCPUdata.GetCandidatesExiting(scene_id, state_id);
    int ivol                = state.GetLogicalId();
    auto &shell             = fCPUdata.fShells[ivol];
    for (size_t isurf = 0; isurf < nsurf; ++isurf) {
      auto surf = fCPUdata.fLocalSurfaces[shell.fSurfaces[isurf]];
      if ((surf.fFrame.type == FrameType::kNoFrame && candidatesExiting[isurf] > 0) ||
          (surf.fFrame.type != FrameType::kNoFrame && candidatesExiting[isurf] == 0)) {
        std::cout << "=== Wrong candidates for state:\n";
        state.Print();
        printf("   candExiting:   ");
        int j = 0;
        for (auto candidate : candidatesExiting)
          printf("  %d: %d ", j++, candidate);
        printf("\n");
        return false;
      }
    }
    return true;
  };

  // recursive geometry visitor lambda validating exiting candidates
  typedef std::function<void(vecgeom::VPlacedVolume const *)> funcValidate_t;
  funcValidate_t validateExitingCandidates = [&](vecgeom::VPlacedVolume const *pvol) {
    state.Push(pvol);
    const auto vol          = pvol->GetLogicalVolume();
    auto ivol               = vol->id();
    auto daughters          = vol->GetDaughters();
    int nd                  = daughters.size();
    unsigned short scene_id = 0, newscene_id = 0;
    bool is_scene               = state.GetSceneId(scene_id, newscene_id);
    auto state_id               = state.GetId();
    VolumeShellCPU const &shell = fCPUdata.fShells[ivol];
    auto nsurf_local            = shell.fSurfaces.size();

    if (!checkExitingCandidates(scene_id, state_id, nsurf_local)) return false;
    if (is_scene && !visited[ivol]) {
      if (!checkExitingCandidates(newscene_id, 0, nsurf_local)) return false;
    }

    bool do_daughters = is_scene ? (!visited[ivol]) : true;
    if (do_daughters) {
      for (int id = 0; id < nd; ++id) {
        validateExitingCandidates(daughters[id]);
      }
    }
    visited[ivol] = is_scene;
    state.Pop();
    return true;
  };

  // add a dummy common surface since index 0 is not allowed for correctly handling sides
  fCPUdata.fCommonSurfaces.push_back({});

  auto world = vecgeom::GeoManager::Instance().GetWorld();
  // Count touchables per scene since this is not stored in the navigation table
  countStatesPerScene(world);
  // Pre-allocate data in scene arrays
  // fCPUdata.fSceneTouchables already filled
  fCPUdata.fSceneStartIndex.insert(fCPUdata.fSceneStartIndex.end(), numscenes, 0);
  fCPUdata.fSurfHash.insert(fCPUdata.fSurfHash.end(), numscenes, {});
  int startindex = 0;
  for (int scene_id = 0; scene_id < numscenes; ++scene_id) {
    // Reserve a place also for the outside state in each scene
    int nt                              = ++fCPUdata.fSceneTouchables[scene_id];
    fCPUdata.fSceneStartIndex[scene_id] = startindex;
    startindex += nt;
    fCPUdata.fCandidatesEntering.insert(fCPUdata.fCandidatesEntering.end(), nt, {});
    fCPUdata.fCandidatesExiting.insert(fCPUdata.fCandidatesExiting.end(), nt, {});
    fCPUdata.fFrameIndEntering.insert(fCPUdata.fFrameIndEntering.end(), nt, {});
    fCPUdata.fFrameIndExiting.insert(fCPUdata.fFrameIndExiting.end(), nt, {});
    fCPUdata.fSidesEntering.insert(fCPUdata.fSidesEntering.end(), nt, {});
    fCPUdata.fSidesExiting.insert(fCPUdata.fSidesExiting.end(), nt, {});
  }

  // before we construct the common surfaces from the local surfaces, we can mark convex boolean surfaces such that
  // they are treated as non-boolean surfaces. this requires the bounding boxes of the BVH. Init the information
  // needed to contruct and use the BVH
  InitBVHData();

  vecgeom::ABBoxManager<Real_b>::Instance().InitABBoxesForSurfaces(fCPUdata);
  // Commented out Boolean convexity checking
  // FindConvexBooleanSurfaces();
  vecgeom::ABBoxManager<Real_b>::Instance().InitABBoxesForSurfaces(fCPUdata, /*crop=*/true);

  std::fill(visited.begin(), visited.end(), false);
  createCommonSurfaces(world);

  for (size_t isurf = 1; isurf < fCPUdata.fCommonSurfaces.size(); ++isurf) {
    // Compute the default states in case no frame on the surface is hit
    ComputeDefaultStates(isurf);
    // Sort placed surfaces on sides by geometry depth (bigger depth comes first)
    SortSides(isurf);
    // Convert transformations of placed surfaces in the local frame of the common surface
    ConvertTransformations(isurf);
  }

  // Adjust scene frame index pointers
  // Lambda adjusting fSceneCSind
  auto adjustFrameIndex = [&](int isurf, char iside, int iglob) {
    auto const &surf = fCPUdata.fCommonSurfaces[isurf];
    Side const &side = (iside == kLside) ? surf.fLeftSide : surf.fRightSide;
    for (int i = 0; i < side.fNsurf; ++i) {
      if (side.fSurfaces[i] == std::abs(iglob)) return (iglob > 0) ? i + 1 : -(i + 1);
    }
    return 0;
  };

  for (size_t iframe = 0; iframe < fCPUdata.fFramedSurf.size(); ++iframe) {
    auto &framed_surf = fCPUdata.fFramedSurf[iframe];
    if (framed_surf.fSceneCS > 0) {
      auto iglob = framed_surf.fSceneCSind;
      auto index = adjustFrameIndex(framed_surf.fSceneCS, kLside, iglob);
      if (index == 0) index = adjustFrameIndex(framed_surf.fSceneCS, kRside, iglob);
      if (index != 0) framed_surf.fSceneCSind = index;
    }
  }

  // Create the full surface candidate list for each navigation state
  CreateCandidateLists();

  // Validate exiting candidates
  std::fill(visited.begin(), visited.end(), false);
  state.Clear();
  validateExitingCandidates(world);

  // Compute side divisions for all sides of common surfaces
  ComputeSideDivisions();

  // Now update the surface data structure used for navigation
  UpdateSurfData();

  ////////////////////////// BVH /////////////////////////////

  // Surf bounding boxes are already initialized as they were needed for convexity check of boolean surfaces

  // At this point, the AABBs for solids, templated by vecgeom::Precision have already been
  // initialized. In case vecgeom::Precision and Real_t are different, we need to also initialize
  // them for ABBoxManager<Real_t>
  vecgeom::ABBoxManager<Real_b>::Instance().InitABBoxesForCompleteGeometry();

  // Then initialize the BVH for each Logical Volume
  std::vector<vecgeom::LogicalVolume const *> lvols;
  vecgeom::GeoManager::Instance().GetAllLogicalVolumes(lvols);

  // Allocate space for the BVHs
  using Real_b    = typename SurfData<Real_t>::Real_b;
  fSurfData->fBVH = new bvh::BVHsurf<Real_b>[fCPUdata.fShells.size()];
  fSurfData->fBVHSolids =
      new bvh::BVHsurf<Real_b>[fCPUdata.fShells.size()]; // Auxiliary BVH for location, built from solid AABBs

  for (auto logical_volume : lvols) {
    // Get the index allocated for this shell's BVH
    int bvh_index = fCPUdata.fShells[logical_volume->id()].fBVH;
    int nSolidBoxes{0};
    int nSurfBoxes{0};
    Vector3D<Real_b> *surfBoxes{nullptr};
    Vector3D<Real_b> *solidBoxes{nullptr};
    surfBoxes = vecgeom::ABBoxManager<Real_b>::Instance().GetSurfaceABBoxes(logical_volume->id(), nSurfBoxes, fCPUdata);
    solidBoxes = vecgeom::ABBoxManager<Real_b>::Instance().GetABBoxes(logical_volume, nSolidBoxes);

    // Note: we construct the BVH directly in the surface data and do not copy it from the CPU data. Still, we use the
    // CPUdata to get the indices and data needed for initialization. Destroy the empty BVH object
    fSurfData->fBVH[bvh_index].Clear();
    // Construct the correct BVH from the logical volume
    // bvh::InitBVH<Real_t>(logical_volume->id(), fSurfData->fBVH[bvh_index], *fSurfData);
    bvh::InitBVH<Real_t>(logical_volume->id(), fSurfData->fBVH[bvh_index], fCPUdata, surfBoxes, nSurfBoxes);
    // Construct the solids BVH
    if (logical_volume->GetDaughters().size() > 0) {
      // For solids, we can only construct a BVH if there are daughter volumes
      // Otherwise the instance remains uninitialized
      bvh::InitBVH<Real_t>(logical_volume->id(), fSurfData->fBVHSolids[bvh_index], fCPUdata, solidBoxes, nSolidBoxes);
    }
  }

  ////////////////////////////////////////////////////////////
  if (fVerbose > 0) {
    for (size_t isurf = 1; isurf < fCPUdata.fCommonSurfaces.size(); ++isurf)
      PrintCommonSurface(isurf);
  }

  if (fVerbose > 1) {
    PrintCandidateLists();
    std::cout << "Visited " << nphysical << " physical volumes, created " << fCPUdata.fCommonSurfaces.size() - 1
              << " common surfaces\n";
  }

  return true;
}

template <typename Real_t>
bool BrepHelper<Real_t>::CreateLocalSurfaces()
{
  // add identity first in the list of local transformations
  TransformationMP<vecgeom::Precision> identity;
  //  Iterate logical volumes and create local surfaces
  std::vector<vecgeom::LogicalVolume *> volumes;
  auto n_registered_volumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  vecgeom::GeoManager::Instance().GetAllLogicalVolumes(volumes);

  SetNvolumes(n_registered_volumes);
  // TODO: Implement a VUnplacedVolume::CreateSurfaces interface for surface creation
  // create a placeholder for surface data
  for (auto volume : volumes) {
    vecgeom::VUnplacedVolume const *solid = volume->GetUnplacedVolume();
    bool result                           = conv::CreateSolidSurfaces<vecgeom::Precision>(solid, volume->id());
    if (!result) {
      VECGEOM_LOG(critical) << "Could not convert volume " << volume->id() << ": " << volume->GetName();
      return false;
    }
    // Finalize logic expression
    if (!fCPUdata.fShells[volume->id()].fSimplified) {
      auto &crtlogic = fCPUdata.fShells[volume->id()].fLogic;
      vgbrep::logichelper::LogicExpressionConstruct lc(crtlogic);
      lc.Simplify(crtlogic);
      vgbrep::logichelper::insert_jumps(crtlogic);
      fCPUdata.fShells[volume->id()].fSimplified = true;
    }
  }

  if (fVerbose > 0) {
    for (auto volume : volumes) {
      auto const &shell = fCPUdata.fShells[volume->id()];
      printf("shell %d for volume %s:\n", volume->id(), volume->GetName());
      logichelper::print_logic(shell.fLogic);
      for (int lsurf_id : shell.fSurfaces) {
        auto const &lsurf = fCPUdata.fLocalSurfaces[lsurf_id];
        printf(" local surf %d (logic_id=%d): ", lsurf_id, lsurf.fLogicId);
        lsurf.fTrans.Print();
        printf("\n");
      }
    }
  }

  return true;
}

template <typename Real_t>
void BrepHelper<Real_t>::DumpBVH(uint ivol)
{
  auto lvol = vecgeom::GeoManager::Instance().GetLogicalVolume(ivol);
  assert(lvol->id() == ivol);
  int bvh_index = fSurfData->fShells[ivol].fBVH;
  auto fname    = std::string(lvol->GetName()) + ".bin";
  bvh::DumpBVH(fSurfData->fBVH[bvh_index], fname.c_str());
}

template <typename Real_t>
bool BrepHelper<Real_t>::EqualFrames(Side const &side, int i1, int i2)
{
  using Vector3D = vecgeom::Vector3D<vecgeom::Precision>;

  auto const &s1 = fCPUdata.fFramedSurf[side.fSurfaces[i1]];
  auto const &s2 = fCPUdata.fFramedSurf[side.fSurfaces[i2]];
  if (s1.fFrame.type != s2.fFrame.type) return false;
  // Get displacement vector between the 2 frame centers and check if it has null length
  vecgeom::Transformation3DMP<vecgeom::Precision> const &t1 = s1.fTrans;
  vecgeom::Transformation3DMP<vecgeom::Precision> const &t2 = s2.fTrans;
  Vector3D tdiff                                            = t1.Translation() - t2.Translation();
  // TODO: Check if this has to always hold with the new mask types!!
  if (!ApproxEqualVector(tdiff, {0, 0, 0})) return false;

  // Different treatment of different frame types
  switch (s1.fFrame.type) {
  case FrameType::kRangeZ:
    break;
  case FrameType::kRing: {
    auto mask1 = fCPUdata.fRingMasks[s1.fFrame.id];
    auto mask2 = fCPUdata.fRingMasks[s2.fFrame.id];

    // Inner radius must be the same
    if (!ApproxEqual(mask1.rangeR[0], mask2.rangeR[0]) || !ApproxEqual(mask1.rangeR[1], mask2.rangeR[1]) ||
        mask1.isFullCirc != mask2.isFullCirc)
      return false;
    if (mask1.isFullCirc) return true;
    // Unit-vectors used for checking sphi
    auto phimin1 = t1.InverseTransformDirection(Vector3D{mask1.vecSPhi[0], mask1.vecSPhi[1], 0});
    auto phimin2 = t2.InverseTransformDirection(Vector3D{mask2.vecSPhi[0], mask2.vecSPhi[1], 0});
    // Rmax vectors used for checking dphi and outer radius
    auto phimax1 = t1.InverseTransformDirection(Vector3D{mask1.vecEPhi[0], mask1.vecEPhi[1], 0});
    auto phimax2 = t2.InverseTransformDirection(Vector3D{mask2.vecEPhi[0], mask2.vecEPhi[1], 0});

    if (ApproxEqualVector(phimin1, phimin2) && ApproxEqualVector(phimax1, phimax2)) return true;
    break;
  }
  case FrameType::kZPhi: {
    auto mask1 = fCPUdata.fZPhiMasks[s1.fFrame.id];
    auto mask2 = fCPUdata.fZPhiMasks[s2.fFrame.id];

    // They are on the same side, so there is no flipping,
    // and z extents must be equal
    if (!ApproxEqual(mask1.rangeZ[0], mask2.rangeZ[0]) || !ApproxEqual(mask1.rangeZ[1], mask2.rangeZ[1])) return false;
    if (mask1.isFullCirc) return true;
    // Unit-vectors used for checking sphi
    auto phimin1 = t1.InverseTransformDirection(Vector3D{mask1.vecSPhi[0], mask1.vecSPhi[1], 0});
    auto phimin2 = t2.InverseTransformDirection(Vector3D{mask2.vecSPhi[0], mask2.vecSPhi[1], 0});
    // Rmax vectors used for checking dphi and outer radius
    auto phimax1 = t1.InverseTransformDirection(Vector3D{mask1.vecEPhi[0], mask1.vecEPhi[1], 0});
    auto phimax2 = t2.InverseTransformDirection(Vector3D{mask2.vecEPhi[0], mask2.vecEPhi[1], 0});

    if (ApproxEqualVector(phimin1, phimin2) && ApproxEqualVector(phimax1, phimax2)) return true;
    break;
  }
  case FrameType::kRangeSph:
    // if (ApproxEqualVector(v1, v2)) return true; //Must be changed.
    break;
  case FrameType::kWindow: {
    auto frameData1 = fCPUdata.fWindowMasks[s1.fFrame.id];
    auto frameData2 = fCPUdata.fWindowMasks[s2.fFrame.id];

    // Vertices
    Vector3D v11 = t1.InverseTransformDirection(Vector3D{frameData1.rangeU[0], frameData1.rangeV[0], 0}); // 1 down left
    Vector3D v12 = t1.InverseTransformDirection(Vector3D{frameData1.rangeU[1], frameData1.rangeV[1], 0}); // 1 up right
    Vector3D v21 = t2.InverseTransformDirection(Vector3D{frameData2.rangeU[0], frameData2.rangeV[0], 0}); // 2 down left
    Vector3D v22 = t2.InverseTransformDirection(Vector3D{frameData2.rangeU[1], frameData2.rangeV[1], 0}); // 2 up right

    // Diagonals
    auto diag1 = v12 - v11;
    auto diag2 = v22 - v21;

    return ApproxEqualVector(diag1.Abs(), diag2.Abs());
  }
  case FrameType::kTriangle:
    // to be implemented
    break;
  case FrameType::kQuadrilateral: {
    // to be implemented
    break;
  }
  default:
    break;
  };
  return false;
}

template <typename Real_t>
void BrepHelper<Real_t>::InitBVHData()
{
  std::vector<vecgeom::LogicalVolume const *> lvols;
  vecgeom::GeoManager::Instance().GetAllLogicalVolumes(lvols);

  TransformationMP<vecgeom::Precision> identity;
  assert(fCPUdata.fPVolTrans.size() == 0);
  fCPUdata.fPVolTrans.push_back(identity);

  for (auto lvol : lvols) {
    // Get the shell of the LV
    auto &rootShell = fCPUdata.fShells[lvol->id()];

    // Assign a BVH slot
    rootShell.fBVH = lvol->id();

    // Initialize array of exiting surfaces which have a frame
    for (uint i = 0; i < rootShell.fSurfaces.size(); i++) {
      if (fCPUdata.fLocalSurfaces[rootShell.fSurfaces[i]].fFrame.type != FrameType::kNoFrame)
        rootShell.fExitingSurfaces.push_back(i);
    }

    for (const auto &pvol : lvol->GetDaughters()) {
      // Add the PVol ID to the root shell
      rootShell.fDaughterPvolIds.push_back(pvol->id());

      // Get the shell
      const auto &shell = fCPUdata.fShells[pvol->GetLogicalVolume()->id()];

      // get transformation to placed volume
      int vol_trans_id = fCPUdata.fPVolTrans.size();
      fCPUdata.fPVolTrans.push_back(vecgeom::Transformation3DMP<vecgeom::Precision>(*pvol->GetTransformation()));

      // Add the PVol transformation to the root shell
      rootShell.fDaughterPvolTrans.push_back(vol_trans_id);

      // Iterate over the local surfaces which have a frame in this shell
      for (uint i = 0; i < shell.fSurfaces.size(); i++) {
        if (fCPUdata.fLocalSurfaces[shell.fSurfaces[i]].fFrame.type != FrameType::kNoFrame) {
          // Add the index of the surface to the list of entering surfaces for the LV
          rootShell.fEnteringSurfaces.push_back(shell.fSurfaces[i]);
          // Add the id of the Pvol to which this surface belongs
          rootShell.fEnteringSurfacesPvol.push_back(pvol->id());
          // store transformation to placed volume and id to logical volume
          rootShell.fEnteringSurfacesPvolTrans.push_back(vol_trans_id);
          rootShell.fEnteringSurfacesLvolIds.push_back(pvol->GetLogicalVolume()->id());
        }
      }
    }
  }
}

template <typename Real_t>
void BrepHelper<Real_t>::FindConvexBooleanSurfaces()
{
  using Real_b = typename SurfData<Real_t>::Real_b;

  std::vector<vecgeom::LogicalVolume const *> lvols;
  vecgeom::GeoManager::Instance().GetAllLogicalVolumes(lvols);

  long nsurf_bool      = 0;
  long nsurf_bool_conv = 0;

  for (auto lvol : lvols) {
    // Get the shell of the root LV
    auto &rootShell = fCPUdata.fShells[lvol->id()];

    // std::cout << " CHECKING SHELL WITH ID " << lvol->id() << std::endl;
    // for accessing the AABBs we need the BVH AABB precision Real_b
    auto boxes = vecgeom::ABBoxManager<Real_b>::Instance().fVolToSurfaceABBoxesMap[lvol->id()];

    // loop over all exiting surfaces of the shell to check each surfaces for convexity
    for (auto idsurf = 0u; idsurf < rootShell.fExitingSurfaces.size(); idsurf++) {

      bool surf_convex = true;
      bool inner_surf  = false;
      // Get the surface
      auto exiting_ind = rootShell.fExitingSurfaces[idsurf];
      auto &localSurface =
          fCPUdata.fLocalSurfaces[rootShell.fSurfaces[exiting_ind]]; // NOT const because we potentially change the
                                                                     // logic id!

      if (localSurface.fLogicId == 0) continue; // skip non-boolean surfaces
      nsurf_bool++;
      if (localSurface.fLogicId < 0) continue; // flipped surfaces cannot be made non-boolean at this point

      if (localSurface.fSkipConvexity) continue;
      switch (localSurface.fSurface.type) {
      case SurfaceType::kCylindrical:
      case SurfaceType::kSpherical:
        if (localSurface.fSurface.IsFlipped()) inner_surf = true;
        break;
      case SurfaceType::kConical:
        if (localSurface.fSurface.IsFlipped()) inner_surf = true;
        break;
      case SurfaceType::kTorus:
        if (fCPUdata.fTorusData[localSurface.fSurface.id].IsFlipped()) inner_surf = true;
        break;
      default:
        break;
      }
      // Local transformation of this surface
      auto const &surfaceTransform    = localSurface.fTrans;
      auto const inv_surfaceTransform = surfaceTransform.Inverse();

      // printf("\n\nStart checking convexity of surface %i with transformation \n", idsurf);
      // surfaceTransform.Print();
      // printf("\n");
      for (auto other_idsurf = 0u; other_idsurf < rootShell.fExitingSurfaces.size(); other_idsurf++) {

        if (other_idsurf == idsurf) continue;

        auto other_exiting_ind        = rootShell.fExitingSurfaces[other_idsurf];
        auto const other_localSurface = fCPUdata.fLocalSurfaces[rootShell.fSurfaces[other_exiting_ind]];

        // skip flipped surfaces if the checked surface is also flipped, because the bounding box can be fully inside,
        // while the surface is still outside, leading to false convex surfaces
        switch (other_localSurface.fSurface.type) {
        case SurfaceType::kCylindrical:
        case SurfaceType::kSpherical:
          if ((other_localSurface.fSurface.IsFlipped()) && inner_surf) surf_convex = false;
          break;
        case SurfaceType::kConical:
          if (other_localSurface.fSurface.IsFlipped() && inner_surf) surf_convex = false;
          break;
        case SurfaceType::kTorus:
          if (fCPUdata.fTorusData[other_localSurface.fSurface.id].IsFlipped() && inner_surf) surf_convex = false;
          break;
        default:
          break;
        }

        bool flip = localSurface.fLogicId < 0;
        flip ^= other_localSurface.fLogicId < 0;
        Real_t tol = 10000 * vecgeom::kToleranceStrict<Real_t>;
        // Real_t tol = other_localSurface.fLogicId < 0 ? 10000 * vecgeom::kToleranceStrict<Real_t>
        //                                              : 10000 * vecgeom::kToleranceStrict<Real_t>;
        // auto const &other_surfaceTransform = fCPUdata.fLocalTrans[other_localSurface.fTrans];
        // printf("checking other surface %i with transformation \n", other_idsurf);
        // other_surfaceTransform.Print();
        // printf("\n");

        // get other bounding box
        auto const other_lowert = boxes[2 * other_idsurf];
        auto const other_uppert = boxes[2 * other_idsurf + 1];
        Vector3D<double> other_lower(other_lowert[0], other_lowert[1], other_lowert[2]);
        Vector3D<double> other_upper(other_uppert[0], other_uppert[1], other_uppert[2]);
        // printf("BB: lower other %f %f %f\n", other_lower.x(), other_lower.y(), other_lower.z());
        // printf("BB: upper other %f %f %f\n", other_upper.x(), other_upper.y(), other_upper.z());

        // transform bounding into frame of the surface to check against
        vecgeom::ABBoxManager<double>::Instance().TransformBoundingBox(other_lower, other_upper, inv_surfaceTransform);
        // define the 8 corner points of the bounding box
        Vector3D<Real_t> corner1(other_lower.x(), other_lower.y(), other_lower.z());
        Vector3D<Real_t> corner2(other_upper.x(), other_lower.y(), other_lower.z());
        Vector3D<Real_t> corner3(other_lower.x(), other_upper.y(), other_lower.z());
        Vector3D<Real_t> corner4(other_upper.x(), other_upper.y(), other_lower.z());
        Vector3D<Real_t> corner5(other_lower.x(), other_lower.y(), other_upper.z());
        Vector3D<Real_t> corner6(other_upper.x(), other_lower.y(), other_upper.z());
        Vector3D<Real_t> corner7(other_lower.x(), other_upper.y(), other_upper.z());
        Vector3D<Real_t> corner8(other_upper.x(), other_upper.y(), other_upper.z());
        // check for inside
        bool inside_c1 = localSurface.fSurface.Inside(corner1, fCPUdata, flip, tol);
        bool inside_c2 = localSurface.fSurface.Inside(corner2, fCPUdata, flip, tol);
        bool inside_c3 = localSurface.fSurface.Inside(corner3, fCPUdata, flip, tol);
        bool inside_c4 = localSurface.fSurface.Inside(corner4, fCPUdata, flip, tol);
        bool inside_c5 = localSurface.fSurface.Inside(corner5, fCPUdata, flip, tol);
        bool inside_c6 = localSurface.fSurface.Inside(corner6, fCPUdata, flip, tol);
        bool inside_c7 = localSurface.fSurface.Inside(corner7, fCPUdata, flip, tol);
        bool inside_c8 = localSurface.fSurface.Inside(corner8, fCPUdata, flip, tol);

        // printf(" Is inside ? 8 corners: %i %i %i %i %i %i %i %i\n", inside_c1, inside_c2, inside_c3, inside_c4, inside_c5, inside_c6, inside_c7, inside_c8);
        // printf(" z value ? 2 values: lower point in z %.15f upper point in z %.15f\n", other_lower.z(), other_upper.z());

        surf_convex &=
            inside_c1 && inside_c2 && inside_c3 && inside_c4 && inside_c5 && inside_c6 && inside_c7 && inside_c8;
      } // inner loop over each surface that is checked against
      if (surf_convex) {
        // printf("Found Convex surface, setting logicId to 0");
        localSurface.fLogicId = 0;
        nsurf_bool_conv++;
      } else {
        // printf("Surface concave, need to do full boolean check");
      }
    } // loop over initial surfaces
  }
  if (fVerbose > 0)
    printf("\n Boolean Surface Convexity Check:\n convex boolean surfaces: %li total number of boolean surfaces %li "
           "ratio: %f\n",
           nsurf_bool_conv, nsurf_bool, double(nsurf_bool_conv) / nsurf_bool);
}

template <typename Real_t>
void BrepHelper<Real_t>::PrintCandidates(vecgeom::NavigationState const &state)
{
  printf("\nCandidate surfaces per state: ");
  state.Print();
  unsigned short scene_id = 0, newscene_id = 0;
  bool is_scene = state.GetSceneId(scene_id, newscene_id);
  printf("is_scene=%d  scene_id=%d  newscene_id=%d\n", int(is_scene), scene_id, newscene_id);
  auto &cand          = fSurfData->GetCandidates(scene_id, state.GetId());
  auto &cand_newscene = fSurfData->GetCandidates(newscene_id, 0);
  int ncand           = is_scene ? cand.fNExiting + cand_newscene.fNcand - cand_newscene.fNExiting : cand.fNcand;
  printf(" %d candidates: exiting={", ncand);
  for (int i = 0; i < cand.fNExiting; ++i)
    printf("%d (side %d, ind %d) ", cand.fCandidates[i], int(cand.fSides[i]), cand.fFrameInd[i]);
  printf("} entering={");
  if (is_scene) {
    // Entering scene candidates are the entering newscene candidates
    for (int i = cand_newscene.fNExiting; i < cand_newscene.fNcand; ++i)
      printf("%d (side %d, ind %d) ", cand_newscene.fCandidates[i], int(cand_newscene.fSides[i]),
             cand_newscene.fFrameInd[i]);
  } else {
    for (int i = cand.fNExiting; i < cand.fNcand; ++i)
      printf("%d (side %d, ind %d) ", cand.fCandidates[i], int(cand.fSides[i]), cand.fFrameInd[i]);
  }
  printf("}\n");
}

template <typename Real_t>
void BrepHelper<Real_t>::PrintCandidateLists()
{
  vecgeom::NavigationState state;

  // recursive geometry visitor lambda printing the candidates lists
  // We have no direct access from a state (contiguous) id to the actual state index
  typedef std::function<void(vecgeom::VPlacedVolume const *)> func_t;
  func_t printCandidates = [&](vecgeom::VPlacedVolume const *pvol) {
    state.Push(pvol);
    const auto vol = pvol->GetLogicalVolume();
    auto daughters = vol->GetDaughters();
    int nd         = daughters.size();
    PrintCandidates(state);

    // do daughters
    for (int id = 0; id < nd; ++id) {
      printCandidates(daughters[id]);
    }
    state.Pop();
  };

  PrintCandidates(state); // outside state
  printCandidates(vecgeom::GeoManager::Instance().GetWorld());
}

template <typename Real_t>
template <typename Real_i, typename Transformation_t>
void BrepHelper<Real_t>::PrintFramedSurface(FramedSurface<Real_i, Transformation_t> const &surf)
{
  // get frame data
  std::stringstream framedata;
  framedata << "FS: " << to_cstring(surf.fFrame.type);
  switch (surf.fFrame.type) {
  case FrameType::kRing: {
    auto const &data = fCPUdata.fRingMasks[surf.fFrame.id];
    framedata << " { rangeR{" << data.rangeR[0] << ", " << data.rangeR[1] << "}, isFullCirc{"
              << (data.isFullCirc ? "true" : "false") << "}, vecSPhi{" << data.vecSPhi << "}, vecEPhi{" << data.vecEPhi
              << "}}";
  } break;
  case FrameType::kZPhi: {
    auto const &data = fCPUdata.fZPhiMasks[surf.fFrame.id];
    framedata << " { rangeZ{" << data.rangeZ[0] << ", " << data.rangeZ[1] << "}, alpha{"
              << vecCore::math::ACos(1. / data.invcalf) * vecgeom::kRadToDeg << "}, vecSPhi{" << data.vecSPhi
              << "}, vecEPhi{" << data.vecEPhi << "}, isFullCirc{" << (data.isFullCirc ? "true" : "false") << "}}";

  } break;
  case FrameType::kWindow: {
    auto const &data = fCPUdata.fWindowMasks[surf.fFrame.id];
    framedata << " { rangeU{" << data.rangeU[0] << ", " << data.rangeU[1] << "}, rangeV{" << data.rangeV[0] << ", "
              << data.rangeV[1] << "}}";
  } break;
  // TODO: Support these
  case FrameType::kTriangle: {
    auto const &data = fCPUdata.fTriangleMasks[surf.fFrame.id];
    framedata << " { p{ 0:{" << data.p_[0] << "}, 1:{" << data.p_[1] << "}, 2:{" << data.p_[2] << "}, n0:{"
              << data.n_[0] << "}, n1:{" << data.n_[1] << "}, n2:{" << data.n_[2] << "}}";
  } break;
  case FrameType::kQuadrilateral: {
    auto const &data = fCPUdata.fQuadMasks[surf.fFrame.id];
    framedata << " { p{ 0:{" << data.p_[0] << "}, 1:{" << data.p_[1] << "}, 2:{" << data.p_[2] << "}, 3:{" << data.p_[3]
              << "}, n0:{" << data.n_[0] << "}, n1:{" << data.n_[1] << "}, n2:{" << data.n_[2] << "}, n3:{"
              << data.n_[3] << "}}";
  } break;
  case FrameType::kNoFrame:
  case FrameType::kRangeZ:
    // return (local[2] > vecgeom::MakeMinusTolerant<true>(u[0]) &&
    //        local[2] < vecgeom::MakePlusTolerant<true>(u[1]));
  case FrameType::kRangeSph:
    // return (rsq > vecgeom::MakeMinusTolerantSquare<true>(u[0]) &&
    //        rsq < vecgeom::MakePlusTolerantSquare<true>(u[1]));
    break;
  default:
    // unhandled
    framedata << " { no such frame type }";
  };

  framedata << "    fTrans{" << surf.fTrans << "} fParent{" << surf.fParent << "} fTraversal{" << surf.fTraversal
            << "} fLogicId{" << surf.fLogicId << "} fNeverCheck{" << surf.fNeverCheck << "} ";
  if (surf.fSceneCS) framedata << "fSceneCS{" << surf.fSceneCS << "} fSceneCSind{" << surf.fSceneCSind << "} ";
  framedata << "fEmbedding{" << to_cstring(surf.fEmbedding) << "} ";
  framedata << "fEmbedded{" << to_cstring(surf.fEmbedded) << "} ";
  framedata << "fVirtualParent{" << to_cstring(surf.fVirtualParent) << "} ";
  if (surf.fFrame.type != FrameType::kNoFrame) framedata << "fSurfIndex{" << surf.fSurfIndex << "} ";
  std::cout << framedata.str() << "\n    fState: ";
  surf.PrintState();
}

template <typename Real_t>
void BrepHelper<Real_t>::PrintCommonSurface(int common_id)
{
  if (common_id >= fSurfData->fNcommonSurf) {
    VECGEOM_LOG(error) << "You are trying to print a non-existent common surface " << common_id;
    return;
  }
  auto round0 = [](Real_t x) { return (std::abs(x) < vecgeom::kToleranceStrict<Real_t>) ? Real_t(0) : x; };
  vecgeom::Vector3D<Real_t> normal;
  const vecgeom::Vector3D<Real_t> lnorm(0, 0, 1);
  auto const &surf = fSurfData->fCommonSurfaces[common_id];
  surf.fTrans.InverseTransformDirection(lnorm, normal);
  vecgeom::NavigationState default_state(surf.fDefaultState);
  int scene_id = surf.GetSceneId();
  printf("\n== common surface %d: type: %s, scene %d, default state: ", common_id, to_cstring(surf.fType), scene_id);
  default_state.Print();
  printf(" transformation: ");
  surf.fTrans.Print();
  switch (surf.fType) {
  case SurfaceType::kPlanar: {
    printf("\n   \x1B[34mleft:\x1B[0m %d surfaces, num_parents=%d, normal: (%g, %g, "
           "%g)\n",
           surf.fLeftSide.fNsurf, surf.fLeftSide.fNumParents, round0(normal[0]), round0(normal[1]), round0(normal[2]));
    break;
  }
  case SurfaceType::kCylindrical: {
    printf("\n   \x1B[34mleft\x1B[0m: %d surfaces, num_parents=%d, {radius{%g}}\n", surf.fLeftSide.fNsurf,
           surf.fLeftSide.fNumParents, fSurfData->fFramedSurf[surf.fLeftSide.fSurfaces[0]].fSurface.fRadius);
    break;
  }
  case SurfaceType::kConical: {
    printf("\n   \x1B[34mleft\x1B[0m: %d surfaces, num_parents=%d, {radius{%g}, slope{%g}}\n", surf.fLeftSide.fNsurf,
           surf.fLeftSide.fNumParents, fSurfData->fFramedSurf[surf.fLeftSide.fSurfaces[0]].fSurface.fRadius,
           fSurfData->fFramedSurf[surf.fLeftSide.fSurfaces[0]].fSurface.fSlope);
    break;
  }
  case SurfaceType::kElliptical:
  case SurfaceType::kSpherical:
  case SurfaceType::kTorus:
  case SurfaceType::kArb4:
  default:
    VECGEOM_LOG(error) << "Surface type " << to_cstring(surf.fType) << " not implemented";
  }
  for (int i = 0; i < surf.fLeftSide.fNsurf; ++i) {
    int idglob         = surf.fLeftSide.fSurfaces[i];
    auto const &placed = fSurfData->fFramedSurf[idglob];
    printf("%d: iframe=%d ", i, idglob);
    PrintFramedSurface(placed);
  }
  if (surf.fRightSide.fNsurf > 0) {
    switch (surf.fType) {
    case SurfaceType::kPlanar: {
      printf("   \x1B[31mright:\x1B[0m %d surfaces, num_parents=%d, normal: (%g, "
             "%g, %g)\n",
             surf.fRightSide.fNsurf, surf.fRightSide.fNumParents, round0(-normal[0]), round0(-normal[1]),
             round0(-normal[2]));
      break;
    }
    case SurfaceType::kCylindrical: {
      printf("   \x1B[31mright:\x1B[0m %d surfaces, num_parents=%d}\n", surf.fRightSide.fNsurf,
             surf.fRightSide.fNumParents);
      break;
    }
    case SurfaceType::kConical: {
      printf("\n   \x1B[31mright\x1B[0m: %d surfaces, num_parents=%d\n", surf.fRightSide.fNsurf,
             surf.fRightSide.fNumParents);
      break;
    }
    case SurfaceType::kElliptical:
    case SurfaceType::kSpherical:
    case SurfaceType::kTorus:
    case SurfaceType::kArb4:
    default:
      VECGEOM_LOG(error) << "Surface type " << to_cstring(surf.fType) << " not implemented";
    }
  } else {
    printf("   \x1B[31mright:\x1B[0m 0 surfaces\n");
  }

  for (int i = 0; i < surf.fRightSide.fNsurf; ++i) {
    int idglob         = surf.fRightSide.fSurfaces[i];
    auto const &placed = fSurfData->fFramedSurf[idglob];
    printf("%d: iframe=%d ", i, idglob);
    PrintFramedSurface(placed);
  }
}

template <typename Real_t>
void BrepHelper<Real_t>::PrintSurfData()
{
  constexpr int megabyte = 1024 * 1024;
  float total = 0, size = 0;
  auto msg = VECGEOM_LOG(info);
  size_t ndivX{0}, ndivY{0}, ndivZ{0}, ndivR{0}, ndivXY{0};
  for (auto const &div : fCPUdata.fSideDivisions) {
    ndivX += div.fAxis == AxisType::kX;
    ndivY += div.fAxis == AxisType::kY;
    ndivZ += div.fAxis == AxisType::kZ;
    ndivR += div.fAxis == AxisType::kR;
    ndivXY += div.fAxis == AxisType::kXY;
  }

  msg << "___________________________________________________________________________________\n";
  msg << " Surface model info:  " << vecgeom::GeoManager::Instance().GetTotalNodeCount() + 1 << " touchables, "
      << fSurfData->fNscenes << " scenes\n";
  size = float(fSurfData->fNshells * sizeof(VolumeShell) + fSurfData->fNlocalSurf * sizeof(int)) / megabyte;
  total += size;
  msg << "    volume shells          = " << fSurfData->fNshells << " [" << size << " MB]\n";
  size =
      float(fSurfData->fNEnteringSurfaces * 4 * sizeof(int) + fSurfData->fNEnteringSurfaces * sizeof(int)) / megabyte;
  total += size;
  msg << "    Entering surface list  = " << fSurfData->fNEnteringSurfaces << " [" << size << " MB]\n";
  size = float(fSurfData->fNExitingSurfaces * 4 * sizeof(int) + fSurfData->fNExitingSurfaces * sizeof(int)) / megabyte;
  total += size;
  msg << "    Exiting surface list   = " << fSurfData->fNExitingSurfaces << " [" << size << " MB]\n";
  size = float(fSurfData->fNvolTrans * sizeof(Transformation)) / megabyte;
  total += size;
  msg << "    volume transformations = " << fSurfData->fNvolTrans << " [" << size << " MB]\n";
  size = float(fSurfData->fNlocalSurf * sizeof(FramedSurface<Real_t, TransformationMP<Real_t>>)) / megabyte;
  total += size;
  msg << "    local surfaces         = " << fSurfData->fNlocalSurf << " [" << size << " MB]\n";
  size = float(fSurfData->fNglobalSurf * sizeof(FramedSurface<Real_t, TransformationMP<Real_t>>)) / megabyte;
  total += size;
  msg << "    global surfaces        = " << fSurfData->fNglobalSurf << " [" << size << " MB]\n";
  size = float(fSurfData->fNcommonSurf * sizeof(CommonSurface<Real_t>) + fSurfData->fNsides * sizeof(int)) / megabyte;
  total += size;
  msg << "    common surfaces        = " << fSurfData->fNcommonSurf << " [" << size << " MB]\n";
  size = float(fSurfData->fNsideDivisions * sizeof(SideDivision_t) + fSurfData->fNslices * sizeof(SliceCand) +
               fSurfData->fNsliceCandidates * sizeof(int)) /
         megabyte;
  total += size;
  msg << "    side divisions         = " << fSurfData->fNsideDivisions << " [" << size << " MB]\n";
  msg << "      planar { X=" << ndivX << "  Y=" << ndivY << "  kR=" << ndivR << " XY=" << ndivXY
      << " } cylindrical { Z=" << ndivZ << " }\n";
  size = float(fSurfData->fSizeCandList * sizeof(int)) / megabyte;
  total += size;
  msg << "    candidates             = " << fSurfData->fSizeCandList << " [" << size << " MB]\n";
  size = float(fSurfData->fNwindows * sizeof(WindowMask_t)) / megabyte;
  total += size;
  msg << "    window masks           = " << fSurfData->fNwindows << " [" << size << " MB]\n";
  size = float(fSurfData->fNrings * sizeof(RingMask_t)) / megabyte;
  total += size;
  msg << "    ring masks             = " << fSurfData->fNrings << " [" << size << " MB]\n";
  size = float(fSurfData->fNzphis * sizeof(ZPhiMask_t)) / megabyte;
  total += size;
  msg << "    Z/phi masks            = " << fSurfData->fNzphis << " [" << size << " MB]\n";
  size = float(fSurfData->fNtriangs * sizeof(TriangleMask_t)) / megabyte;
  total += size;
  msg << "    triangle masks         = " << fSurfData->fNtriangs << " [" << size << " MB]\n";
  size = float(fSurfData->fNquads * sizeof(QuadMask_t)) / megabyte;
  total += size;
  msg << "    quad masks             = " << fSurfData->fNquads << " [" << size << " MB]\n";
  size = float(fSurfData->fNshells * sizeof(int)) / megabyte;
  // Add the internal allocation for each BVH
  for (int i = 0; i < fSurfData->fNshells; i++) {
    size += float(fSurfData->fBVH[fSurfData->fShells[i].fBVH].GetAllocatedSize()) / megabyte;
  }
  total += size;
  msg << "    BVHs                   = " << fSurfData->fNshells << " [" << size << " MB]\n";

  msg << " Total: " << total << "[MB]\n";
  msg << "___________________________________________________________________________________\n";
}

template <typename Real_t>
void BrepHelper<Real_t>::SetNvolumes(int nvolumes)
{
  if (fCPUdata.fShells.size() > 0) {
    VECGEOM_LOG(warning) << "BrepHelper::SetNvolumes already called for this instance";
    return;
  }
  fCPUdata.fShells.resize(nvolumes);
  fCPUdata.fSceneShells.resize(nvolumes);
}

template <typename Real_t>
void BrepHelper<Real_t>::UpdateMaskData()
{
  fSurfData->fNwindows = fCPUdata.fWindowMasks.size();
  delete[] fSurfData->fWindowMasks;
  fSurfData->fWindowMasks = new WindowMask_t[fCPUdata.fWindowMasks.size()];
  for (size_t i = 0; i < fCPUdata.fWindowMasks.size(); ++i)
    fSurfData->fWindowMasks[i] = fCPUdata.fWindowMasks[i];

  fSurfData->fNrings = fCPUdata.fRingMasks.size();
  delete[] fSurfData->fRingMasks;
  fSurfData->fRingMasks = new RingMask_t[fCPUdata.fRingMasks.size()];
  for (size_t i = 0; i < fCPUdata.fRingMasks.size(); ++i)
    fSurfData->fRingMasks[i] = fCPUdata.fRingMasks[i];

  fSurfData->fNzphis = fCPUdata.fZPhiMasks.size();
  delete[] fSurfData->fZPhiMasks;
  fSurfData->fZPhiMasks = new ZPhiMask_t[fCPUdata.fZPhiMasks.size()];
  for (size_t i = 0; i < fCPUdata.fZPhiMasks.size(); ++i)
    fSurfData->fZPhiMasks[i] = fCPUdata.fZPhiMasks[i];

  fSurfData->fNquads = fCPUdata.fQuadMasks.size();
  delete[] fSurfData->fQuadMasks;
  fSurfData->fQuadMasks = new QuadMask_t[fCPUdata.fQuadMasks.size()];
  for (size_t i = 0; i < fCPUdata.fQuadMasks.size(); ++i)
    fSurfData->fQuadMasks[i] = fCPUdata.fQuadMasks[i];

  fSurfData->fNtriangs      = fCPUdata.fTriangleMasks.size();
  fSurfData->fTriangleMasks = new TriangleMask_t[fCPUdata.fTriangleMasks.size()];
  for (size_t i = 0; i < fCPUdata.fTriangleMasks.size(); ++i)
    fSurfData->fTriangleMasks[i] = fCPUdata.fTriangleMasks[i];
}

template <typename Real_t>
void BrepHelper<Real_t>::UpdateSurfData()
{
  // Create and copy surface data

  // Volume transformations (used for placed volumes)
  fSurfData->fNvolTrans = fCPUdata.fPVolTrans.size();
  fSurfData->fPVolTrans = new TransformationMP<Real_t>[fCPUdata.fPVolTrans.size()];
  for (size_t i = 0; i < fCPUdata.fPVolTrans.size(); ++i)
    fSurfData->fPVolTrans[i] = fCPUdata.fPVolTrans[i];

  // Local surfaces (per volume)
  auto numLocalSurf      = fCPUdata.fLocalSurfaces.size();
  fSurfData->fNlocalSurf = numLocalSurf;
  fSurfData->fLocalSurf  = new FramedSurface<Real_t, TransformationMP<Real_t>>[numLocalSurf];
  for (size_t i = 0; i < numLocalSurf; ++i)
    fSurfData->fLocalSurf[i] = fCPUdata.fLocalSurfaces[i];

  // Global surfaces (used on common surfaces)
  auto numGlobalSurf      = fCPUdata.fFramedSurf.size();
  fSurfData->fNglobalSurf = numGlobalSurf;
  fSurfData->fFramedSurf  = new FramedSurface<Real_t, vecgeom::Transformation2DMP<Real_t>>[numGlobalSurf];
  for (size_t i = 0; i < numGlobalSurf; ++i)
    fSurfData->fFramedSurf[i] = FramedSurface<Real_t, vecgeom::Transformation2DMP<Real_t>>(fCPUdata.fFramedSurf[i]);

  // Unplaced surface data
  fSurfData->fNellip    = fCPUdata.fEllipData.size();
  fSurfData->fEllipData = new EllipData_t[fCPUdata.fEllipData.size()];
  for (size_t i = 0; i < fCPUdata.fEllipData.size(); ++i)
    fSurfData->fEllipData[i] = fCPUdata.fEllipData[i];

  fSurfData->fNtorus    = fCPUdata.fTorusData.size();
  fSurfData->fTorusData = new TorusData_t[fCPUdata.fTorusData.size()];
  for (size_t i = 0; i < fCPUdata.fTorusData.size(); ++i)
    fSurfData->fTorusData[i] = fCPUdata.fTorusData[i];

  fSurfData->fNarb4    = fCPUdata.fArb4Data.size();
  fSurfData->fArb4Data = new Arb4Data_t[fCPUdata.fArb4Data.size()];
  for (size_t i = 0; i < fCPUdata.fArb4Data.size(); ++i)
    fSurfData->fArb4Data[i] = fCPUdata.fArb4Data[i];

  // Create Masks
  UpdateMaskData();

  // Copy common surfaces
  size_t size_sides = 0;
  for (auto const &surf : fCPUdata.fCommonSurfaces)
    size_sides += surf.fLeftSide.fNsurf + surf.fRightSide.fNsurf;
  fSurfData->fNsides         = size_sides;
  fSurfData->fSides          = new int[size_sides];
  int *current_side          = fSurfData->fSides;
  fSurfData->fNcommonSurf    = fCPUdata.fCommonSurfaces.size();
  fSurfData->fCommonSurfaces = new CommonSurface<Real_t>[fSurfData->fNcommonSurf];
  for (size_t i = 0; i < fCPUdata.fCommonSurfaces.size(); ++i) {
    // Raw copy of surface (wrong pointers in sides)
    fSurfData->fCommonSurfaces[i] = fCPUdata.fCommonSurfaces[i];
    // Copy left sides content in buffer
    for (auto isurf = 0; isurf < fCPUdata.fCommonSurfaces[i].fLeftSide.fNsurf; ++isurf)
      current_side[isurf] = fCPUdata.fCommonSurfaces[i].fLeftSide.fSurfaces[isurf];
    // Make left sides arrays point to the buffer
    fSurfData->fCommonSurfaces[i].fLeftSide.fSurfaces = current_side;
    current_side += fCPUdata.fCommonSurfaces[i].fLeftSide.fNsurf;
    // Copy right sides content in buffer
    for (auto isurf = 0; isurf < fCPUdata.fCommonSurfaces[i].fRightSide.fNsurf; ++isurf)
      current_side[isurf] = fCPUdata.fCommonSurfaces[i].fRightSide.fSurfaces[isurf];
    // Make right sides arrays point to the buffer
    fSurfData->fCommonSurfaces[i].fRightSide.fSurfaces = current_side;
    current_side += fCPUdata.fCommonSurfaces[i].fRightSide.fNsurf;
    // Copy parent surface indices
    fSurfData->fCommonSurfaces[i].fLeftSide.fNumParents  = fCPUdata.fCommonSurfaces[i].fLeftSide.fNumParents;
    fSurfData->fCommonSurfaces[i].fRightSide.fNumParents = fCPUdata.fCommonSurfaces[i].fRightSide.fNumParents;
  }

  // Copy scenes
  auto nscenes        = fCPUdata.fSceneStartIndex.size();
  fSurfData->fNscenes = nscenes;
  // Copy start indices and sizes per scene
  fSurfData->fSceneStartIndex = new int[nscenes];
  fSurfData->fSceneTouchables = new int[nscenes];
  for (size_t scene_id = 0; scene_id < nscenes; ++scene_id) {
    fSurfData->fSceneStartIndex[scene_id] = fCPUdata.fSceneStartIndex[scene_id];
    fSurfData->fSceneTouchables[scene_id] = fCPUdata.fSceneTouchables[scene_id];
  }

  // Copy candidates lists, frame indices, and sides
  // There is one list of candidates per state
  size_t num_states   = fCPUdata.fCandidatesEntering.size();
  fSurfData->fNStates = num_states;

  auto size_candidates = 0;
  for (auto const &list : fCPUdata.fCandidatesExiting) {
    size_candidates += 2 * list.size() + list.size() * sizeof(char) / sizeof(int) + 1;
  }
  for (auto const &list : fCPUdata.fCandidatesEntering) {
    size_candidates += 2 * list.size() + list.size() * sizeof(char) / sizeof(int) + 1;
  }
  // We may need one additional integer per state
  size_candidates += num_states;

  // Stores candidate, frame indices, and sides
  fSurfData->fSizeCandList = size_candidates;
  fSurfData->fCandList     = new int[size_candidates];
  int *current_candidates  = fSurfData->fCandList;

  fSurfData->fCandidates = new Candidates[num_states];

  for (size_t i = 0; i < num_states; ++i) {
    // Copy Candidate surface, and Frame indices after
    // For each state the memory will contain:
    // EnteringSurfaces - ExitingSurfaces - EnteringFrameIdx - ExitingFrameIdx

    size_t offset = 0;

    // Copy Exiting Candidates
    auto ncandExiting = fCPUdata.fCandidatesExiting[i].size();
    for (size_t icand = 0; icand < ncandExiting; icand++) {
      current_candidates[icand] = (fCPUdata.fCandidatesExiting[i])[icand];
    }

    offset += ncandExiting;

    // Copy Entering Candidates
    auto ncandEntering = fCPUdata.fCandidatesEntering[i].size();
    for (size_t icand = 0; icand < ncandEntering; icand++) {
      current_candidates[icand + offset] = (fCPUdata.fCandidatesEntering[i])[icand];
    }

    offset += ncandEntering;

    // Copy Exiting Frame Indices
    for (size_t icand = 0; icand < ncandExiting; icand++) {
      current_candidates[icand + offset] = (fCPUdata.fFrameIndExiting[i])[icand];
    }

    offset += ncandExiting;

    // Copy Entering Frame Indices
    for (size_t icand = 0; icand < ncandEntering; icand++) {
      current_candidates[icand + offset] = (fCPUdata.fFrameIndEntering[i])[icand];
    }

    offset += ncandEntering;

    // Copy exiting sides
    char *current_sides = reinterpret_cast<char *>(current_candidates + offset);
    for (size_t icand = 0; icand < ncandExiting; icand++) {
      current_sides[icand] = (fCPUdata.fSidesExiting[i])[icand];
    }

    // Copy entering sides
    current_sides += ncandExiting;
    for (size_t icand = 0; icand < ncandEntering; icand++) {
      current_sides[icand] = (fCPUdata.fSidesEntering[i])[icand];
    }

    // compute number of integers represented bu the sides (stored as chars)
    auto ncand  = ncandEntering + ncandExiting;
    int add_one = ((ncand * sizeof(char)) % sizeof(int)) > 0 ? 1 : 0;
    offset += ncand * sizeof(char) / sizeof(int) + add_one;

    // Store the necessary information in the Candidates Struct
    fSurfData->fCandidates[i].fNcand    = ncand;
    fSurfData->fCandidates[i].fNExiting = ncandExiting;

    fSurfData->fCandidates[i].fCandidates = current_candidates;
    // Move the pointer to the start of the Frame index list
    fSurfData->fCandidates[i].fFrameInd = current_candidates + ncand;
    // Move the pointer to the start of the sides list
    fSurfData->fCandidates[i].fSides = reinterpret_cast<char *>(current_candidates + 2 * ncand);
    // Move the pointer to the start of the next Candidate index list
    current_candidates += offset;
  }

  // Copy volume shells
  auto numShells      = fCPUdata.fShells.size();
  fSurfData->fNshells = numShells;
  fSurfData->fShells  = new VolumeShell[numShells];

  fSurfData->fSurfShellList = new int[numLocalSurf];
  int *current_surf         = fSurfData->fSurfShellList;

  size_t sizeLogic = 0;
  for (size_t i = 0; i < numShells; ++i) {
    sizeLogic += fCPUdata.fShells[i].fLogic.size();
  }
  fSurfData->fLogicList    = new logic_int[sizeLogic];
  fSurfData->fNlogic       = sizeLogic;
  logic_int *current_logic = fSurfData->fLogicList;

  // Compute the sum of entering surfaces for all Lvols
  for (const auto &shell : fCPUdata.fShells) {
    fSurfData->fNExitingSurfaces += shell.fExitingSurfaces.size();
    fSurfData->fNEnteringSurfaces += shell.fEnteringSurfaces.size();
  }
  fSurfData->fNPlacedVolumes = vecgeom::GeoManager::Instance().GetPlacedVolumesCount();

  fSurfData->fShellExitingSurfaceList           = new int[fSurfData->fNExitingSurfaces];
  fSurfData->fShellEnteringSurfaceList          = new int[fSurfData->fNEnteringSurfaces];
  fSurfData->fShellEnteringSurfacePvolList      = new int[fSurfData->fNEnteringSurfaces];
  fSurfData->fShellEnteringSurfacePvolTransList = new int[fSurfData->fNEnteringSurfaces];
  fSurfData->fShellEnteringSurfaceLvolIdList    = new int[fSurfData->fNEnteringSurfaces];
  fSurfData->fShellDaughterPvolIdList           = new int[fSurfData->fNPlacedVolumes];
  fSurfData->fShellDaughterPvolTransList        = new int[fSurfData->fNPlacedVolumes];

  int *currentExitingSurface                = fSurfData->fShellExitingSurfaceList;
  int *currentEnteringSurface               = fSurfData->fShellEnteringSurfaceList;
  int *currentEnteringSurfacePvol           = fSurfData->fShellEnteringSurfacePvolList;
  int *currentShellEnteringSurfacePvolTrans = fSurfData->fShellEnteringSurfacePvolTransList;
  int *currentShellEnteringSurfaceLvolId    = fSurfData->fShellEnteringSurfaceLvolIdList;
  int *currentShellDaughterPvolId           = fSurfData->fShellDaughterPvolIdList;
  int *currentShellDaughterPvolTrans        = fSurfData->fShellDaughterPvolTransList;

  for (size_t i = 0; i < numShells; ++i) {
    auto const &surfaces         = fCPUdata.fShells[i].fSurfaces;
    auto nsurf                   = surfaces.size();
    fSurfData->fShells[i].fNsurf = nsurf;
    for (size_t isurf = 0; isurf < nsurf; isurf++)
      current_surf[isurf] = surfaces[isurf];
    fSurfData->fShells[i].fSurfaces = current_surf;
    current_surf += nsurf;
    auto nlogic                        = fCPUdata.fShells[i].fLogic.size();
    fSurfData->fShells[i].fLogic.size_ = nlogic;
    fSurfData->fShells[i].fLogic.data_ = current_logic;
    int iitem                          = 0;
    for (auto item : fCPUdata.fShells[i].fLogic)
      fSurfData->fShells[i].fLogic.data_[iitem++] = item;
    current_logic += nlogic;

    auto const &exitingSurfaces           = fCPUdata.fShells[i].fExitingSurfaces;
    auto const &enteringSurfaces          = fCPUdata.fShells[i].fEnteringSurfaces;
    auto const &enteringSurfacesPvol      = fCPUdata.fShells[i].fEnteringSurfacesPvol;
    auto const &enteringSurfacesPvolTrans = fCPUdata.fShells[i].fEnteringSurfacesPvolTrans;
    auto const &enteringSurfacesLvolIds   = fCPUdata.fShells[i].fEnteringSurfacesLvolIds;
    auto const &daughterPvolIds           = fCPUdata.fShells[i].fDaughterPvolIds;
    auto const &daughterPvolTrans         = fCPUdata.fShells[i].fDaughterPvolTrans;
    auto nExitingSurf                     = exitingSurfaces.size();
    auto nEnteringSurf                    = enteringSurfaces.size();
    auto nDaughterPvol                    = daughterPvolIds.size();
    // Init number of exiting/entering surfaces in shell
    fSurfData->fShells[i].fNExitingSurfaces  = nExitingSurf;
    fSurfData->fShells[i].fNEnteringSurfaces = nEnteringSurf;
    fSurfData->fShells[i].fNDaughterPvols    = nDaughterPvol;
    // Copy indices corresponding to this shell to surfData
    for (size_t isurf = 0; isurf < nExitingSurf; isurf++)
      currentExitingSurface[isurf] = exitingSurfaces[isurf];
    for (size_t isurf = 0; isurf < nEnteringSurf; isurf++) {
      currentEnteringSurface[isurf]               = enteringSurfaces[isurf];
      currentEnteringSurfacePvol[isurf]           = enteringSurfacesPvol[isurf];
      currentShellEnteringSurfacePvolTrans[isurf] = enteringSurfacesPvolTrans[isurf];
      currentShellEnteringSurfaceLvolId[isurf]    = enteringSurfacesLvolIds[isurf];
    }
    // Copy PV global IDs
    for (size_t ivol = 0; ivol < nDaughterPvol; ivol++) {
      currentShellDaughterPvolId[ivol]    = daughterPvolIds[ivol];
      currentShellDaughterPvolTrans[ivol] = daughterPvolTrans[ivol];
    }

    // Set the members of this shell to point to the newly copied data
    fSurfData->fShells[i].fExitingSurfaces           = currentExitingSurface;
    fSurfData->fShells[i].fEnteringSurfaces          = currentEnteringSurface;
    fSurfData->fShells[i].fEnteringSurfacesPvol      = currentEnteringSurfacePvol;
    fSurfData->fShells[i].fEnteringSurfacesPvolTrans = currentShellEnteringSurfacePvolTrans;
    fSurfData->fShells[i].fEnteringSurfacesLvolIds   = currentShellEnteringSurfaceLvolId;
    fSurfData->fShells[i].fDaughterPvolIds           = currentShellDaughterPvolId;
    fSurfData->fShells[i].fDaughterPvolTrans         = currentShellDaughterPvolTrans;
    // Move the pointers in surfData for the next shell
    currentExitingSurface += nExitingSurf;
    currentEnteringSurface += nEnteringSurf;
    currentEnteringSurfacePvol += nEnteringSurf;
    currentShellEnteringSurfacePvolTrans += nEnteringSurf;
    currentShellEnteringSurfaceLvolId += nEnteringSurf;
    currentShellDaughterPvolId += nDaughterPvol;
    currentShellDaughterPvolTrans += nDaughterPvol;

    fSurfData->fShells[i].fBVH = fCPUdata.fShells[i].fBVH;
  }
}

// Template instantiations for float and double, allowing to precompile these types
template class BrepHelper<double>;
template class BrepHelper<float>;

} // namespace vgbrep