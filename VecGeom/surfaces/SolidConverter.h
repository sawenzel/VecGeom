#ifndef VECGEOM_SURFACE_SOLIDCONVERTER_H_
#define VECGEOM_SURFACE_SOLIDCONVERTER_H_

#include <cassert>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/LogicHelper.h>
#include <VecGeom/surfaces/CpuTypes.h>

#include <VecGeom/volumes/Box.h>
#include <VecGeom/volumes/Trd.h>
#include <VecGeom/volumes/Tube.h>
#include <VecGeom/volumes/BooleanVolume.h>

namespace vgbrep {

template <typename Real_t>
struct SolidConverter {
  using SurfData_t     = SurfData<Real_t>;
  using CylData_t      = CylData<Real_t>;
  using ConeData_t     = ConeData<Real_t>;
  using SphData_t      = SphData<Real_t>;
  using WindowMask_t   = WindowMask<Real_t>;
  using RingMask_t     = RingMask<Real_t>;
  using ZPhiMask_t     = ZPhiMask<Real_t>;
  using TriangleMask_t = TriangleMask<Real_t>;
  using QuadMask_t     = QuadrilateralMask<Real_t>;
  using CPUsurfData_t  = CPUsurfData<Real_t>;

  CPUsurfData_t &fCPUdata; ///< Reference to data held by BrepHelper

  SolidConverter(CPUsurfData_t &data) : fCPUdata{data} {}

  bool CreateSolidSurfaces(vecgeom::VUnplacedVolume const *solid, int volId, Transformation *localtrans = nullptr)
  {
    bool success{true};
    auto const &shell = fCPUdata.fShells[volId];
    auto isurf_first  = shell.fSurfaces.size();

    auto createSurfacesLocal = [&]() {
      auto box = dynamic_cast<vecgeom::UnplacedBox const *>(solid);
      if (box) return CreateBoxSurfaces(*box, volId);

      auto tube = dynamic_cast<vecgeom::UnplacedTube const *>(solid);
      if (tube) return CreateTubeSurfaces(*tube, volId);

      auto trd = dynamic_cast<vecgeom::UnplacedTrd const *>(solid);
      if (trd) return CreateTrdSurfaces(*trd, volId);

      auto bstruct = vecgeom::BooleanHelper::GetBooleanStruct(solid);
      if (bstruct) return CreateBooleanSurfaces(*bstruct, volId);

      return false;
    };

    success = createSurfacesLocal();

    if (success && localtrans) {
      auto isurf_last = shell.fSurfaces.size();
      for (size_t i = isurf_first; i < isurf_last; ++i) {
        auto const &surf = fCPUdata.fLocalSurfaces[shell.fSurfaces[i]];
        Transformation trans(*localtrans);
        trans.MultiplyFromRight(fCPUdata.fLocalTrans[surf.fTrans]);
        fCPUdata.fLocalTrans[surf.fTrans] = trans;
      }
    }
    return success;
  }

  // Specific solid converters
  // The code for creating solid-specific surfaces should sit in the specific solid struct type
  bool CreateBoxSurfaces(vecgeom::UnplacedBox const &box, int logical_id)
  {
    const bool use_surf_safety = true;
    int isurf;
    LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3 & 4 & 5
    // surface at -dx:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.y(), box.z()}),
                               CreateLocalTransformation({-box.x(), 0, 0, -90, 90, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(isurf);
    // surface at +dx:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.y(), box.z()}),
                               CreateLocalTransformation({box.x(), 0, 0, 90, 90, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
    // surface at -dy:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.x(), box.z()}),
                               CreateLocalTransformation({0, -box.y(), 0, 0, 90, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
    // surface at +dy:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.x(), box.z()}),
                               CreateLocalTransformation({0, box.y(), 0, 0, -90, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
    // surface at -dz:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.x(), box.y()}),
                               CreateLocalTransformation({0, 0, -box.z(), 0, 180, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
    // surface at +dz:
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{box.x(), box.y()}),
                               CreateLocalTransformation({0, 0, box.z(), 0, 0, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
    AddLogicToShell(logical_id, logic);
    return true;
  }

  bool CreateTubeSurfaces(vecgeom::UnplacedTube const &tube, int logical_id)
  {
    LogicExpressionCPU logic; // top & bottom & [rmin] & rmax & (dphi < 180) ? sphi * ephi : sphi | ephi
    auto sphi = tube.sphi();
    auto dphi = tube.dphi();
    auto ephi = tube.sphi() + tube.dphi();

    assert(dphi > vecgeom::kTolerance);

    auto Rmean = (tube.rmin() + tube.rmax()) / 2;
    auto Rdiff = (tube.rmax() - tube.rmin()) / 2;

    assert(Rdiff > 0);

    bool fullCirc        = ApproxEqual(dphi, vecgeom::kTwoPi);
    bool smallerPi       = dphi < (vecgeom::kPi - vecgeom::kTolerance);
    bool use_surf_safety = true;

    int isurf;
    Real_t surfdata[2];

    // We need angles in degrees for transformations
    auto sphid = vecgeom::kRadToDeg * sphi;
    auto ephid = vecgeom::kRadToDeg * ephi;

    // surface at +dz
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar),
                               CreateFrame(kRing, RingMask_t{tube.rmin(), tube.rmax(), fullCirc, sphi, ephi}),
                               CreateLocalTransformation({0, 0, tube.z(), 0, 0, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(isurf);
    // surface at -dz
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar),
                               CreateFrame(kRing, RingMask_t{tube.rmin(), tube.rmax(), fullCirc, sphi, ephi}),
                               CreateLocalTransformation({0, 0, -tube.z(), 0, 180, -sphid - ephid}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
    // inner cylinder
    if (tube.rmin() > vecgeom::kTolerance) {
      surfdata[0] = tube.rmin();
      isurf       = CreateLocalSurface(CreateUnplacedSurface(kCylindrical, surfdata, /*flipped=*/true),
                                       CreateFrame(kZPhi, ZPhiMask_t{-tube.z(), tube.z(), fullCirc, sphi, ephi}),
                                       CreateLocalTransformation({0, 0, 0, 0, 0, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);
    }
    // outer cylinder
    surfdata[0] = tube.rmax();
    isurf       = CreateLocalSurface(CreateUnplacedSurface(kCylindrical, surfdata),
                                     CreateFrame(kZPhi, ZPhiMask_t{-tube.z(), tube.z(), fullCirc, sphi, ephi}),
                                     CreateLocalTransformation({0, 0, 0, 0, 0, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);

    if (ApproxEqual(dphi, vecgeom::kTwoPi)) {
      AddLogicToShell(logical_id, logic);
      return true;
    }
    // plane cap at Sphi
    isurf =
        CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{Rdiff, tube.z()}),
                           CreateLocalTransformation({Rmean * std::cos(sphi), Rmean * std::sin(sphi), 0, sphid, 90, 0}),
                           use_surf_safety && smallerPi);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(lplus); // '('
    logic.push_back(isurf);

    // plane cap at Sphi+Dphi
    isurf = CreateLocalSurface(
        CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{Rdiff, tube.z()}),
        CreateLocalTransformation({Rmean * std::cos(ephi), Rmean * std::sin(ephi), 0, ephid, -90, 0}),
        use_surf_safety && smallerPi);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(smallerPi ? land : lor);
    logic.push_back(isurf);
    logic.push_back(lminus); // ')'
    AddLogicToShell(logical_id, logic);
    return true;
  }

  bool CreateTrdSurfaces(vecgeom::UnplacedTrd const &trd, int logical_id)
  {
    LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3 & 4 & 5
    bool use_surf_safety = true;
    auto dx              = trd.dx1() - trd.dx2();
    auto dy              = trd.dy1() - trd.dy2();
    auto dzx             = vecgeom::Sqrt(4 * trd.dz() * trd.dz() + dy * dy) * 0.5;
    auto dzy             = vecgeom::Sqrt(4 * trd.dz() * trd.dz() + dx * dx) * 0.5;

    auto phix = ApproxEqual(dy, 0.) ? 90 : vecgeom::ATan(2 * trd.dz() / dy) * vecgeom::kRadToDeg;
    auto phiy = ApproxEqual(dx, 0.) ? 90 : vecgeom::ATan(2 * trd.dz() / dx) * vecgeom::kRadToDeg;
    if (phix < 0) phix = 180 + phix;
    if (phiy < 0) phiy = 180 + phiy;

    auto movey = (trd.dy1() + trd.dy2()) * 0.5;
    auto movex = (trd.dx1() + trd.dx2()) * 0.5;

    // Bottom face
    int isurf =
        CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{trd.dx1(), trd.dy1()}),
                           CreateLocalTransformation({0, 0, -trd.dz(), 0, 180, 0}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(isurf);

    // Top face
    isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{trd.dx2(), trd.dy2()}),
                               CreateLocalTransformation({0, 0, trd.dz()}), use_surf_safety);
    AddSurfaceToShell(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);

    // Sides parallel to x axis
    if (vecgeom::Abs(dx) > vecgeom::kTolerance) {
      // At -dy
      isurf = CreateLocalSurface(
          CreateUnplacedSurface(kPlanar),
          CreateFrame(kQuadrilateral, QuadMask_t{-trd.dx1(), -dzx, trd.dx1(), -dzx, trd.dx2(), dzx, -trd.dx2(), dzx}),
          CreateLocalTransformation({0, -movey, 0, 0, phix, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);

      // At +dy
      isurf = CreateLocalSurface(
          CreateUnplacedSurface(kPlanar),
          CreateFrame(kQuadrilateral, QuadMask_t{-trd.dx1(), -dzx, trd.dx1(), -dzx, trd.dx2(), dzx, -trd.dx2(), dzx}),
          CreateLocalTransformation({0, movey, 0, 180, phix, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);
    } else { // We have rectangles.
      isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{trd.dx1(), dzx}),
                                 CreateLocalTransformation({0, -movey, 0, 0, phix, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);

      // At +dy
      isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{trd.dx1(), dzx}),
                                 CreateLocalTransformation({0, movey, 0, 180, phix, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);
    }
    // Sides parallel to y axis
    if (vecgeom::Abs(dy) > vecgeom::kTolerance) {
      // At -dx
      isurf = CreateLocalSurface(
          CreateUnplacedSurface(kPlanar),
          CreateFrame(kQuadrilateral, QuadMask_t{-trd.dy1(), -dzy, trd.dy1(), -dzy, trd.dy2(), dzy, -trd.dy2(), dzy}),
          CreateLocalTransformation({-movex, 0, 0, -90, phiy, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);

      // At +dx
      isurf = CreateLocalSurface(
          CreateUnplacedSurface(kPlanar),
          CreateFrame(kQuadrilateral, QuadMask_t{-trd.dy1(), -dzy, trd.dy1(), -dzy, trd.dy2(), dzy, -trd.dy2(), dzy}),
          CreateLocalTransformation({movex, 0, 0, 90, phiy, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);
    } else { // We have rectangles.
      // At -dx
      isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{trd.dy1(), dzy}),
                                 CreateLocalTransformation({-movex, 0, 0, -90, phiy, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);

      // At +dx
      isurf = CreateLocalSurface(CreateUnplacedSurface(kPlanar), CreateFrame(kWindow, WindowMask_t{trd.dy1(), dzy}),
                                 CreateLocalTransformation({movex, 0, 0, 90, phiy, 0}), use_surf_safety);
      AddSurfaceToShell(logical_id, isurf);
      logic.push_back(land);
      logic.push_back(isurf);
    }
    AddLogicToShell(logical_id, logic);
    return true;
  }

  bool CreateBooleanSurfaces(vecgeom::BooleanStruct const &bstruct, int logical_id)
  {
    using Bnode = logichelper::Bnode;
    Transformation trans;
    Bnode top(trans, logical_id, bstruct);
    AppendLogicTo(top, fCPUdata.fShells[logical_id].fLogic);
    // Assign logic id to each surface
    for (auto id : fCPUdata.fShells[logical_id].fSurfaces) {
      // All frames of Boolean surfaces must be checked
      fCPUdata.fLocalSurfaces[id].fUseSurfSafety = false;
      // Set the logic id for Boolean surfaces
      if (logichelper::is_negated(id, fCPUdata.fShells[logical_id].fLogic))
        fCPUdata.fLocalSurfaces[id].fLogicId = -id;
      else
        fCPUdata.fLocalSurfaces[id].fLogicId = id;
    }
    return true;
  }

  void AppendLogicTo(logichelper::Bnode const &node, LogicExpressionCPU &logic, bool negate = false)
  {
    using Bnode  = logichelper::Bnode;
    using Placed = logichelper::Placed;
    // open paranthesys
    logic.push_back(lplus);

    auto left_complexity  = node.left_->GetComplexity();
    auto right_complexity = node.right_->GetComplexity();
    // use commutativity of AND/OR to keep left node as least complex
    bool swap = left_complexity > right_complexity;
    if (swap) {
      auto complexity  = left_complexity;
      left_complexity  = right_complexity;
      right_complexity = complexity;
    }
    Placed *new_left   = swap ? node.right_ : node.left_;
    Placed *new_right  = swap ? node.left_ : node.right_;
    bool new_neg_left  = swap ? node.neg_right_ : node.neg_left_;
    bool new_neg_right = swap ? node.neg_left_ : node.neg_right_;

    // left node
    if (left_complexity > 0) {
      auto left_node = static_cast<Bnode *>(new_left);
      AppendLogicTo(*left_node, logic, negate ^ new_neg_left);
    } else {
      // Leaf node. Check if negated.
      // append leaf expression
      logic.push_back(lplus);
      size_t istart = logic.size();
      CreateSolidSurfaces(new_left->fUnplaced, new_left->fVolId, &new_left->fTrans);
      int inserts = 0;
      if (new_neg_left ^ negate) logichelper::negate_logic(logic, istart, logic.size() - 1, inserts);
      // printf("inserted left solid logic: ");
      // logichelper::print_logic(logic, istart, 0, false);
      logic.push_back(lminus);
    }

    switch (node.op_) {
    case vecgeom::kUnion:
      if (negate)
        logic.push_back(land);
      else
        logic.push_back(lor);
      break;
    case vecgeom::kIntersection:
      if (negate)
        logic.push_back(lor);
      else
        logic.push_back(land);
      break;
    case vecgeom::kSubtraction:
      printf("cannot find subtraction here\n");
      break;
    };

    // right node
    if (right_complexity > 0) {
      auto right_node = static_cast<Bnode *>(new_right);
      AppendLogicTo(*right_node, logic, negate ^ new_neg_right);
    } else {
      // Leaf node. Check if negated.
      // append leaf expression
      logic.push_back(lplus);
      size_t istart = logic.size();
      CreateSolidSurfaces(new_right->fUnplaced, new_right->fVolId, &new_right->fTrans);
      int inserts = 0;
      if (new_neg_right ^ negate) logichelper::negate_logic(logic, istart, logic.size() - 1, inserts);
      // printf("inserted right solid logic: ");
      // logichelper::print_logic(logic, istart, 0, false);
      logic.push_back(lminus);
    }

    logic.push_back(lminus);
  }

  UnplacedSurface CreateUnplacedSurface(SurfaceType type, Real_t *data = nullptr, bool flip = false)
  {
    switch (type) {
    case kPlanar:
      return UnplacedSurface(type);
    case kCylindrical:
    case kSpherical:
      fCPUdata.fCylSphData.push_back({data[0], flip});
      return UnplacedSurface(type, fCPUdata.fCylSphData.size() - 1);
    case kConical:
      fCPUdata.fConeData.push_back({data[0], data[1], flip});
      return UnplacedSurface(type, fCPUdata.fConeData.size() - 1);
    case kTorus:
    case kGenSecondOrder:
      std::cout << "kTorus, kGenSecondOrder unhandled\n";
      return UnplacedSurface(type);
    };
    return UnplacedSurface(type);
  }

  // Creators for different types of frames.
  Frame CreateFrame(FrameType type, WindowMask<Real_t> const &mask)
  {
    int id = fCPUdata.fWindowMasks.size();
    fCPUdata.fWindowMasks.push_back(mask);
    return Frame(type, id);
  }

  Frame CreateFrame(FrameType type, RingMask<Real_t> const &mask)
  {
    int id = fCPUdata.fRingMasks.size();
    fCPUdata.fRingMasks.push_back(mask);
    return Frame(type, id);
  }

  Frame CreateFrame(FrameType type, ZPhiMask<Real_t> const &mask)
  {
    int id = fCPUdata.fZPhiMasks.size();
    fCPUdata.fZPhiMasks.push_back(mask);
    return Frame(type, id);
  }

  Frame CreateFrame(FrameType type, QuadrilateralMask<Real_t> const &mask)
  {
    int id = fCPUdata.fQuadMasks.size();
    fCPUdata.fQuadMasks.push_back(mask);
    return Frame(type, id);
  }

  int CreateLocalTransformation(Transformation const &trans)
  {
    int id = fCPUdata.fLocalTrans.size();
    fCPUdata.fLocalTrans.push_back(trans);
    return id;
  }

  int CreateLocalSurface(UnplacedSurface const &unplaced, Frame const &frame, int trans, bool use_surf_safety)
  {
    int id = fCPUdata.fLocalSurfaces.size();
    fCPUdata.fLocalSurfaces.push_back({unplaced, frame, trans, use_surf_safety});
    return id;
  }

  int AddSurfaceToShell(int logical_id, int isurf)
  {
    if (fCPUdata.fShells.size() == 0) {
      std::cout << "BrepHelper::AddSurfaceToShell: need to call SetNvolumes first\n";
      return -1;
    }
    assert(logical_id < (int)fCPUdata.fShells.size() && "surface shell id exceeding number of volumes");
    int id = fCPUdata.fShells[logical_id].fSurfaces.size();
    fCPUdata.fShells[logical_id].fSurfaces.push_back(isurf);
    return id;
  }

  void AddLogicToShell(int logical_id, LogicExpressionCPU &logic)
  {
    // Add solid logic to existing shell logic
    auto &crtlogic = fCPUdata.fShells[logical_id].fLogic;
    crtlogic.insert(crtlogic.end(), logic.begin(), logic.end());
  }
};

} // namespace vgbrep
#endif
