#ifndef VECGEOM_SURFACE_POLYCONECONVERTER_H_
#define VECGEOM_SURFACE_POLYCONECONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Polycone.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Polycone
/// @tparam Real_t Precision type
/// @param polycone Polycone solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreatePolyconeSurfaces(vecgeom::UnplacedPolycone const &polycone, int logical_id)
{
  using RingMask_t = RingMask<Real_t>;
  using ZPhiMask_t = ZPhiMask<Real_t>;
  using Vector3D   = vecgeom::Vector3D<vecgeom::Precision>;

  LogicExpressionCPU
      logic; // top & bottom & [rmin] & rmax & (dphi < 180) ? sphi * ephi : sphi | ephi  auto rmin1 = cone.GetRmin1();

  auto nSect = polycone.GetNSections();
  auto sphi  = polycone.GetStartPhi();
  auto dphi  = polycone.GetDeltaPhi();
  auto ephi  = polycone.GetEndPhi();
  assert(dphi > vecgeom::kTolerance);

  bool fullCirc  = ApproxEqual(dphi, vecgeom::kTwoPi);
  bool smallerPi = dphi < (vecgeom::kPi - vecgeom::kTolerance);

  int isurf;
  vecgeom::Precision surfdata[2];

  // We need angles in degrees for transformations
  auto sphid = vecgeom::kRadToDeg * sphi;
  auto ephid = vecgeom::kRadToDeg * ephi;

  vecgeom::Transformation3D transformation;
  std::vector<Vector3D> vert; // Stores 3D coordinates of the four corners that create a quadrilateral.
  auto csphi = std::cos(sphi);
  auto ssphi = std::sin(sphi);
  auto cephi = std::cos(ephi);
  auto sephi = std::sin(ephi);

  // loop over sections, at each section the full cone-like solid is constructed
  for (int i = 0; i < nSect; i++) {
    logic.push_back(lplus); // '(' parenthesis around section

    auto z1    = polycone.GetZAtPlane(i);     // note: indices for GetZAtPlane is the
    auto z2    = polycone.GetZAtPlane(i + 1); // number of distinct z points of the polycone
    auto rmin1 = polycone.GetRmin1AtSection(i);
    auto rmax1 = polycone.GetRmax1AtSection(i);
    auto rmin2 = polycone.GetRmin2AtSection(i);
    auto rmax2 = polycone.GetRmax2AtSection(i);

    assert(rmax1 - rmin1 > -vecgeom::kTolerance);
    assert(rmax2 - rmin2 > -vecgeom::kTolerance);

    // if rmax == rmin then for safety it is set to rmax = rmin + ConeTolerance already in the solid model
    // For using mixed precision, the compiled tolerance must be replaced by the mixed tolerance
    if (rmax1 - rmin1 == vecgeom::kConeTolerance) rmax1 = rmin1 + vecgeom::kToleranceCone<Real_t>;
    if (rmax2 - rmin2 == vecgeom::kConeTolerance) rmax2 = rmin2 + vecgeom::kToleranceCone<Real_t>;

    // construct top and bottom surfaces:
    // if there are previous or following sections, the top and bottom surface consist
    // of up to two rings (inner and outer ring). All of those are real surfaces

    // lets start with the adjusted bottom rings:
    // case where there is a previous section
    if (i > 0) {
      auto rmin_prev = polycone.GetRmin2AtSection(i - 1);
      auto rmax_prev = polycone.GetRmax2AtSection(i - 1);

      bool has_inner_ring = rmin1 < rmin_prev - vecgeom::kTolerance;
      bool has_outer_ring = rmax1 > rmax_prev + vecgeom::kTolerance;

      if (has_inner_ring) {
        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
            builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmin1, rmin_prev, fullCirc, sphi, ephi}),
            builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z1, 0, 180, -sphid - ephid}));
        builder::GetSurface<Real_t>(isurf).fEmbedding = false;
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        logic.push_back(isurf);
        logic.push_back(land);
      }
      if (has_outer_ring) {
        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
            builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmax_prev, rmax1, fullCirc, sphi, ephi}),
            builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z1, 0, 180, -sphid - ephid}));
        builder::GetSurface<Real_t>(isurf).fEmbedding = false;
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        logic.push_back(isurf);
        logic.push_back(land);
      }

      // add virtual surface at z1 if there is no real one
      if (!has_inner_ring && !has_outer_ring) {
        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar), Frame{FrameType::kNoFrame},
            builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z1, 0, 180, 0}));
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        logic.push_back(isurf);
        logic.push_back(land);
      }
    }
    // adjusted top rings in case there is a following section
    if (i < nSect - 1) {
      auto rmin_next = polycone.GetRmin1AtSection(i + 1);
      auto rmax_next = polycone.GetRmax1AtSection(i + 1);

      bool has_inner_ring = rmin2 < rmin_next - vecgeom::kTolerance;
      bool has_outer_ring = rmax2 > rmax_next + vecgeom::kTolerance;

      if (has_inner_ring) {
        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
            builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmin2, rmin_next, fullCirc, sphi, ephi}),
            builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z2, 0, 0, 0}));
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        logic.push_back(isurf);
        logic.push_back(land);
      }
      if (has_outer_ring) {
        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
            builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmax_next, rmax2, fullCirc, sphi, ephi}),
            builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z2, 0, 0, 0}));
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        logic.push_back(isurf);
        logic.push_back(land);
      }

      // add virtual surface at z2 if there is no real one
      if (!has_inner_ring && !has_outer_ring) {
        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar), Frame{FrameType::kNoFrame},
            builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z2, 0, 0, 0}));
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        logic.push_back(isurf);
        logic.push_back(land);
      }
    }

    // first section has the full initial ring as bottom surface
    if (i == 0) {
      // real surface at z1
      if (rmax1 - rmin1 > vecgeom::kTolerance) {
        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
            builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmin1, rmax1, fullCirc, sphi, ephi}),
            builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z1, 0, 180, -sphid - ephid}));
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        logic.push_back(isurf);
        logic.push_back(land);
      }
    }

    // last section has the full initial ring as top surface
    if (i == nSect - 1) {
      // real surface at z2
      if (rmax1 - rmin1 > vecgeom::kTolerance) {
        isurf = builder::CreateLocalSurface<Real_t>(
            builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
            builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmin2, rmax2, fullCirc, sphi, ephi}),
            builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z2, 0, 0, 0}));
        builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
        logic.push_back(isurf);
        logic.push_back(land);
      }
    }

    // Cones:
    const Real_t z_shift = 0.5 * (z1 + z2);
    // inner cone
    if (rmin2 > vecgeom::kTolerance || rmin1 > vecgeom::kTolerance) {
      surfdata[0] = 0.5 * (rmin1 + rmin2);
      surfdata[1] = (rmin2 - rmin1) / (z2 - z1);
      auto stype  = std::abs(surfdata[1]) > vecgeom::kTolerance ? SurfaceType::kConical : SurfaceType::kCylindrical;
      isurf       = builder::CreateLocalSurface<Real_t>(
          builder::CreateUnplacedSurface<Real_t>(stype, surfdata, /*flipped=*/true),
          builder::CreateFrame<Real_t>(FrameType::kZPhi,
                                       ZPhiMask_t{z1 - z_shift, z2 - z_shift, fullCirc, rmin1, rmin2, sphi, ephi}),
          builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z_shift, 0, 0, 0}));
      builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
      logic.push_back(isurf);
      logic.push_back(land);
    }
    // outer cone
    surfdata[0] = 0.5 * (rmax1 + rmax2);
    surfdata[1] = (rmax2 - rmax1) / (z2 - z1);
    auto stype  = std::abs(surfdata[1]) > vecgeom::kTolerance ? SurfaceType::kConical : SurfaceType::kCylindrical;
    isurf       = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(stype, surfdata),
        builder::CreateFrame<Real_t>(FrameType::kZPhi,
                                     ZPhiMask_t{z1 - z_shift, z2 - z_shift, fullCirc, rmax1, rmax2, sphi, ephi}),
        builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, z_shift, 0, 0, 0}));
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(isurf);

    if (!fullCirc) {
      // corners of the sphi and ephi faces
      std::vector<Vector3D> corners = {{rmin1 * csphi, rmin1 * ssphi, z1}, {rmax1 * csphi, rmax1 * ssphi, z1},
                                       {rmax2 * csphi, rmax2 * ssphi, z2}, {rmin2 * csphi, rmin2 * ssphi, z2},
                                       {rmax1 * cephi, rmax1 * sephi, z1}, {rmin1 * cephi, rmin1 * sephi, z1},
                                       {rmin2 * cephi, rmin2 * sephi, z2}, {rmax2 * cephi, rmax2 * sephi, z2}};

      // plane cap at sphi
      vert  = {corners[0], corners[1], corners[2], corners[3]};
      isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
      assert(isurf >= 0);
      if (!smallerPi) builder::GetSurface<Real_t>(isurf).fEmbedding = false;
      logic.push_back(land);
      logic.push_back(lplus); // '('
      logic.push_back(isurf);

      // plane cap at sphi+dphi
      vert  = {corners[4], corners[5], corners[6], corners[7]};
      isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
      assert(isurf >= 0);
      if (!smallerPi) builder::GetSurface<Real_t>(isurf).fEmbedding = false;
      logic.push_back(smallerPi ? land : lor);
      logic.push_back(isurf);
      logic.push_back(lminus); // ')'
    }

    logic.push_back(lminus);                 // ')' // close parenthesis after section
    if (i < nSect - 1) logic.push_back(lor); // union with next section if any
  }                                          // end loop over sections
  builder::AddLogicToShell<Real_t>(logical_id, logic);

  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
