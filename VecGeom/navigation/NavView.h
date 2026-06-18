//------------------------------- -*- C++ -*- -------------------------------//
// Copyright VecGeom contributors: see top-level LICENSE file for details
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
//! \file VecGeom/navigation/NavView.h
//---------------------------------------------------------------------------//
#pragma once

#include "VecGeom/base/Assert.h"
#include "VecGeom/base/Math.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/navigation/BVHNavigator.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#if !VECGEOM_DEVICE_COMPILE
#include <ostream>
#endif

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//

//! Enable this function on CPU and GPU
#define VECGEOM_FUNCTION VECCORE_ATT_HOST_DEVICE
//! Throw an error for functions not implemented
#define VECGEOM_NOT_IMPLEMENTED(WHAT) VECGEOM_RUNTIME_THROW("not implemented", WHAT, nullptr)

// Define numeric constants for non-preprocessor use
#define VECEGOM_NAV_INDEX 1
#define VECEGOM_NAV_TUPLE 2

#if defined(VECGEOM_USE_NAVINDEX)
#define VECEGOM_NAV VECEGOM_NAV_INDEX
#elif defined(VECGEOM_USE_NAVTUPLE)
#define VECEGOM_NAV VECEGOM_NAV_TUPLE
#else
#define VECEGOM_NAV 0
#endif

//! When switching to production use, these will be moved to a wrapper class
#define VECGEOM_NAV_VALIDATE(COND, MSG) VECGEOM_VALIDATE(COND, MSG)

//---------------------------------------------------------------------------//

namespace vecgeom {

#if VECEGOM_NAV == VECEGOM_NAV_INDEX
using OpaqueNavPath = ::vecgeom::NavIndex_t;
#elif VECEGOM_NAV == VECEGOM_NAV_TUPLE
using OpaqueNavPath = ::vecgeom::NavTuple_t;
#endif

//! Tag struct to indicate point-in-volume location is to be performed
struct UnknownPath {};

//! Tag instance to imply "locate within world volume"
constexpr inline UnknownPath unknown_path{};

//! Tag struct to indicate an exact path is known
struct FullPath {
  OpaqueNavPath path{};
  bool boundary{false};
};

/*!
 * Classification of a boundary search result.
 */
enum class NavFindResultKind {
  hit,       //!< intersection found
  miss,      //!< no intersection found within specified search distance
  reentrant, //!< on a boundary headed into it: zero distance to cross
  error,     //!< overlap detected
  size_      // (enum size sentinel)
};

#if !VECGEOM_DEVICE_COMPILE
inline char const *ToString(NavFindResultKind k)
{
  // TODO: use corecel::EnumToString
  constexpr auto size                   = static_cast<int>(NavFindResultKind::size_);
  static char const *const values[size] = {"hit", "miss", "reentrant", "error"};
  return values[static_cast<int>(k)];
}

//! Print the result for debugging and testing
inline std::ostream &operator<<(std::ostream &os, NavFindResultKind k) { return (os << ToString(k)); }
#endif

/*!
 * Boundary search result.
 *
 * This can either be a "hit" with a strictly positive distance, or a non-hit result (see \c NavFindResultKind ).
 */
class NavFindResult {
public:
  using Kind = NavFindResultKind;
  using Real = ::vecgeom::Precision;

  //! Default is to miss
  VECGEOM_FUNCTION NavFindResult() : kind_{Kind::miss} {}

  //! Initialize with error or 'miss' condition
  explicit VECGEOM_FUNCTION NavFindResult(Kind k) : kind_{k} { VECGEOM_ASSERT(kind_ != Kind::hit); }

  //! Initialize with *positive* distance to hit
  explicit VECGEOM_FUNCTION NavFindResult(Real distance) : kind_{Kind::hit}, distance_{distance}
  {
    VECGEOM_ASSERT(distance_ > 0);
  }

  //! Get the category of search result: hit, miss, reentrant, error
  VECGEOM_FUNCTION Kind GetKind() const { return kind_; }

  //! Get the distance to a hit: can be called only when kind == hit
  VECGEOM_FUNCTION Real GetDistance() const
  {
    VECGEOM_ASSERT(kind_ == Kind::hit);
    return distance_;
  }

#if !VECGEOM_DEVICE_COMPILE
  //! Print the result for debugging and testing
  friend std::ostream &operator<<(std::ostream &os, NavFindResult const &r)
  {
    os << r.GetKind();
    if (r.GetKind() == Kind::hit) {
      os << '@' << r.GetDistance();
    }
    return os;
  }
#endif

private:
  Kind kind_{};
  Real distance_{};
};

/*!
 * Low-level VecGeom navigation interface for a single thread.
 *
 * \c NavView is an ephemeral wrapper around externally owned VecGeom
 * navigation state.  It provides direct access to VecGeom navigator
 * primitives regardless of storage, track-slot management, execution architecture, or implementation navigator/state.
 *
 * The view holds non-owning references to:
 *  - Current \c VgNavState: encodes the current + previous navigation path
 *    and surface-crossing flag.
 *  - Next \c VgNavState: scratch buffer used for half-relocation.
 *  - Global position as a \c Span<Real,3>.
 *  - Global direction as a \c Span<Real,3>.
 *
 * The caller is responsible for keeping track of the on-boundary flag and the
 * straight-line distance between operations (see https://codimd.web.cern.ch/s/xZ5V-Mryl ).
 *
 * The \c OpaquePath is a low-level, POD-compatible representation of a navigation path that is selected at
 * build time:
 *  - \c VgNavIndex   when \c VECGEOM_USE_NAVINDEX is set.
 *  - \c NavTuple     when \c VECGEOM_USE_NAVTUPLE  is set.
 * G4VG converts between Geant4 touchable paths and \c OpaquePath.
 *
 * Initialization of a state is done in one of two ways:
 * - Directly from a full navigation path. This use case is for converting directly from a Geant4 "touchable history".
 * - By performing a full point-in-volume search based on a starting position. This is needed to initialize "primary"
 *   particles which have no geometric metadata.
 *

 * Typical straight-tracking usage:
 * \code
 *   NavView view(cur_state, nxt_state, pos_span, dir_span);
 *   view.Initialize(unknown_path, pos_span, dir_span);
 *   NavView::FindResult prop = view.FindNextBoundary(max_dist);
 *   if (prop.boundary) {
 *     Real3 norm = view.CalcNormal();   // cache before crossing if needed
 *     view.MoveToBoundary(prop.distance);
 *     view.CrossBoundary();
 *   } else {
 *     view.MoveInternal(prop.distance);
 *   }
 * \endcode
 *
 * \note This is based on a collaborative document at https://codimd.web.cern.ch/s/xZ5V-Mryl that establishes a testable contract between VecGeom and other codes.
 *
 * \note All points are in the "global" (world) coordinate system unless otherwise specified.
 */
class NavView {
public:
  //!@{
  //! \name Type aliases
  using Real            = ::vecgeom::Precision;
  using Real3           = ::vecgeom::Vector3D<Real>;
  using SpanReal3       = ::vecgeom::Vector3D<Real> &;
  using SpanConstReal3  = ::vecgeom::Vector3D<Real> const &;
  using NavState        = ::vecgeom::NavigationState;
  using LogicalVolumeId = unsigned int;
  using PlacedVolumeId  = unsigned int;
  using OpaquePath      = OpaqueNavPath;
  using FindResult      = NavFindResult;
  //!@}

  // Construct a view to externally owned navigation state and position/direction
  inline VECGEOM_FUNCTION NavView(NavState &cur_state, NavState &nxt_state, SpanReal3 pos, SpanReal3 dir);

  //// STATE-INDEPENDENT FUNCTIONS ////

  //! Opaque path indicating "outside"
  static VECGEOM_FUNCTION OpaquePath GetOutsidePath() { return NavIndex_t{0}; }

  //! Opaque path indicating "world"
  static VECGEOM_FUNCTION OpaquePath GetWorldPath() { return NavIndex_t{1}; }

  //// INITIALIZATION ////

  // Initialize navigation from an existing opaque path and boundary state
  inline VECGEOM_FUNCTION void Initialize(FullPath fp, SpanConstReal3 pos, SpanConstReal3 dir);

  // Initialize navigation from position and direction within the world
  inline VECGEOM_FUNCTION void Initialize(UnknownPath, SpanConstReal3 pos, SpanConstReal3 dir);

  //// OPERATIONS ////

  // Compute a conservative isotropic distance to the nearest boundary
  inline VECGEOM_FUNCTION Real FindSafety(Real max);

  // Find the straight-line distance to the next boundary
  inline VECGEOM_FUNCTION FindResult FindNextBoundary(Real max);

  // Change the current track direction
  inline VECGEOM_FUNCTION void ChangeDirection(SpanConstReal3 dir);

  // Move to an arbitrary position within the current safety sphere
  inline VECGEOM_FUNCTION void MoveInternal(SpanConstReal3 pos);

  // Move along the current direction without reaching a boundary
  inline VECGEOM_FUNCTION void MoveInternal(Real step);

  // Advance to the boundary at the exact distance returned b
  inline VECGEOM_FUNCTION void MoveToBoundary(Real step);

  // Compute the surface normal at the current boundary
  inline VECGEOM_FUNCTION Real3 CalcNormal() const;

  // Cross the current boundary and update the navigation state
  inline VECGEOM_FUNCTION void CrossBoundary();

  //// ACCESSORS ////

  // Whether the navigation state is outside the world or uninitialized
  inline VECGEOM_FUNCTION bool IsOutside() const;

  // Whether the navigation state is on the boundary of a volume
  inline VECGEOM_FUNCTION bool IsOnBoundary() const;

  // VecGeom logical volume ID of the current volume (NOT outside)
  inline VECGEOM_FUNCTION LogicalVolumeId GetLogicalVolumeId() const;

  // VecGeom placed volume ID of the current volume instance (NOT outside)
  inline VECGEOM_FUNCTION PlacedVolumeId GetPlacedVolumeId() const;

  // Access the current navigation path as an opaque value
  inline VECGEOM_FUNCTION OpaquePath const &GetOpaquePath() const;

  //! Current global position.
  VECGEOM_FUNCTION SpanConstReal3 GetPosition() const { return pos_; }

  //! Current global direction (unit vector).
  VECGEOM_FUNCTION SpanConstReal3 GetDirection() const { return dir_; }

private:
  //// TYPES ////

  using Navigator = ::vecgeom::BVHNavigator;

  //// DATA ////

  NavState &cur_state_; //!< Current navigation path
  NavState &nxt_state_; //!< Scratch "next" navigation path
  SpanReal3 pos_;       //!< Global position [cm]
  SpanReal3 dir_;       //!< Global direction (unit vector)

  //// HELPER FUNCTIONS ////

  // Move a straight-line distance without error checking
  inline VECGEOM_FUNCTION void MoveInternalImpl(Real distance);

  //! Copy data from a const span to a span: this will change based on the definition of Span3
  static VECGEOM_FUNCTION void CopyFrom(SpanReal3 dst, SpanConstReal3 src) { dst = src; }

  // Validate normalized directions: will be replaced by corecel::ArraySoftUnit
  inline static VECGEOM_FUNCTION bool IsUnitNormal(SpanConstReal3 v);
};

//---------------------------------------------------------------------------//
// CONSTRUCTOR
//---------------------------------------------------------------------------//

/*!
 * Construct from externally owned navigation state and position/direction.
 *
 * All arguments are borrowed references; the caller must ensure they
 * outlive this view.
 *
 * \param cur_state  Current navigation state (encodes path + boundary
 *                   flag).
 * \param nxt_state  Scratch navigation state used as a "next" buffer for
 *                   half-relocation.
 * \param pos        Mutable pointer to the global position
 * \param dir        Mutable pointer to the global direction
 */
VECGEOM_FUNCTION NavView::NavView(NavState &cur_state, NavState &nxt_state, SpanReal3 pos, SpanReal3 dir)
    : cur_state_{cur_state}, nxt_state_{nxt_state}, pos_{pos}, dir_{dir}
{
  VECGEOM_NAV_VALIDATE(&cur_state_ != &nxt_state_, << "current and next must point to different objects");
}

//---------------------------------------------------------------------------//
// INITIALIZATION
//---------------------------------------------------------------------------//

/*!
 * Initialize navigation from an existing opaque path (CPU only).
 *
 * Intended for ADePT-style use where a Geant4 touchable path has already
 * been converted to an \c OpaquePath by G4VG.
 *
 * \pre \c pos must lie inside or on the boundary of the volume encoded by
 *      \c path, and must not belong to a child volume of it.
 * \pre \c dir must be normalised to unit length within machine precision.
 */
VECGEOM_FUNCTION void NavView::Initialize(FullPath fp, SpanConstReal3 pos, SpanConstReal3 dir)
{
  VECGEOM_NAV_VALIDATE(IsUnitNormal(dir), << "direction " << dir << " is not a unit vector");

  // Copy position and direction into the externally owned storage
  CopyFrom(pos_, pos);
  CopyFrom(dir_, dir);

  // Clear current and next states
  cur_state_.Clear();
  nxt_state_.Clear();

  // NOTE: SetNavIndex can actually take a tuple when VECGEOM_USE_NAVTUPLE
  cur_state_.SetNavIndex(fp.path);
  cur_state_.SetBoundaryState(fp.boundary);
}

/*!
 * Initialize navigation by searching for the given position and direction.
 *
 * Performs a full geometry location to find the enclosing volume.  Works
 * on both host and device, and is the preferred initialisation path for
 * generating primary particles from spatial distributions (e.g., uniform
 * box) on device. It is allowed for the position to be outside or on the
 * boundary of the world.
 *
 * \pre \c dir must be normalised to unit length within machine precision.
 */
VECGEOM_FUNCTION void NavView::Initialize(UnknownPath, SpanConstReal3 pos, SpanConstReal3 dir)
{
  VECGEOM_NAV_VALIDATE(IsUnitNormal(dir), << "direction " << dir << " is not a unit vector");
  // Copy position and direction into the externally owned storage
  CopyFrom(pos_, pos);
  CopyFrom(dir_, dir);

  // Clear current and nexdt states
  cur_state_.Clear();
  nxt_state_.Clear();

#if 0
  // TODO: when adding surface support (BVHSurfNavigator)
  // Surface navigator identifies worlds by integer ID
  PlacedVolumeId world = vecgeom::NavigationState::WorldId();
#else
  auto const *world = vecgeom::GeoManager::Instance().GetWorld();
#endif
  constexpr bool check_top = true;

  auto *top_pv = Navigator::LocatePointIn(world, pos_, cur_state_, check_top);
  VECGEOM_NAV_VALIDATE((top_pv == nullptr) == this->IsOutside(), << "inconsistent initialization result");
}

//---------------------------------------------------------------------------//
// OPERATIONS
//---------------------------------------------------------------------------//

/*!
 * Compute a conservative isotropic distance to the nearest boundary.
 *
 * Returns a safety radius \em s satisfying the following constraints for
 * the maximum input distance \em d and the true inscribed-sphere radius
 * \em t:
 *  - \f$ s > 0 \f$
 *  - \f$ s \le d \f$.
 *  - \f$ s \le t \f$ (conservative upper bound on the inscribed sphere).
 *  - Ideally \f$ s \ge C\,t \f$ for a geometry-independent constant \em C
 *    (bounded relative error), though this cannot always be guaranteed.
 *
 * \pre The track is not on a boundary.
 * \pre \c max is strictly positive.
 *
 * \param max Maximum search distance.
 * \return Conservative safety radius.
 */
VECGEOM_FUNCTION auto NavView::FindSafety(Real max_dist) -> Real
{
  VECGEOM_NAV_VALIDATE(!this->IsOnBoundary() && !this->IsOutside(),
                       << "cannot search for safety when on a boundary or outside");
  VECGEOM_NAV_VALIDATE(max_dist > 0, << "safety max " << max_dist << " is nonpositive");
  using std::fmax;

  Real result = Navigator::ComputeSafety(pos_, cur_state_, max_dist);
  // TODO: some shapes return a limit greater than max_dist; fix for each one
  result = fmin(max_dist, result);
  // TODO: when on the wrong side of the boundary, a *negative* distance is returned
  result = fmax(0, result);

  VECGEOM_NAV_VALIDATE(0 <= result && result <= max_dist,
                       << "returned safety " << result << " is out of bounds (0, " << max_dist << "]");
  return result;
}

/*!
 * Find the straight-line distance to the next boundary.
 *
 * The returned \c NavFindResult encodes:
 *  - \c boundary=true, \c distance>0: an intersection was found at the
 *    returned distance (\em intersection case).  The next state is cached
 *    internally; call \c MoveToBoundary then \c CrossBoundary.
 *  - \c boundary=false, \c distance=max: no boundary within \c max_dist
 *    (\em no-intersection case).  Call \c MoveInternal.
 *  - \c NavFindResult::IsReentrant : track is on a boundary headed
 *    outward (\em reentrant case).  Typically handled by the caller as a
 *    degenerate step.
 *  - \c NavFindResult::IsError Negative solid distance signals a geometry
 *    overlap (\em error case).
 *
 * \pre The track position is inside the current volume or on a boundary.
 * \pre \c max_dist is strictly positive.
 *
 * \param max_dist Maximum search distance.
 * \return \c FindResult containing distance and boundary flag.
 *
 * \post Internal next state is cached (current implementation).
 * \post The next distance is \em not cached; the caller must store it if
 *       \c MoveToBoundary will need it.
 */
VECGEOM_FUNCTION auto NavView::FindNextBoundary(Real max_dist) -> FindResult
{
  VECGEOM_NAV_VALIDATE(!this->IsOutside(), << "cannot search while outside");
  VECGEOM_NAV_VALIDATE(max_dist > 0, << "max_dist=" << max_dist << " must be positive");

  // Note: search distance may need to be bumped slightly to capture boundaries at maximum distance
  auto next_distance = Navigator::ComputeStepAndNextVolume(pos_, dir_, max_dist, cur_state_, nxt_state_);
  // Next distance should only be non-positive if a boundary is hit
  VECGEOM_ASSERT(nxt_state_.IsOnBoundary() || next_distance > 0);

  VECGEOM_NAV_VALIDATE(next_distance <= max_dist,
                       << "returned distance " << next_distance << " is beyond maximum " << max_dist);

  if (!nxt_state_.IsOnBoundary())
    return FindResult{NavFindResultKind::miss};
  else if (next_distance > 0)
    return NavFindResult{next_distance}; // Hit
  else if (next_distance == 0)
    return NavFindResult{NavFindResultKind::reentrant}; // Coincident
  else
    return NavFindResult{NavFindResultKind::error}; // Backward or NaN (error)
}

/*!
 * Change the current track direction.
 *
 * May be called while on a boundary (e.g., before EM boundary crossing or
 * after optical reflection).
 *
 * \pre \c dir is normalised to unit length within machine epsilon.
 *
 * \param dir New direction unit vector.
 *
 * \post Any cached next state, next placed-volume, and next surface are
 *       invalidated; \c FindNextBoundary must be called before any move.
 */
VECGEOM_FUNCTION void NavView::ChangeDirection(SpanConstReal3 dir)
{
  VECGEOM_NAV_VALIDATE(IsUnitNormal(dir), << "direction " << dir << " is not a unit vector");
  dir_ = dir;
  // NOTE: this is here for error checking purposes and should be removed for production use
  nxt_state_.Clear();
}

/*!
 * Move to an arbitrary position within the current safety sphere.
 *
 * Used for charged tracks whose position is displaced transversely inside
 * the safety bubble (e.g.\ by magnetic-field or multiple-scattering
 * correction), where the displacement is guaranteed not to cross a
 * boundary.
 *
 * \pre \c pos lies within the current safety sphere and is not on a boundary.
 *
 * \param pos New global position.
 *
 * \post Boundary flag is false.
 */
VECGEOM_FUNCTION void NavView::MoveInternal(SpanConstReal3 pos)
{
  // TODO: check against safety?
  VECGEOM_NAV_VALIDATE(!this->IsOutside(), << "cannot move while outside");
  pos_ = pos;
  cur_state_.SetBoundaryState(false);
  nxt_state_.SetBoundaryState(false);
  // TODO: validate the volume hasn't changed
}

/*!
 * Move along the current direction without reaching a boundary.
 *
 * Used for neutral tracks (photons) and for charged tracks near a
 * boundary where the safety is less than the sagitta or miss distance.
 * Unlike Geant4, no re-location call is needed after this move.
 *
 * \pre \c step is strictly positive.
 * \pre \c step is strictly less than the distance to the next boundary.
 *
 * \param step Distance to advance along the current direction.
 *
 * \post Boundary flag is false.
 */
VECGEOM_FUNCTION void NavView::MoveInternal(Real step)
{
  VECGEOM_NAV_VALIDATE(!this->IsOutside(), << "cannot move while outside");
  this->MoveInternalImpl(step);
  cur_state_.SetBoundaryState(false);
}

/*!
 * Advance to the boundary at the exact distance returned by
 * \c FindNextBoundary.
 *
 * \pre \c FindNextBoundary was called and returned \c boundary=true.
 * \pre \c ChangeDirection has \em not been called since the last
 *      \c FindNextBoundary.
 *
 * \param step Exact distance returned by the preceding \c FindNextBoundary
 *             call.
 *
 * \post The track is on the boundary (distance from true surface less than
 *       a small tolerance, but potentially more than machine epsilon).
 * \post Boundary state is true.
 */
VECGEOM_FUNCTION void NavView::MoveToBoundary(Real step)
{
  VECGEOM_NAV_VALIDATE(nxt_state_.IsOnBoundary(), << "next state is not on a boundary");
  this->MoveInternalImpl(step);
  cur_state_.SetBoundaryState(true);
  // TODO: verify endpoint is on boundary of current or next volume?
}

/*!
 * Compute the surface normal at the current boundary.
 *
 * Calculates the outward-facing normal of the solid being \c exited without
 * requiring extra storage or an expensive search, because both the current
 * and next navigation states are available.  The application must cache
 * the result before calling \c CrossBoundary if the normal is needed
 * afterwards.
 *
 * The sign of the returned vector is \em not guaranteed: it may point out
 * of the current volume or into a daughter, depending on VecGeom's
 * implementation.
 *
 * \pre The track is on a boundary and \c FindNextBoundary has been called.
 * \pre \c MoveToBoundary has been called.
 * \pre \c CrossBoundary has \em not yet been called.
 *
 * \return Surface normal direction (sign unspecified).
 */
VECGEOM_FUNCTION auto NavView::CalcNormal() const -> Real3
{
  VECGEOM_NAV_VALIDATE(nxt_state_.IsOnBoundary(), << "FindNextBoundary not yet called");
  VECGEOM_NAV_VALIDATE(cur_state_.IsOnBoundary(), << "MoveToBoundary not yet called");

  // TODO: need to select current or child volume?
  VECGEOM_NOT_IMPLEMENTED("CalcNormal");
}

/*!
 * Cross the current boundary and update the navigation state.
 *
 * Performs relocation: the "next" half-cooked state becomes the fully
 * correct current state.  The "next" state is then invalidated, so
 * \c FindNextBoundary must be called again before the next
 * \c MoveToBoundary or \c CrossBoundary.
 *
 * \pre The track is on a boundary (i.e.\ \c FindNextBoundary and
 *      \c MoveToBoundary have been called in order).
 *
 * \post The current navigation path reflects the new volume; calls to
 *       \c GetLogicalVolumeId and \c GetPlacedVolumeId will return updated
 *       values.
 * \post The internal next state is invalidated.
 *
 * \todo Should we \em swap current and next state so that we can move
 * back and forth cheaply, and then how would that affect FindNextBoundary?
 * What about a method ReenterBoundary()?
 */
VECGEOM_FUNCTION void NavView::CrossBoundary()
{
  VECGEOM_NAV_VALIDATE(nxt_state_.IsOnBoundary(), << "FindNextBoundary not yet called");
  VECGEOM_NAV_VALIDATE(cur_state_.IsOnBoundary(), << "MoveToBoundary not yet called");

  cur_state_ = nxt_state_;
  if (!cur_state_.IsOutside()) {
    Navigator::RelocateToNextVolume(pos_, dir_, cur_state_);
  }

  nxt_state_.Clear();
}

//---------------------------------------------------------------------------//
// ACCESSORS
//---------------------------------------------------------------------------//

/*!
 * Whether the navigation state is outside.
 *
 * This should be "true" for uninitialized states as well.
 */
VECGEOM_FUNCTION bool NavView::IsOutside() const { return cur_state_.IsOutside(); }

/*!
 * Whether the navigation state is on the boundary of a volume.
 *
 * This should always be \c false if outside.
 */
VECGEOM_FUNCTION bool NavView::IsOnBoundary() const { return cur_state_.IsOnBoundary(); }

/*!
 * VecGeom logical volume ID of the current volume.
 *
 * On host this can be mapped to \c vecgeom::LogicalVolume* and from there
 * to Adept/Celeritas materials, sensitive-detector regions, or Geant4 logical
 * volume pointers.
 *
 * \pre \c IsOutside must be \c false
 */
VECGEOM_FUNCTION auto NavView::GetLogicalVolumeId() const -> LogicalVolumeId
{
  VECGEOM_NAV_VALIDATE(!cur_state_.IsOutside(), << "cannot query ID while outside");
  return cur_state_.GetLogicalId();
}

/*!
 * VecGeom placed volume ID of the current volume instance.
 *
 * On host this can be mapped to \c vecgeom::VPlacedVolume* and from there
 * to a \c G4VPhysicalVolume*.
 */
VECGEOM_FUNCTION auto NavView::GetPlacedVolumeId() const -> PlacedVolumeId
{
  VECGEOM_NAV_VALIDATE(!cur_state_.IsOutside(), << "cannot query ID while outside");
  auto *pv = cur_state_.Top();
  VECGEOM_ASSERT(pv);
  return pv->id();
}

/*!
 * Access the current navigation path as an opaque value.
 *
 * The path encodes the full touchable hierarchy and is only meaningful in
 * the context of the VecGeom geometry from which this view was
 * constructed.
 */
VECGEOM_FUNCTION auto NavView::GetOpaquePath() const -> OpaquePath const & { return cur_state_.GetState(); }

/*!
 * Move a straight-line distance without error checking.
 *
 * This is optimized to avoid aliasing issues and to use fused multiply-add for higher precision.
 */
VECGEOM_FUNCTION void NavView::MoveInternalImpl(Real distance)
{
  using std::fma;
  Real3 newpos;
  for (int i = 0; i < 3; ++i) {
    newpos[i] = fma(distance, dir_[i], pos_[i]);
  }
  pos_ = newpos;
}
/*!
 * Validate directions have unit normal.
 *
 * This will be replaced by corecel::ArraySoftUnit : see
 * https://celeritas-project.github.io/celeritas/user/implementation/corecel/numerics.html#_CPPv4I0EN9celeritas13ArraySoftUnitE
 * .
 */
VECGEOM_FUNCTION bool NavView::IsUnitNormal(SpanConstReal3 v)
{
  using namespace std;
  constexpr Real epsilon{1e-14};
  Real length_sq{0};
  for (int i = 0; i < 3; ++i) {
    length_sq = fma(v[i], v[i], length_sq);
  }
  // Factor of 3 is a loose bound of triangle inequality
  return fabs(length_sq - 1) < 3 * epsilon;
}

//---------------------------------------------------------------------------//
} // namespace vecgeom
