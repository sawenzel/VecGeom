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

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//

//! Enable this function on CPU and GPU
#define VECGEOM_FUNCTION VECCORE_ATT_HOST_DEVICE

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

//! This will allow extra validation to be enabled for testing
#define VECGEOM_NAV_VALIDATE(COND, MSG) VECGEOM_VALIDATE(COND, MSG)

//---------------------------------------------------------------------------//

namespace vecgeom {

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
 * The \c OpaquePath is a low-level, POD-compatible representation of a navigation path that is selected at
 * build time:
 *  - \c VgNavIndex   when \c VECGEOM_USE_NAVINDEX is set.
 *  - \c NavTuple     when \c VECGEOM_USE_NAVTUPLE  is set.
 * G4VG converts between Geant4 touchable paths and \c OpaquePath.
 *
 * \note This is based on a collaborative document at https://codimd.web.cern.ch/s/xZ5V-Mryl that establishes a testable contract between VecGeom and other codes.
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
#if VECEGOM_NAV == VECEGOM_NAV_INDEX
  using OpaquePath = ::vecgeom::NavIndex_t;
#elif VECEGOM_NAV == VECEGOM_NAV_TUPLE
  using OpaquePath = ::vecgeom::NavTuple_t;
#endif
  //!@}

  // Construct a view to externally owned navigation state and position/direction
  inline VECGEOM_FUNCTION NavView(NavState &cur_state, NavState &nxt_state, SpanReal3 pos, SpanReal3 dir);

  //// INITIALIZATION ////

  // Initialize navigation from an existing opaque path
  inline VECGEOM_FUNCTION void Initialize(OpaquePath const &path, SpanConstReal3 pos, SpanConstReal3 dir);

  // Initialize navigation from position and direction alone
  inline VECGEOM_FUNCTION void Initialize(SpanConstReal3 pos, SpanConstReal3 dir);

  //// ACCESSORS ////

  // Whether the navigation state is outside the world or uninitialized
  inline VECGEOM_FUNCTION bool IsOutside() const;

  // VecGeom logical volume ID of the current volume (NOT outside)
  inline VECGEOM_FUNCTION LogicalVolumeId GetLogicalVolumeId() const;

  // VecGeom placed volume ID of the current volume instance (NOT outside)
  inline VECGEOM_FUNCTION PlacedVolumeId GetPlacedVolumeId() const;

  // Access the current navigation path as an opaque value
  inline VECGEOM_FUNCTION OpaquePath const &GetOpaquePath() const;

  //! Current global position.
  VECGEOM_FUNCTION SpanConstReal3 Position() const { return pos_; }

  //! Current global direction (unit vector).
  VECGEOM_FUNCTION SpanConstReal3 Direction() const { return dir_; }

private:
  //// TYPES ////

  using Navigator = ::vecgeom::BVHNavigator;

  //// DATA ////

  NavState &cur_state_; //!< Current navigation path
  NavState &nxt_state_; //!< Scratch "next" navigation path
  SpanReal3 pos_;       //!< Global position [cm]
  SpanReal3 dir_;       //!< Global direction (unit vector)

  //// HELPER FUNCTIONS ////

  //! Copy data from a const span to a span: this will change based on the definition of Span3
  static VECGEOM_FUNCTION void CopyFrom(SpanReal3 dst, SpanConstReal3 src) { dst = src; }
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
 * \pre \p path must be valid and represent a volume inside the world.
 * \pre \p pos must lie inside or on the boundary of the volume encoded by
 *      \p path.
 * \pre \p dir must be normalised to unit length within machine precision.
 *
 * \post Navigation state is fully initialised.
 */
VECGEOM_FUNCTION void NavView::Initialize(OpaquePath const &path, SpanConstReal3 pos, SpanConstReal3 dir)
{
  // NOTE: SetNavIndex can actually take a tuple when VECGEOM_USE_NAVTUPLE
  cur_state_.SetNavIndex(path);
  VECGEOM_VALIDATE(!cur_state_.IsOutside(), << "cannot initialize with outside state");
  nxt_state_.Clear();
  CopyFrom(pos_, pos);
  CopyFrom(dir_, dir);
}

/*!
 * Initialize navigation by searching for the given position and direction.
 *
 * Performs a full geometry location to find the enclosing volume.  Works
 * on both host and device, and is the preferred initialisation path for
 * generating primary particles from spatial distributions (e.g., uniform
 * box) on device.
 *
 * \pre \p dir must be normalised to unit length within machine precision.
 *
 * \post If the point is inside the world volume, then the complete
 * navigation path to the deepest enclosing volume is found.
 */
VECGEOM_FUNCTION void NavView::Initialize(SpanConstReal3 pos, SpanConstReal3 dir)
{
  // Copy position and direction into the externally owned storage
  CopyFrom(pos_, pos);
  CopyFrom(dir_, dir);

  // Set up current state and locate the deepest enclosing volume
  cur_state_.Clear();
#if 0
  // TODO: when adding surface support (BVHSurfNavigator)
  // Surface navigator identifies worlds by integer ID
  PlacedVolumeId world = vecgeom::NavigationState::WorldId();
#else
  auto const *world = vecgeom::GeoManager::Instance().GetWorld();
#endif
  constexpr bool contains_point = true;

  // Clear next state
  nxt_state_.Clear();

  bool result = Navigator::LocatePointIn(world, pos_, cur_state_, contains_point);
  VECGEOM_NAV_VALIDATE(result == !this->IsOutside(), << "inconsistent post-locate state");
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

//---------------------------------------------------------------------------//
} // namespace vecgeom
