/// @file GenericPolyconeSection.h
/// @brief Generic polycone z-section storage.
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_GENERICPOLYCONESECTION_H_
#define VECGEOM_GENERICPOLYCONESECTION_H_

#include "VecGeom/volumes/ConeStruct.h"
#include "VecGeom/volumes/CoaxialConesStruct.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct GenericPolyconeSection;);
VECGEOM_DEVICE_DECLARE_CONV(struct, GenericPolyconeSection);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// @brief One z section of a generic polycone.
/// @details Generic polycones are decomposed into adjacent z intervals. Each
/// section stores a shifted coaxial-cones helper that owns the radial contours
/// in that interval. Navigation kernels transform polycone-local points by
/// `fShift` before forwarding to this helper.
struct GenericPolyconeSection {
  /// @brief Construct an empty section.
  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeSection() : fCoaxialCones(0), fShift(0.0), fTubular(0), fConvex(0) {}

  /// @brief Destroy a section wrapper.
  VECCORE_ATT_HOST_DEVICE
  ~GenericPolyconeSection() {}

  /// Coaxial-cones helper describing the section-local radial contours.
  CoaxialConesStruct<Precision> *fCoaxialCones;

  /// Z shift from polycone coordinates to the section-local frame.
  double fShift;

  /// Whether the section is tubular.
  bool fTubular;

  /// Whether the section is convex with respect to the whole polycone.
  bool fConvex;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
