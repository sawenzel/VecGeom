/**
 * @file placed_trap.cpp
 * @author Guilherme Lima (Guilherme.Lima@cern.ch)
 *
 * 140407 G.Lima - based on equivalent Geant4 code, while adapting to VecGeom interface
 */

#include "volumes/placed_trap.h"
#include "backend/implementation.h"
#include "base/aos3d.h"
#include "base/soa3d.h"
#ifdef VECGEOM_ROOT
#include "TGeoArb8.h"
#endif
#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#endif

//#include <stdio.h>

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void PlacedTrap::PrintType() const {
  printf("PlacedTrap");
}

void PlacedTrap::Inside(SOA3D<Precision> const &points, bool *const output) const {
	Inside_Looper<translation::kGeneric, rotation::kGeneric>(*this, points, output);
}

void PlacedTrap::Inside(AOS3D<Precision> const &points, bool *const output) const {
    Inside_Looper<translation::kGeneric, rotation::kGeneric>(*this, points, output);
}


// call the looper pattern which calls the appropriate shape methods

void PlacedTrap::DistanceToIn(SOA3D<Precision> const &positions,
                             SOA3D<Precision> const &directions,
                             Precision const *const step_max,
                             Precision *const output) const {
    DistanceToIn_Looper<translation::kGeneric, rotation::kGeneric>( *this, positions, directions, step_max, output );
}

void PlacedTrap::DistanceToIn(AOS3D<Precision> const &positions,
                             AOS3D<Precision> const &directions,
                             Precision const *const step_max,
                             Precision *const output) const { 
    DistanceToIn_Looper<translation::kGeneric, rotation::kGeneric>( *this, positions, directions, step_max, output );
}


void PlacedTrap::DistanceToOut( AOS3D<Precision> const &position,
				AOS3D<Precision> const &direction,
				Precision const * const step_max,
				Precision *const distances ) const {
   // call the looper pattern which calls the appropriate shape methods
   DistanceToOut_Looper(*this, position, direction, step_max, distances);
}


void PlacedTrap::DistanceToOut( SOA3D<Precision> const &position,
							   SOA3D<Precision> const &direction,
                               Precision const * const step_max,
                               Precision *const distances) const {
   // call the looper pattern which calls the appropriate shape methods
   DistanceToOut_Looper(*this, position, direction, step_max, distances);
}


void PlacedTrap::SafetyToIn( SOA3D<Precision> const &position,
                             Precision *const safeties ) const {
    SafetyToIn_Looper<translation::kGeneric, rotation::kGeneric>(*this, position, safeties);
}


void PlacedTrap::SafetyToIn( AOS3D<Precision> const &position,
                             Precision *const safeties ) const {
    SafetyToIn_Looper<translation::kGeneric, rotation::kGeneric>(*this, position, safeties);
}


void PlacedTrap::SafetyToOut( SOA3D<Precision> const &position,
                              Precision *const safeties ) const {
    SafetyToOut_Looper(*this, position, safeties);
}

void PlacedTrap::SafetyToOut( AOS3D<Precision> const &position,
                              Precision *const safeties ) const {
    SafetyToOut_Looper(*this, position, safeties);
}


#ifdef VECGEOM_CUDA

namespace {

__global__
void ConstructOnGpu(LogicalVolume const *const logical_volume,
                    TransformationMatrix const *const matrix,
                    VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) PlacedTrap(logical_volume, matrix);
}

} // End anonymous namespace

VPlacedVolume* PlacedTrap::CopyToGpu(LogicalVolume const *const logical_volume,
				     TransformationMatrix const *const matrix,
				     VPlacedVolume *const gpu_ptr) const {
  ConstructOnGpu<<<1, 1>>>(logical_volume, matrix, gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedTrap::CopyToGpu(LogicalVolume const *const logical_volume,
				     TransformationMatrix const *const matrix) const {
  VPlacedVolume *const gpu_ptr = AllocateOnGpu<PlacedTrap>();
  return CopyToGpu(logical_volume, matrix, gpu_ptr);
}

#endif // VECGEOM_CUDA

VPlacedVolume const* PlacedTrap::ConvertToUnspecialized() const {
  return new PlacedTrap(logical_volume_, matrix_);
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTrap::ConvertToRoot() const {
    Precision radToDeg = 57.29577951308232; // 180.0 / PI
	auto tanThetaSphi = unplaced_trap()->getTthetaSphi();
	auto tanThetaCphi = unplaced_trap()->getTthetaCphi();
	// theta is necessarily in the range 0..90deg, so tanTheta>=0
	Precision theta = radToDeg * atan( sqrt(tanThetaSphi*tanThetaSphi + tanThetaCphi*tanThetaCphi) );
	Precision phi = radToDeg * atan2( tanThetaSphi, tanThetaCphi );
	if(phi<0) phi += 360.0;

	// ROOT expects angles in DEG
	return new TGeoTrap("", getDz(), theta, phi,
						getDy1(), getDx1(), getDx2(), radToDeg*atan(getTalpha1()),
						getDy2(), getDx3(), getDx4(), radToDeg*atan(getTalpha2()) );
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTrap::ConvertToUSolids() const {
	auto tanThetaSphi = unplaced_trap()->getTthetaSphi();
	auto tanThetaCphi = unplaced_trap()->getTthetaCphi();
	// theta is necessarily in the range 0..90deg, so tanTheta>=0
	Precision theta = atan( sqrt(tanThetaSphi*tanThetaSphi + tanThetaCphi*tanThetaCphi) );
	Precision phi   = atan2( tanThetaSphi, tanThetaCphi );
	if(phi<0) phi += 360.0;

	// USolids package expects angles in RAD
	return new UTrap("", getDz(), theta, phi,
					 getDy1(), getDx1(), getDx2(), atan(getTalpha1()),
					 getDy2(), getDx3(), getDx4(), atan(getTalpha2()) );
}
#endif

} // End global namespace
