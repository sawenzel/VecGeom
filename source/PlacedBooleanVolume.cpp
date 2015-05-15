/*
 * PlacedBooleanVolume.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: swenzel
 */

#include "volumes/PlacedBooleanVolume.h"
#include "volumes/SpecializedBooleanVolume.h"
#include "volumes/UnplacedBooleanVolume.h"
#include "volumes/LogicalVolume.h"
#include "base/Vector3D.h"
#include "base/RNG.h"

#include <iostream>

namespace vecgeom {


Vector3D<Precision> PlacedBooleanVolume::GetPointOnSurface() const {
    // implementation taken from G4
    int counter=0;
    Vector3D<Precision> p;

    double leftarea = const_cast<VPlacedVolume *> (GetUnplacedVolume()->fLeftVolume)->SurfaceArea();
    double rightarea = const_cast<VPlacedVolume *> (GetUnplacedVolume()->fRightVolume)->SurfaceArea();
    double arearatio = leftarea/( leftarea + rightarea);

    do{
       counter++;
       if( counter > 1000 ){
           std::cerr << "WARNING : COULD NOT GENERATE POINT ON SURFACE FOR BOOLEAN\n";
           return p;
       }

       
       UnplacedBooleanVolume *unplaced = (UnplacedBooleanVolume*)GetUnplacedVolume();
       if( RNG::Instance().uniform() < arearatio ){
          p = ((UnplacedBooleanVolume *)unplaced->fLeftVolume)->GetPointOnSurface();
       }
       else {
          p = ((UnplacedBooleanVolume *)unplaced->fRightVolume)->GetPointOnSurface();
       }
    } while( Inside(p) != vecgeom::kSurface );
    return p;
}

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN( SpecializedBooleanVolume, kUnion)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN( SpecializedBooleanVolume, kIntersection)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN( SpecializedBooleanVolume, kSubtraction)

#endif // VECGEOM_NVCC

} // End namespace vecgeom
