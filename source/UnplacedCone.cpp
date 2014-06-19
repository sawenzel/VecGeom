/*
 * UnplacedCone.cpp
 *
 *  Created on: Jun 18, 2014
 *      Author: swenzel
 */


#include "volumes/UnplacedCone.h"
#include "volumes/SpecializedCone.h"
#include "volumes/utilities/GenerationUtilities.h"

#include "management/VolumeFactory.h"

namespace VECGEOM_NAMESPACE {

    void UnplacedCone::Print() const {
     printf("UnplacedCone {rmin1 %.2f, rmax1 %.2f, rmin2 %.2f, "
          "rmax2 %.2f, phistart %.2f, deltaphi %.2f}",
             fRmin1, fRmax2, fRmin2, fRmax2, fSPhi, fDPhi);
    }

    void UnplacedCone::Print(std::ostream &os) const {
        os << "UnplacedCone; please implement Print to outstream\n";
    }

    // what else to implement ??


}

