/*
 * TestCone.cpp
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#include "volumes/Cone.h"
#include "volumes/LogicalVolume.h"
#include "base/Transformation3D.h"
#include <cstdio>

// to test compilation and functionality of the cone
using namespace vecgeom;

struct SomeCone
{
};

void testInstantiation()
{
    UnplacedCone acone(10,20,15,25,100, 0, 2.*M_PI);

    // some basic tests here to test the interface
    acone.Print();
    double volume = acone.Capacity();
    printf("have volume %lf\n",volume);

    Transformation3D * t = new Transformation3D();
    // test instanteation of a concrete placed specialized shape
    SpecializedCone<translation::kIdentity, rotation::kIdentity, SomeCone> scone(
            new LogicalVolume(&acone), t );

    printf("have volume %lf\n",scone.Capacity());
}

int main()
{
    testInstantiation();
}
