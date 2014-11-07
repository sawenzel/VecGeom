/*
 * boolminustest.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: swenzel
 */

#include "base/Transformation3D.h"
#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Tube.h"
#include "volumes/kernel/shapetypes/TubeTypes.h"
#include "volumes/TBooleanMinusVolume.h"
#include "management/GeoManager.h"
#include "benchmarking/Benchmarker.h"

using namespace vecgeom;


// now create a specialized box by hand (instead of the factory)
// we know that it is only translated
typedef SpecializedBox<translation::kIdentity, rotation::kIdentity>
        OriginBox_t;
typedef SpecializedTube<translation::kGeneric, rotation::kIdentity, TubeTypes::HollowTube>
        TranslatedHollowTube_t;

//typedef TUnplacedBooleanMinusVolume<
//             OriginBox_t, TranslatedBox_t > BoxMinusBox_t;
typedef TUnplacedBooleanMinusVolume BoxMinusBox_t;

typedef TSpecializedBooleanMinusVolume<OriginBox_t, TranslatedHollowTube_t,
        translation::kGeneric, rotation::kIdentity> SpecializedVol_t;

int main()
{
    UnplacedBox worldUnplaced(10.,10.,10.);
    LogicalVolume world = LogicalVolume("world", &worldUnplaced);

    // components for boolean solid
    UnplacedBox motherbox(5.,5.,5.);
    UnplacedTube subtractedtube(0.5,2.,2.,0,2.*M_PI);
    // translation for boolean solid right shape
    Transformation3D translation(-2.5,0,0);



    TranslatedHollowTube_t const * placedsubtractedtube =
            new TranslatedHollowTube_t(new LogicalVolume("",&subtractedtube), &translation);

    OriginBox_t const * placedmotherbox = new OriginBox_t(new LogicalVolume("",&motherbox), &Transformation3D::kIdentity);

    // now make the unplaced boolean solid
    BoxMinusBox_t booleansolid(placedmotherbox, placedsubtractedtube);
    LogicalVolume booleanlogical("booleanL",&booleansolid);
    // placement of boolean solid
    Transformation3D placement(5, 5, 5);

    // make a special solid by hand ( need to sort out factory later )
    SpecializedVol_t * sp = new SpecializedVol_t("booleanspecialized", &booleanlogical, &placement );
    //booleansolid.CreateSpecializedVolume()
    //world.PlaceDaughter("boolean", &booleanlogical, &placement);

    // add this boolean solid to the world
    world.PlaceDaughter( sp );

    VPlacedVolume *worldPlaced = world.Place();
    GeoManager::Instance().SetWorld(worldPlaced);

    Benchmarker tester(GeoManager::Instance().GetWorld());
      tester.SetVerbosity(3);
      tester.SetRepetitions(1024);
      tester.SetPointCount(1024);
      tester.RunBenchmark();
   // tester.RunToOutBenchmark();

    return 0;
}

