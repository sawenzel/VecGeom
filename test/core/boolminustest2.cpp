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
#include "volumes/TBooleanMinusVolume.h"
#include "management/GeoManager.h"
#include "benchmarking/Benchmarker.h"

using namespace vecgeom;


// now create a specialized box by hand (instead of the factory)
// we know that it is only translated
typedef SpecializedBox<translation::kIdentity, rotation::kIdentity> OriginBox_t;
typedef SpecializedBox<translation::kGeneric, rotation::kIdentity> TranslatedBox_t;
//typedef TUnplacedBooleanMinusVolume<
//             OriginBox_t, TranslatedBox_t > BoxMinusBox_t;
typedef TUnplacedBooleanMinusVolume BoxMinusBox_t;

typedef TSpecializedBooleanMinusVolume<OriginBox_t, TranslatedBox_t,
        translation::kGeneric, rotation::kIdentity> SpecializedVol_t;

int main()
{
    UnplacedBox worldUnplaced(10.,10.,10.);
    LogicalVolume world = LogicalVolume("world", &worldUnplaced);

    // components for boolean solid
    UnplacedBox motherbox(5.,5.,5.);
    UnplacedBox subtractedbox(2.,2.,2);
    // translation for boolean solid right shape
    Transformation3D translation(-2.5,0,0);


    VPlacedVolume *worldPlaced = world.Place();
    GeoManager::Instance().SetWorld(worldPlaced);


    // now create a specialized box by hand (instead of the factory)
    // we know that it is only translated
    typedef SpecializedBox<translation::kIdentity, rotation::kIdentity> OriginBox_t;
    typedef SpecializedBox<translation::kGeneric, rotation::kIdentity> TranslatedBox_t;

    TranslatedBox_t const * placedsubtractedbox = new TranslatedBox_t(new LogicalVolume("",&subtractedbox), &translation);

    // now create a specialized box by hand (instead of the factory)
    // we know that it is only translated
    OriginBox_t const * placedmotherbox = new OriginBox_t(new LogicalVolume("",&motherbox), &Transformation3D::kIdentity);

    // now make the unplaced boolean solid
    BoxMinusBox_t booleansolid(placedmotherbox, placedsubtractedbox);
    LogicalVolume booleanlogical("booleanL",&booleansolid);
    // placement of boolean solid
    Transformation3D placement(5, 5, 5);

    // make a special solid by hand ( need to sort out factory later )
    SpecializedVol_t * sp = new SpecializedVol_t("booleanspecialized", &booleanlogical, &placement );
    //booleansolid.CreateSpecializedVolume()
    //world.PlaceDaughter("boolean", &booleanlogical, &placement);

    // add this boolean solid to the world
    world.PlaceDaughter( sp );

    Benchmarker tester(GeoManager::Instance().GetWorld());
    tester.SetVerbosity(3);
    tester.SetPoolMultiplier(1);
    tester.SetRepetitions(1024);
    tester.SetPointCount(1<<10);
    tester.RunInsideBenchmark();
    tester.RunToOutBenchmark();

    return 0;

}

