/**
 * \author Sandro Wenzel (sandro.wenzel@cern.ch)
 */

// A complex test for a couple of features:
// a) creation of a ROOT geometry
// b) creation of a VecGeom geometry
// c) navigation in VecGeom geometry

#include <iostream>

#include "management/root_manager.h"
#include "volumes/placed_volume.h"
#include "navigation/navigationstate.h"
#include "navigation/simple_navigator.h"

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include <cassert>

using namespace vecgeom;

// creates a four level box detector
// this modifies the global gGeoManager instance ( so no need for any return )
void CreateRootGeom()
{
	double L = 10.;
	double Lz = 10.;
	const double Sqrt2 = sqrt(2.);
	TGeoVolume * world =  ::gGeoManager->MakeBox("worldl",0, L, L, Lz );
	TGeoVolume * boxlevel2 = ::gGeoManager->MakeBox("b2l",0, Sqrt2*L/2./2., Sqrt2*L/2./2., Lz );
	TGeoVolume * boxlevel3 = ::gGeoManager->MakeBox("b3l",0, L/2./2., L/2./2., Lz);
	TGeoVolume * boxlevel1 = ::gGeoManager->MakeBox("b1l",0, L/2., L/2., Lz );

	boxlevel2->AddNode( boxlevel3, 0, new TGeoRotation("rot1",0,0,45));
	boxlevel1->AddNode( boxlevel2, 0, new TGeoRotation("rot2",0,0,-45));
	world->AddNode(boxlevel1, 0, new TGeoTranslation(-L/2.,0,0));
	world->AddNode(boxlevel1, 1, new TGeoTranslation(+L/2.,0,0));
	::gGeoManager->SetTopVolume(world);
	::gGeoManager->CloseGeometry();
}

void test1()
{
	VPlacedVolume const * world = RootManager::Instance().world();
	SimpleNavigator nav;
	NavigationState state(4);
	VPlacedVolume const * vol;

	// point should be in world
	Vector3D<Precision> p1(0, 9* 10/10., 0);

	vol=nav.LocatePoint( world, p1, state, true );
	assert( RootManager::Instance().tgeonode( vol ) == ::gGeoManager->GetTopNode());
	std::cerr << "test1 passed" << std::endl;
}

void test2()
{
	// inside box3 check
	VPlacedVolume const * world = RootManager::Instance().world();
	SimpleNavigator nav;
	NavigationState state(4);
	VPlacedVolume const * vol;

	// point should be in world
	Vector3D<Precision> p1(-5., 0., 0.);

	vol=nav.LocatePoint( world, p1, state, true );
	assert( std::strcmp( RootManager::Instance().tgeonode( vol )->GetName() , "bl3_0" ) );
	std::cerr << "test2 passed" << std::endl;
}



int main()
{
    CreateRootGeom();
	RootManager::Instance().LoadRootGeometry();
    RootManager::Instance().world()->logical_volume()->PrintContent();

    RootManager::Instance().PrintNodeTable();

    test1();
    test2();

    return 0;
}
