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
#include "base/rng.h"

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


	// point should be in box3
	Vector3D<Precision> p1(-5., 0., 0.);
	vol=nav.LocatePoint( world, p1, state, true );
	assert( std::strcmp( RootManager::Instance().tgeonode( vol )->GetName() , "b3l_0" ) == 0);

	// point should also be in box3
	Vector3D<Precision> p2(5., 0., 0.);
	state.Clear();
	vol=nav.LocatePoint( world, p2, state, true );
	assert( std::strcmp( RootManager::Instance().tgeonode( vol )->GetName() , "b3l_0" ) == 0 );
	std::cerr << "test2 passed" << std::endl;
}


void test3()
{
	// inside box1 left check
	VPlacedVolume const * world = RootManager::Instance().world();
	SimpleNavigator nav;
	NavigationState state(4);
	VPlacedVolume const * vol;
	Vector3D<Precision> p1(-9/10., 9*5/10., 0.);
	vol=nav.LocatePoint( world, p1, state, true );
	assert( std::strcmp( RootManager::Instance().tgeonode( vol )->GetName() , "b1l_0" ) == 0);
	std::cerr << "test3 passed" << std::endl;
}

void test3_2()
{
	// inside box1 right check
	VPlacedVolume const * world = RootManager::Instance().world();
	SimpleNavigator nav;
	NavigationState state(4);
	VPlacedVolume const * vol;
	Vector3D<Precision> p1(9/10., 9*5/10., 0.);
	vol=nav.LocatePoint( world, p1, state, true );
	assert( std::strcmp( RootManager::Instance().tgeonode( vol )->GetName() , "b1l_1" ) == 0);
	std::cerr << "test3_2 passed" << std::endl;
}

void test4()
{
	// inside box2 check
	VPlacedVolume const * world = RootManager::Instance().world();
	SimpleNavigator nav;
	NavigationState state(4);
	VPlacedVolume const * vol;
	Vector3D<Precision> p1(5., 9*5/10., 0.);
	vol=nav.LocatePoint( world, p1, state, true );
	assert( std::strcmp( RootManager::Instance().tgeonode( vol )->GetName() , "b2l_0" ) == 0);
	std::cerr << "test4 passed" << std::endl;
}

void test5()
{
	// outside world check
	VPlacedVolume const * world = RootManager::Instance().world();
	SimpleNavigator nav;
	NavigationState state(4);
	VPlacedVolume const * vol;
	Vector3D<Precision> p1(-20, 0., 0.);
	vol=nav.LocatePoint( world, p1, state, true );
	assert( vol == 0 );
	std::cerr << "test5 passed" << std::endl;
}

void test6()
{
	// statistical test  - comparing with ROOT navigation functionality
	// generate points
	for(int i=0;i<100000;++i)
	{
		double x = RNG::Instance().uniform(-10,10);
		double y = RNG::Instance().uniform(-10,10);
		double z = RNG::Instance().uniform(-10,10);

		// ROOT navigation
		TGeoNavigator  *nav = ::gGeoManager->GetCurrentNavigator();
		TGeoNode * node =nav->FindNode(x,y,z);

		// VecGeom navigation
		NavigationState state(4);
		SimpleNavigator vecnav;
		state.Clear();
		VPlacedVolume const *vol= vecnav.LocatePoint( RootManager::Instance().world(),
				Vector3D<Precision>(x,y,z) , state, true);

		assert( RootManager::Instance().tgeonode(vol) == node );
	}
	std::cerr << "test6 (statistical) passed" << std::endl;
}

// relocation test
void test7()
{
	// statistical test  - testing global matrix transforms and relocation at the same time
	// consistency check with full LocatePoint method
	// generate points
	for(int i=0;i<100000;++i)
	{
		std::cerr << "###############" << std::endl;
		double x = RNG::Instance().uniform(-10,10);
		double y = RNG::Instance().uniform(-10,10);
		double z = RNG::Instance().uniform(-10,10);

		// ROOT navigation
		TGeoNavigator  *nav = ::gGeoManager->GetCurrentNavigator();
		TGeoNode * node =nav->FindNode(x,y,z);

		// VecGeom navigation
		Vector3D<Precision> p(x,y,z);
		std::cerr << "GLOBAL " << p << std::endl;
		NavigationState state(4);
		SimpleNavigator vecnav;
		VPlacedVolume const *vol1= vecnav.LocatePoint( RootManager::Instance().world(),
				p , state, true);
		std::cerr << RootManager::Instance().tgeonode( vol1 )->GetName() << std::endl;
		state.Print();

		// now we move global point in x direction and find new volume and path
		NavigationState state2(4);
		p+=Vector3D<Precision>(1.,0,0);
		std::cerr << "NEW GLOBAL " << p << std::endl;
		VPlacedVolume const *vol2= vecnav.LocatePoint( RootManager::Instance().world(),
					p , state2, true);
		std::cerr << RootManager::Instance().tgeonode( vol2 )->GetName() << std::endl;

		// same with relocation
		// need local point first
		TransformationMatrix globalm = state.TopMatrix();
		std::cerr << globalm << std::endl;
		Vector3D<Precision> localp;
		globalm.Transform<1,0>( p, localp );
		std::cerr << localp << " "  << std::endl;

		VPlacedVolume const *vol3= vecnav.RelocatePointFromPath( localp, state );
		std::cerr << vol1 << " " << vol2 << " " << vol3 << std::endl;
		assert( vol3  == vol2 );
	}
	std::cerr << "test7 (statistical relocation) passed" << std::endl;
}

int main()
{
    CreateRootGeom();
	RootManager::Instance().LoadRootGeometry();
    RootManager::Instance().world()->logical_volume()->PrintContent();

    RootManager::Instance().PrintNodeTable();

    test1();
    test2();
    test3();
    test3_2();
    test4();
    test5();
    test6();
    test7();

    return 0;
}
