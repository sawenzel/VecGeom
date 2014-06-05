/**
 * \author Sandro Wenzel (sandro.wenzel@cern.ch)
 */

// A complex test for a couple of features:
// a) creation of a ROOT geometry
// b) creation of a VecGeom geometry
// c) navigation in VecGeom geometry

#include <iostream>

#include "base/SOA3D.h"
#include "management/RootManager.h"
#include "volumes/PlacedVolume.h"
#include "navigation/NavigationState.h"
#include "navigation/SimpleNavigator.h"
#include "base/RNG.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "base/Stopwatch.h"

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include <cassert>

using namespace VECGEOM_NAMESPACE;

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
	std::cerr << "test6 (statistical location) passed" << std::endl;
}

// relocation test
void test7()
{
	// statistical test  - testing global matrix transforms and relocation at the same time
	// consistency check with full LocatePoint method
	// generate points
	for(int i=0;i<1000000;++i)
	{
		double x = RNG::Instance().uniform(-10,10);
		double y = RNG::Instance().uniform(-10,10);
		double z = RNG::Instance().uniform(-10,10);

		// VecGeom navigation
		Vector3D<Precision> p(x,y,z);
		NavigationState state(4);
		SimpleNavigator vecnav;
		VPlacedVolume const *vol1 =
		    vecnav.LocatePoint(RootManager::Instance().world(), p, state, true);
		if(vol1) ; // avoid compilation warning

		/*
		if ( vol1 != NULL )
		{
			std::cerr << RootManager::Instance().tgeonode( vol1 )->GetName() << std::endl;
		}
*/

		// now we move global point in x direction and find new volume and path
		NavigationState state2(4);
		p+=Vector3D<Precision>(1.,0,0);
		VPlacedVolume const *vol2= vecnav.LocatePoint( RootManager::Instance().world(),
					p , state2, true);
	/*
		if ( vol2 != NULL )
		{
			std::cerr << "new node " << RootManager::Instance().tgeonode( vol2 )->GetName() << std::endl;

			// ROOT navigation
			TGeoNavigator  *nav = ::gGeoManager->GetCurrentNavigator();
			TGeoNode * node =nav->FindNode(p[0],p[1],p[2]);
			std::cerr << "ROOT new: " << node->GetName() << std::endl;
		}*/

		// same with relocation
		// need local point first
		Transformation3D globalm = state.TopMatrix();
		Vector3D<Precision> localp = globalm.Transform<1,0>( p );

		VPlacedVolume const *vol3= vecnav.RelocatePointFromPath( localp, state );
//		std::cerr << vol1 << " " << vol2 << " " << vol3 << std::endl;
		assert( vol3  == vol2 );
	}
	std::cerr << "test7 (statistical relocation) passed" << std::endl;
}

Vector3D<Precision> sampleDir()
{
	Vector3D<Precision> tmp;
    tmp[0]=RNG::Instance().uniform(-1,1);
    tmp[1]=RNG::Instance().uniform(-1,1);
    tmp[2]=RNG::Instance().uniform(-1,1);
    double inversenorm=1./sqrt( tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]);
    tmp[0]*=inversenorm;
    tmp[1]*=inversenorm;
    tmp[2]*=inversenorm;
    return tmp;
}

// navigation
void test8()
{
	// statistical test  of navigation via comparison with ROOT navigation
	for(int i=0;i<1000000;++i)
	{
		//std::cerr << "START ITERATION " << i << std::endl;
		double x = RNG::Instance().uniform(-10,10);
		double y = RNG::Instance().uniform(-10,10);
		double z = RNG::Instance().uniform(-10,10);

		// VecGeom navigation
		Vector3D<Precision> p(x,y,z);
		Vector3D<Precision> d=sampleDir();

		NavigationState state(4), newstate(4);
		SimpleNavigator nav;
		VPlacedVolume const *vol1= nav.LocatePoint( RootManager::Instance().world(),
				p, state, true);
		if(vol1) ; // avoid compilation warning

		double step;
		nav.FindNextBoundaryAndStep( p, d, state, newstate, 1E30, step );

		TGeoNavigator * rootnav = ::gGeoManager->GetCurrentNavigator();

		// this is one of the disatvantages of the ROOT interface:
		// who enforces that I have to call FindNode first before FindNextBoundary
		// this is not apparent from the interface ?
		TGeoNode * node = rootnav->FindNode(x,y,z);
		assert( rootnav->GetCurrentNode()  == RootManager::Instance().tgeonode( state.Top() ) );

		rootnav->SetCurrentPoint(x,y,z);
		rootnav->SetCurrentDirection(d[0],d[1],d[2]);
		rootnav->FindNextBoundaryAndStep( 1E30 );

		if( newstate.Top() != NULL )
		{
			if( rootnav->GetCurrentNode()  != RootManager::Instance().tgeonode( newstate.Top() ) )
			{
				std::cerr << "ERROR ON ITERATION " << i << std::endl;
				std::cerr << i << " " << d << std::endl;
				std::cerr << i << " " << p << std::endl;
				std::cerr << "I AM HERE: " << node->GetName() << std::endl;
				std::cerr << "ROOT GOES HERE: " << rootnav->GetCurrentNode()->GetName() << std::endl;
				std::cerr << rootnav->GetStep() << std::endl;
				std::cerr << "VECGEOM GOES HERE: " << RootManager::Instance().GetName( newstate.Top() ) << std::endl;

				nav.InspectEnvironmentForPointAndDirection( p, d, state );
			}
		}
		assert( state.Top() != newstate.Top() );
	}
	std::cerr << "test8 (statistical navigation) passed" << std::endl;
}

// testing safety functions via the navigator
void test_safety()
{
	// statistical test  of navigation via comparison with ROOT navigation
		for(int i=0;i<1000000;++i)
		{
			//std::cerr << "START ITERATION " << i << std::endl;
			double x = RNG::Instance().uniform(-10,10);
			double y = RNG::Instance().uniform(-10,10);
			double z = RNG::Instance().uniform(-10,10);

			// VecGeom navigation
			Vector3D<Precision> p(x,y,z);

			NavigationState state(4);
			SimpleNavigator nav;
			nav.LocatePoint( RootManager::Instance().world(),
					p, state, true);
			double safety = nav.GetSafety( p, state );

			TGeoNavigator * rootnav = ::gGeoManager->GetCurrentNavigator();
			rootnav->FindNode(x,y,z);
			rootnav->SetCurrentPoint(x,y,z);
			double safetyRoot = rootnav->Safety();

			assert( fabs( safetyRoot - safety ) < 1E-9 );
		}
		std::cerr << "test9 (statistical safetytest from navigation) passed" << std::endl;
}


void testDistanceToOut() {

  const int n = 1<<15;
  for(int i=0; i<n; ++i) {
	VPlacedVolume const* world = GeoManager::Instance().world();
	SOA3D<Precision> points(n);
	SOA3D<Precision> directions(n);
	volumeUtilities::FillUncontainedPoints(*world, points);
	volumeUtilities::FillBiasedDirections(*world, points, 0.8, directions);
	int mismatches = 0;

	//std::cerr << "START ITERATION " << i << std::endl;
	double xpos = RNG::Instance().uniform(-10,10);
	double ypos = RNG::Instance().uniform(-10,10);
	double zpos = RNG::Instance().uniform(-10,10);
	std::cerr<<"spot 2"<< std::endl;
	points.push_back( Vector3D<Precision>(xpos,ypos,zpos) );
	std::cerr<<"spot 3"<< std::endl;
	double xdir = RNG::Instance().uniform(-1,1);
	double ydir = RNG::Instance().uniform(-1,1);
	double zdir = RNG::Instance().uniform(-1,1);
	double fnorm = std::sqrt(xdir*xdir + ydir*ydir + zdir*zdir);
	std::cerr<<"spot 4"<< std::endl;
	xdir /= fnorm;
	ydir /= fnorm;
	zdir /= fnorm;
	std::cerr<<"spot 4a"<< std::endl;
	directions.push_back( Vector3D<Precision>(xdir,ydir,zdir) );
	std::cerr<<"spot 4b"<< std::endl;

	// VecGeom navigation
	// Vector3D<Precision> p(xpos,ypos,zpos);

	Vector3D<Precision> master[n];
	std::cerr<<"spot 4c"<< std::endl;
	Vector3D<Precision> direction[n];
	std::cerr<<"spot 4d"<< std::endl;
	Precision steps[n], vecDistances[n];
	std::cerr<<"spot 4e"<< std::endl;
	Precision master_c[3*n];
	std::cerr<<"spot 4f"<< std::endl;
	Precision direction_c[3*n];
	std::cerr<<"spot 5"<< std::endl;
	for(int i=0; i<n; ++i) {
	  std::cout<<"spot 6: i="<< i <<": "<< points[i][0] <<' '<< points[i][1] <<' '<< points[i][2] << std::endl;
	  master[i][0] = points[i][0];
	  master[i][1] = points[i][1];
	  master[i][2] = points[i][2];
	  std::cout<<"spot 6: i="<< i <<": "<< master[i][0] <<' '<< master[i][1] <<' '<< master[i][2] << std::endl;
	  master_c[3*i+0] = points[i][0];
	  master_c[3*i+1] = points[i][1];
	  master_c[3*i+2] = points[i][2];
	  direction_c[3*i+0] = directions[i][0];
	  direction_c[3*i+1] = directions[i][1];
	  direction_c[3*i+2] = directions[i][2];
	  steps[i] = 1.0e29;
	}


	// vectorized calculations
	Stopwatch timer;
	std::cout<<"spot 7"<< std::endl;
	timer.Start();
	world->DistanceToOut(points,directions,steps,vecDistances);
	Precision vecTime = timer.Stop();

	std::cout<<"spot 8"<< std::endl;

	// scalar calculations
	Precision scalarDistances[n], rootDistances[n];
	timer.Start();
	for(int i = 0; i < n; ++i) {
	  scalarDistances[i] = world->DistanceToOut(master[i], direction[i]);
	}
	Precision scalarTime = timer.Stop();

	std::cout<<"spot 9"<< std::endl;

	// ROOT-based calculations
	Precision local[3*n];
	TGeoNode const *node = gGeoManager->GetTopNode();
	TGeoShape const* worldShape = node->GetVolume()->GetShape();
	for(int i=0; i<n; ++i) {
	    node->GetMatrix()->MasterToLocal(&master_c[3*i], &local[3*i]);
	}

	std::cout<<"spot 10"<< std::endl;

	timer.Start();
	for (int i = 0; i < n; ++i) {
	  //	    node->GetMatrix()->MasterToLocal(&master_c[3*i], &local[3*i]);
	    rootDistances[i] =  worldShape->DistFromInside(&local[3*i], &direction_c[3*i]);
	}
	Precision rootTime = timer.Stop();

	std::cout<<"Timings: scalar="<< scalarTime*1000
		 <<", vect="<< vecTime*1000
		 <<", root="<< rootTime*1000
		 <<" (msec)"<< std::endl;

	// compare results
	for(int i=0; i<n; ++i) {
	    const Precision diff = fabs(vecDistances[i]-rootDistances[i]);
	    if (diff > 1e-12) {
	      std::cerr << "Mismatch: " << vecDistances[i] << " / " << rootDistances[i] <<" / "<< scalarDistances[i] << std::endl;
	      mismatches++;
	    } else {
	      // std::cerr << "Match: " << vecgeom << " / " << root << std::endl;
	    }
	}
	std::cerr << mismatches << " / " << n << " mismatches detected.\n";
  }
}


void foo(Transformation3D const &m, Vector3D<Precision> const p, Vector3D<Precision> & localp )
{
  m.DoRotation<0x1B1>(p,localp);
}

void bar(Transformation3D const &m, Vector3D<Precision> const p, Vector3D<Precision> & localp )
{
  m.DoRotation_new<0x1B1>(p,localp);
}

void testDoRotation() {

  // instantiate a transformation to test Transformation3D::DoRotation()
  // double halfSqrt2 = 0.7071067811865476;
  Transformation3D const matrix_0x1b1 = {0,0,0,0,45,0};
  std::cout << matrix_0x1b1 << std::endl;

  // allocate space
  const int n = 1<<15;
  SOA3D<Precision> points(n);
  //SOA3D<Precision> directions(n);

  // use the world volume to generate random points
  VPlacedVolume const* world = GeoManager::Instance().world();
  volumeUtilities::FillUncontainedPoints(*world, points);
  //volumeUtilities::FillBiasedDirections(*world, points, 0.8, directions);

  // exercise the MatrixTransformation::DoRotation() methods
  Vector3D<Precision> local(0,0,0);
  for(int i=0; i<n; ++i) {
      matrix_0x1b1.Transform<0,0x1B1>( points[i], local );
      Precision norm_points = points[i][0]*points[i][0] + points[i][1]*points[i][1] + points[i][2]*points[i][2];
      Precision norm_local = local[0]*local[0] + local[1]*local[1] + local[2]*local[2];
      assert( fabs( norm_points - norm_local) < 1.e-12 );
  }

  std::cerr << "testDoRotation passed" << std::endl;
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
    test8();
    //testDoRotation();
    testDistanceToOut();
    test_safety();

    return 0;
}
