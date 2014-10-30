/**
 * \author Sandro Wenzel (sandro.wenzel@cern.ch)
 */

// A complex test for a couple of features:
// a) creation of a ROOT geometry
// b) creation of a VecGeom geometry
// c) navigation in VecGeom geometry

#include <iostream>

#include "base/SOA3D.h"
#include "management/GeoManager.h"
#include "management/RootGeoManager.h"
#include "base/AOS3D.h"
#include "volumes/PlacedVolume.h"
#include "volumes/LogicalVolume.h"
#include "navigation/NavigationState.h"
#include "navigation/SimpleNavigator.h"
#include "base/RNG.h"
#include "benchmarking/BenchmarkResult.h"

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TGeoBranchArray.h"
#include <cassert>

#include <vector>

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

void testVecAssign( Vector3D<Precision> const & a, Vector3D<Precision> & b )
{
  b=a;
}

void test1()
{
   VPlacedVolume const * world = RootGeoManager::Instance().world();
   SimpleNavigator nav;
   NavigationState state(4);
   VPlacedVolume const * vol;

   // point should be in world
   Vector3D<Precision> p1(0, 9* 10/10., 0);

   vol=nav.LocatePoint( world, p1, state, true );
   assert( RootGeoManager::Instance().tgeonode( vol ) == ::gGeoManager->GetTopNode());
   std::cerr << "test1 passed" << "\n";
}

void test2()
{
   // inside box3 check
   VPlacedVolume const * world = RootGeoManager::Instance().world();
   SimpleNavigator nav;
   NavigationState state(4);
   VPlacedVolume const * vol;


   // point should be in box3
   Vector3D<Precision> p1(-5., 0., 0.);
   vol=nav.LocatePoint( world, p1, state, true );
   assert( std::strcmp( RootGeoManager::Instance().tgeonode( vol )->GetName() , "b3l_0" ) == 0);

   // point should also be in box3
   Vector3D<Precision> p2(5., 0., 0.);
   state.Clear();
   vol=nav.LocatePoint( world, p2, state, true );
   assert( std::strcmp( RootGeoManager::Instance().tgeonode( vol )->GetName() , "b3l_0" ) == 0 );
   std::cerr << "test2 passed" << "\n";
}


void test3()
{
   // inside box1 left check
   VPlacedVolume const * world = RootGeoManager::Instance().world();
   SimpleNavigator nav;
   NavigationState state(4);
   VPlacedVolume const * vol;
   Vector3D<Precision> p1(-9/10., 9*5/10., 0.);
   vol=nav.LocatePoint( world, p1, state, true );
   assert( std::strcmp( RootGeoManager::Instance().tgeonode( vol )->GetName() , "b1l_0" ) == 0);
   std::cerr << "test3 passed" << "\n";
}

void test3_2()
{
   // inside box1 right check
   VPlacedVolume const * world = RootGeoManager::Instance().world();
   SimpleNavigator nav;
   NavigationState state(4);
   VPlacedVolume const * vol;
   Vector3D<Precision> p1(9/10., 9*5/10., 0.);
   vol=nav.LocatePoint( world, p1, state, true );
   assert( std::strcmp( RootGeoManager::Instance().tgeonode( vol )->GetName() , "b1l_1" ) == 0);
   std::cerr << "test3_2 passed" << "\n";
}

void test4()
{
   // inside box2 check
   VPlacedVolume const * world = RootGeoManager::Instance().world();
   SimpleNavigator nav;
   NavigationState state(4);
   VPlacedVolume const * vol;
   Vector3D<Precision> p1(5., 9*5/10., 0.);
   vol=nav.LocatePoint( world, p1, state, true );
   assert( std::strcmp( RootGeoManager::Instance().tgeonode( vol )->GetName() , "b2l_0" ) == 0);
   std::cerr << "test4 passed" << "\n";
}

void test5()
{
   // outside world check
   VPlacedVolume const * world = RootGeoManager::Instance().world();
   SimpleNavigator nav;
   NavigationState state(4);
   VPlacedVolume const * vol;
   Vector3D<Precision> p1(-20, 0., 0.);
   vol=nav.LocatePoint( world, p1, state, true );
   assert( vol == 0 );

   assert( state.Top() == 0);
   assert( state.IsOutside() == true );
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
      VPlacedVolume const *vol= vecnav.LocatePoint( RootGeoManager::Instance().world(),
            Vector3D<Precision>(x,y,z) , state, true);

      assert( RootGeoManager::Instance().tgeonode(vol) == node );
   }
   std::cerr << "test6 (statistical location) passed" << "\n";
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
      vecnav.LocatePoint(RootGeoManager::Instance().world(), p, state, true);

      /*
      if ( vol1 != NULL )
      {
         std::cerr << RootManager::Instance().tgeonode( vol1 )->GetName() << "\n";
      }
*/

      // now we move global point in x direction and find new volume and path
      NavigationState state2(4);
      p+=Vector3D<Precision>(1.,0,0);
      VPlacedVolume const *vol2= vecnav.LocatePoint( RootGeoManager::Instance().world(),
               p , state2, true);
   /*
      if ( vol2 != NULL )
      {
         std::cerr << "new node " << RootManager::Instance().tgeonode( vol2 )->GetName() << "\n";

         // ROOT navigation
         TGeoNavigator  *nav = ::gGeoManager->GetCurrentNavigator();
         TGeoNode * node =nav->FindNode(p[0],p[1],p[2]);
         std::cerr << "ROOT new: " << node->GetName() << "\n";
      }*/

      // same with relocation
      // need local point first
      Transformation3D globalm = state.TopMatrix();
      Vector3D<Precision> localp = globalm.Transform( p );

      VPlacedVolume const *vol3= vecnav.RelocatePointFromPath( localp, state );
//      std::cerr << vol1 << " " << vol2 << " " << vol3 << "\n";
      assert( vol3  == vol2 );
   }
   std::cerr << "test7 (statistical relocation) passed" << "\n";
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

// unit test very basic navigation functionality
void testnavsimple()
{
    SimpleNavigator nav;
    Vector3D<Precision> d(0, -1, 0);
    Vector3D<Precision> d2(0, 1, 0);

    // point should be in world
    Vector3D<Precision> p1(-1, 9, 0);

    const int maxdepth = GeoManager::Instance().getMaxDepth();
    assert( maxdepth == 4);
    NavigationState currentstate( maxdepth );

    VPlacedVolume const * vol=nav.LocatePoint( GeoManager::Instance().GetWorld(),
            p1, currentstate, true );
    assert( RootGeoManager::Instance().tgeonode( vol ) == ::gGeoManager->GetTopNode());

    NavigationState newstate( maxdepth );

    // check with a large physical step
    double step;
    nav.FindNextBoundaryAndStep( p1, d, currentstate, newstate, kInfinity, step );
    assert( step == 4 );
    assert( newstate.IsOnBoundary() == true );
    assert( std::strcmp( RootGeoManager::Instance().tgeonode( newstate.Top() )->GetName() , "b2l_0" ));

    newstate.Clear();
    nav.FindNextBoundaryAndStep( p1, d, currentstate, newstate, 0.02, step );
    assert( step == 0.02 );
    assert( newstate.Top() == currentstate.Top() );
    assert( newstate.IsOnBoundary() == false );
    assert( newstate.IsOutside() == false );

    newstate.Clear();
    nav.FindNextBoundaryAndStep( p1, d2, currentstate, newstate, kInfinity, step );
    assert( step == 1 );
    assert( newstate.IsOnBoundary( ) == true );
    assert( newstate.Top() == NULL );
    assert( newstate.IsOutside( ) == true );
}

// navigation
void test8()
{
   // statistical test  of navigation via comparison with ROOT navigation
   for(int i=0;i<1000000;++i)
   {
      //std::cerr << "START ITERATION " << i << "\n";
      double x = RNG::Instance().uniform(-10,10);
      double y = RNG::Instance().uniform(-10,10);
      double z = RNG::Instance().uniform(-10,10);

      // VecGeom navigation
      Vector3D<Precision> p(x,y,z);
      Vector3D<Precision> d=sampleDir();

      NavigationState state(4), newstate(4);
      SimpleNavigator nav;
      nav.LocatePoint( RootGeoManager::Instance().world(), p, state, true);
      double step;
      nav.FindNextBoundaryAndStep( p, d, state, newstate, 1E30, step );

      TGeoNavigator * rootnav = ::gGeoManager->GetCurrentNavigator();

      // this is one of the disatvantages of the ROOT interface:
      // who enforces that I have to call FindNode first before FindNextBoundary
      // this is not apparent from the interface ?
      TGeoNode * node = rootnav->FindNode(x,y,z);
      assert( rootnav->GetCurrentNode()  == RootGeoManager::Instance().tgeonode( state.Top() ) );
      //std::cerr << " step " << step << " " << rootnav->GetStep() << "\n";
      //assert( step == rootnav->GetStep() );

      rootnav->SetCurrentPoint(x,y,z);
      rootnav->SetCurrentDirection(d[0],d[1],d[2]);
      rootnav->FindNextBoundaryAndStep( 1E30 );

     // std::cerr << " step " << step << " " << rootnav->GetStep() << "\n";
     // assert( step == rootnav->GetStep() );

      if( newstate.Top() != NULL )
      {
         if( rootnav->GetCurrentNode()  != RootGeoManager::Instance().tgeonode( newstate.Top() ) )
         {
            std::cerr << "ERROR ON ITERATION " << i << "\n";
            std::cerr << i << " " << d << "\n";
            std::cerr << i << " " << p << "\n";
            std::cerr << "I AM HERE: " << node->GetName() << "\n";
            std::cerr << "ROOT GOES HERE: " << rootnav->GetCurrentNode()->GetName() << "\n";
            std::cerr << rootnav->GetStep() << "\n";
            std::cerr << "VECGEOM GOES HERE: " << RootGeoManager::Instance().GetName( newstate.Top() ) << "\n";

            nav.InspectEnvironmentForPointAndDirection( p, d, state );
         }
      }
      assert( state.Top() != newstate.Top() );
   }
   std::cerr << "test8 (statistical navigation) passed" << "\n";
}

// testing safety functions via the navigator
void test_safety()
{
   // statistical test  of navigation via comparison with ROOT navigation
      for(int i=0;i<1000000;++i)
      {
         //std::cerr << "START ITERATION " << i << "\n";
         double x = RNG::Instance().uniform(-10,10);
         double y = RNG::Instance().uniform(-10,10);
         double z = RNG::Instance().uniform(-10,10);

         // VecGeom navigation
         Vector3D<Precision> p(x,y,z);

         NavigationState state(4);
         SimpleNavigator nav;
         nav.LocatePoint( RootGeoManager::Instance().world(),
               p, state, true);
         double safety = nav.GetSafety( p, state );

         TGeoNavigator * rootnav = ::gGeoManager->GetCurrentNavigator();
         rootnav->FindNode(x,y,z);
         rootnav->SetCurrentPoint(x,y,z);
         double safetyRoot = rootnav->Safety();

         assert( fabs( safetyRoot - safety ) < 1E-9 );
      }
      std::cerr << "test9 (statistical safetytest from navigation) passed" << "\n";
}

void test_NavigationStateToTGeoBranchArrayConversion()
{
	for(int i=0;i<100000;++i)
	{
	//std::cerr << "START ITERATION " << i << "\n";
	  double x = RNG::Instance().uniform(-10,10);
	  double y = RNG::Instance().uniform(-10,10);
	  double z = RNG::Instance().uniform(-10,10);

	  // VecGeom navigation
	  Vector3D<Precision> p(x,y,z);
	  Vector3D<Precision> d=sampleDir();
	  
	  NavigationState state(4), newstate(4);
	  SimpleNavigator nav;
	  nav.LocatePoint( RootGeoManager::Instance().world(),
			   p, state, true );
	  double step;
	  nav.FindNextBoundaryAndStep( p, d, state, newstate, 1E30, step );
	  
	  TGeoNavigator * rootnav = ::gGeoManager->GetCurrentNavigator();

	  // we are now testing conversion of states such that the ROOT navigator
	  // does not need to be initialized via FindNode()...
	  TGeoBranchArray * path = state.ToTGeoBranchArray();
	  // path->Print();
	  rootnav->ResetState();
	  rootnav->SetCurrentPoint(x,y,z);
	  rootnav->SetCurrentDirection(d[0],d[1],d[2]);
	  path->UpdateNavigator( rootnav );
	  rootnav->FindNextBoundaryAndStep( 1E30 );
	  
	  if( newstate.Top() != NULL )
	    {
	      if( rootnav->GetCurrentNode()
		  != RootGeoManager::Instance().tgeonode( newstate.Top() ) )
		{
		  std::cerr << "ERROR ON ITERATION " << i << "\n";
		  std::cerr << i << " " << d << "\n";
		  std::cerr << i << " " << p << "\n";
		  std::cerr << "ROOT GOES HERE: " << rootnav->GetCurrentNode()->GetName() << "\n";
		  std::cerr << rootnav->GetStep() << "\n";
		  std::cerr << "VECGEOM GOES HERE: " << RootGeoManager::Instance().GetName( newstate.Top() ) << "\n";
		  nav.InspectEnvironmentForPointAndDirection( p, d, state );
		}
	    }
	  assert( state.Top() != newstate.Top() );
	  delete path;
	}
	std::cerr << "test  (init TGeoBranchArray from NavigationState) passed" << "\n";
}

void test_geoapi()
{
   std::vector<VPlacedVolume *> v1;
   std::vector<LogicalVolume *> v2;

   GeoManager::Instance().getAllLogicalVolumes( v2 );
   assert(v2.size() == 4 );

   GeoManager::Instance().getAllPlacedVolumes( v1 );
   assert(v1.size() == 7 );

   assert(GeoManager::Instance().getMaxDepth()  ==  4);

   std::cerr << "test of geomanager query API passed" << "\n";
}

void test_aos3d()
{
	SOA3D<Precision> container1(1024);
	assert(container1.size() == 0);
	container1.push_back( Vector3D<Precision>(1,0,1));
	assert(container1.size() == 1);
	std::cerr << "test soa3d passed" << std::endl;

	AOS3D<Precision> container2(1024);
	assert(container2.size() == 0);
	container2.push_back( Vector3D<Precision>(1,0,1));
	assert(container2.size() == 1);
	std::cerr << "test aos3d passed" << std::endl;

}

int main()
{
    CreateRootGeom();
    RootGeoManager::Instance().LoadRootGeometry();
    RootGeoManager::Instance().world()->PrintContent();

    RootGeoManager::Instance().PrintNodeTable();

    test_geoapi();
    testnavsimple();
    test_NavigationStateToTGeoBranchArrayConversion();

    test1();
    test2();
    test3();
    test3_2();
    test4();
    test5();
    test6();
    test7();
    test8();
    test_safety();
    test_aos3d();
    return 0;
}
