/*
 * XRayBenchmarkFromROOTFile.cpp
 *
 * this benchmark performs an X-Ray scan of a (logical volume
 * in a) detector
 *
 * the benchmark stresses the distance functions of the volumes as well as
 * the basic higher level navigation functionality
 */

//#include "VUSolid.hh"
#include "management/RootGeoManager.h"
#include "volumes/LogicalVolume.h"

#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/Stopwatch.h"
#include "navigation/SimpleNavigator.h"
#include "base/Transformation3D.h"
#include "base/SOA3D.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <map>
#include <cassert>
#include <sstream>

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoNavigator.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoVoxelFinder.h"

#ifdef VECGEOM_GEANT4
#include "G4Navigator.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Box.hh"
#include "G4ThreeVector.hh"
#include "G4TouchableHistoryHandle.hh"
#include "G4GDMLParser.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PVPlacement.hh"
#include "G4GeometryManager.hh"
#include "management/G4GeoManager.h"
#endif


#undef NDEBUG

#define VERBOSE false   //true or false

using namespace vecgeom;

bool usolids= true;

// a global variable to switch voxels on or off
bool voxelize = true;

// produce a bmp image out of pixel information given in volume_results
int make_bmp(int const * image_result, char const *, int data_size_x, int data_size_y);

__attribute__((noinline))
void DeleteROOTVoxels()
{
    std::cout << " IN DELETE VOXEL METHOD \n";
    int counter=0;
    TObjArray * volist = gGeoManager->GetListOfVolumes();

    std::cout << " entries " << volist->GetEntries() << "\n";

    for(int i=0;i<volist->GetEntries();++i)
    {
        TGeoVolume *vol = (TGeoVolume *) volist->At(i);
        if ( vol!=NULL && vol->GetVoxels()!=0 )
        {
            counter++;
            delete vol->GetVoxels();
            vol->SetVoxelFinder(0);
        }
    }
    std::cout << " deleted " << counter << " Voxels \n";
}

void XRayWithROOT(int axis,
                 Vector3D<Precision> origin,
                 Vector3D<Precision> bbox,
                 Vector3D<Precision> dir,
                 double axis1_start, double axis1_end,
                 double axis2_start, double axis2_end,
                 int data_size_x,
                 int data_size_y,
                 double pixel_axis,
                 int * image) {

    int counter=0;

    // set start point of geantino
    Vector3D<Precision> p(0.,0.,0);

    TGeoNavigator * nav = gGeoManager->GetCurrentNavigator();
    nav->SetCurrentPoint( p.x(), p.y(), p.z() );
    nav->SetCurrentDirection( dir.x(), dir.y(), dir.z() );

    double distancetravelled=0.;
    int crossedvolumecount=0;

    if(VERBOSE) {
       std::cout << " StartPoint(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
       std::cout << " Direction <" << dir[0] << ", " << dir[1] << ", " << dir[2] << ">"<< std::endl;
    }

    // propagate until we leave detector
    TGeoNode const * node = nav->FindNode();

    std::cout << "INITIAL MAT :" <<  node->GetVolume()->GetMaterial()->GetName() << "\n";

    //  if( node ) std::cout <<    node->GetVolume()->GetName() << "\t";
    while( node !=NULL ) {
       node = nav->FindNextBoundaryAndStep( vecgeom::kInfinity );
       distancetravelled+=nav->GetStep();
       counter++;

       if(VERBOSE) {
          if( node != NULL )
              std::cout << " *R " << counter << " * " << " point(" << p[0] << ", " << p[1] << ", " << p[2] << ") goes to " << " VolumeName: "<< node->GetVolume()->GetName()
              << " (MAT: " << node->GetVolume()->GetMaterial()->GetName() << ") :";
          else
              std::cout << "  NULL: ";

          std::cout << " step[" << nav->GetStep()<< "]"<< std::endl;
          double const * pROOT = nav->GetCurrentPoint();
          p = Vector3D<Precision>(pROOT[0],pROOT[1],pROOT[2]);
       }
       // Increase passed_volume
       // TODO: correct counting of travel in "world" bounding box
       crossedvolumecount++;
   } // end while
   // std::cout << crossedvolumecount << "\n";

    if(VERBOSE) {
        std::cout << " PassedVolume:" << "<"<< crossedvolumecount << " ";
        std::cout << " total distance travelled: " << distancetravelled<< std::endl;
    }
} // end XRayWithROOT


void XRayWithVecGeom(int axis,
                  Vector3D<Precision> origin,
                  Vector3D<Precision> bbox,
                  Vector3D<Precision> dir,
                  double axis1_start, double axis1_end,
                  double axis2_start, double axis2_end,
                  int data_size_x,
                  int data_size_y,
                  double pixel_axis,
                  int * image) {

    Stopwatch internaltimer;

    NavigationState * newnavstate = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
    NavigationState * curnavstate = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
    int counter = 0;
            //   std::cout << pixel_count_1 << " " << pixel_count_2 << "\n";

    internaltimer.Start();

    // set start point of XRay
    Vector3D<Precision> p(0,0,0);
    SimpleNavigator nav;
    curnavstate->Clear();
    nav.LocatePoint( GeoManager::Instance().GetWorld(), p, *curnavstate, true );

#ifdef VECGEOM_DISTANCE_DEBUG
            gGeoManager->GetCurrentNavigator()->FindNode( p.x(), p.y(), p.z() );
#endif


    double distancetravelled=0.;
    int crossedvolumecount=0;

    if(VERBOSE) {
      std::cout << " StartPoint(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
      std::cout << " Direction <" << dir[0] << ", " << dir[1] << ", " << dir[2] << ">"<< std::endl;
    }

    while( ! curnavstate->IsOutside() ) {
        double step = 0;
        newnavstate->Clear();

        double safety=nav.GetSafety( p, *curnavstate );

        nav.FindNextBoundaryAndStep( p,
                dir,
                *curnavstate,
                *newnavstate,
                1e20, step);

        distancetravelled+=step;

        if(VERBOSE) {
            if( newnavstate->Top() != NULL )
              std::cout << " *VG " << counter++ << " * point" << p << " goes to " << " VolumeName: " << newnavstate->Top()->GetLabel();
            else
              std::cout << "  NULL: ";

              std::cout << " step[" << step << "]";
              std::cout << " safety[" << safety << "]";
              std::cout << " boundary[" << newnavstate->IsOnBoundary() << "]\n";
        }
        // here we have to propagate particle ourselves and adjust navigation state
        p = p + dir*(step + 1E-6);

        newnavstate->CopyTo(curnavstate);

        // Increase passed_volume
        // TODO: correct counting of travel in "world" bounding box
        if(step>0) crossedvolumecount++;
    } // end while
    if(VERBOSE) {
                        std::cout << " PassedVolume:" << "<"<< crossedvolumecount << " ";
                        std::cout << " Distance: " << distancetravelled<< std::endl;
    }

    internaltimer.Stop();

    std::cout << "VecGeom time " << internaltimer.Elapsed() << "\n";

    NavigationState::ReleaseInstance( curnavstate );
    NavigationState::ReleaseInstance( newnavstate );

} // end XRayWithVecGeom


// stressing the vector interface of navigator
void XRayWithVecGeom_VecNav(int axis,
                  Vector3D<Precision> origin,
                  Vector3D<Precision> bbox,
                  Vector3D<Precision> dir,
                  double axis1_start, double axis1_end,
                  double axis2_start, double axis2_end,
                  int data_size_x,
                  int data_size_y,
                  double pixel_axis,
                  int * image) {
    int counter=0;

    // we need N navstates ( where N should be a multiple of the SIMD width )
    unsigned int N = 8;
    NavigationState ** newnavstates = new NavigationState*[N];
    NavigationState ** curnavstates = new NavigationState*[N];
    for( unsigned int j=0;j<N;++j ){
        newnavstates[j] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
        curnavstates[j] = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
    }

    SOA3D<Precision> points(N);
    SOA3D<Precision> dirs(N);
    SOA3D<Precision> workspaceforlocalpoints(N);
    SOA3D<Precision> workspaceforlocaldirs(N);

    // initialize dirs from dir
    for( unsigned int j=0; j<N; ++j )
        dirs.set(j, dir.x(), dir.y(),dir.z());

    double * steps    = new double[N];
    double * psteps   = new double[N];
    double * safeties = new double[N];
    int * nextnodeworkspace = new int[N]; // some workspace for the navigator; not important here
    // initialize physical steps to infinity
    for(unsigned int j=0;j<N;++j)
        psteps[j]=vecgeom::kInfinity;


    Stopwatch internaltimer;
    internaltimer.Start();
    SimpleNavigator nav;
    // initialize points and locate them is serialized
    for( unsigned int j=0; j<N; ++j ){
                  points.set( j, 0, 0, 0 );
                  curnavstates[j]->Clear();
                  nav.LocatePoint( GeoManager::Instance().GetWorld(), points[j], *curnavstates[j], true );
    }

    double distancetravelled=0.;
    int crossedvolumecount=0;
    if(VERBOSE) {
       std::cout << " StartPoint(" << points[0].x() << ", " << points[1].y() << ", " <<
                  points[2].z() << ")";
       std::cout << " Direction <" << dirs[0].x() << ", " << dirs[1].y() << ", " << dirs[2].z() << ">"<< std::endl;
    }

    // we do the while loop only over the first "particle index"
    // the rest of the particles should follow exactly the same path
    while( ! curnavstates[0]->IsOutside() ) {
        nav.FindNextBoundaryAndStep(
                points,
                dirs,
                workspaceforlocalpoints,
                workspaceforlocaldirs,
                curnavstates,
                newnavstates,
                psteps,
                safeties,
                steps,
                nextnodeworkspace);

        //std::cout << "step " << step << "\n";
        distancetravelled+=steps[0];

        // TODO: DO HERE AN ASSERTION THAT ALL STEPS AGREE

        if(VERBOSE) {
            if( newnavstates[0]->Top() != NULL )
              std::cout << " *VGV " << counter++ << " * point" << points[0] << " goes to " << " VolumeName: " << newnavstates[0]->Top()->GetLabel();
            else
              std::cout << "  NULL: ";

              std::cout << " step[" << steps[0] << "]";
              std::cout << " boundary[" << newnavstates[0]->IsOnBoundary() << "]\n";
        }

        // here we have to propagate particle ourselves and adjust navigation state
        // propagate points
        for(unsigned int j=0;j<N;++j){
            points.set(j, points[j] + dirs[j]*(steps[0] + 1E-6));
            newnavstates[j]->CopyTo(curnavstates[j]);
        }

        // Increase passed_volume
        // TODO: correct counting of travel in "world" bounding box
        if(steps[0]>0) crossedvolumecount++;
     } // end while


    internaltimer.Stop();
    std::cout << "VecGeom vec time (per track) " << internaltimer.Elapsed()/N << "\n";

    for( unsigned int j=0; j<N ; ++j ) {
        NavigationState::ReleaseInstance( curnavstates[j] );
        NavigationState::ReleaseInstance( newnavstates[j] );
    }
} // end XRayWithVecGeomVectorInterface



#ifdef VECGEOM_GEANT4
G4VPhysicalVolume * SetupGeant4Geometry( std::string volumename,
        Vector3D<Precision> worldbbox)
{

    // ATTENTION: THERE IS A (OR MIGHT BE) UNIT MISSMATCH HERE BETWEEN ROOT AND GEANT
    // ROOT = cm and GEANT4 = mm; basically a factor of 10 in all dimensions

     const double UNITCONV=10.;

//   // take G4 geometry from gdml file
     G4GDMLParser parser;
     parser.Read( "cms2015.gdml" );

     G4LogicalVolumeStore * store = G4LogicalVolumeStore::GetInstance();
//
     int found=0;
     G4LogicalVolume * foundvolume = NULL;
     for( auto v : *store )
     {
           std::size_t founds = volumename.compare( v->GetName() );
           if ( founds==0 ){
                found++;
                foundvolume = v;
           }
     }
     std::cerr << " found logical volume " << volumename << " " << found << " times "  << "\n";

     // embed logical volume in a Box
     // create box first
     G4Box * worldb = new G4Box("BoundingBox",
             UNITCONV*worldbbox.x(), UNITCONV*worldbbox.y(), UNITCONV*worldbbox.z());
     G4LogicalVolume * worldlv = new G4LogicalVolume(worldb, 0, "world", 0,0,0);
     G4PVPlacement * worldpv =
            new G4PVPlacement(0,G4ThreeVector(0,0,0),"BoundingBox", worldlv, 0,false, 0,0);

     // embed found logical volume "foundvolume" into world bounding box
     new G4PVPlacement(
               NULL, /* rotation */
               G4ThreeVector(0,0,0), /* translation */
               foundvolume, /* current logical */
               "xrayedpl",
               worldlv, /* this is where it is placed */
               0,0);

     G4GeometryManager::GetInstance()->CloseGeometry( voxelize );

     return worldpv;
}
#endif

// performs the XRay scan using Geant4
#ifdef VECGEOM_GEANT4
int XRayWithGeant4(
        G4VPhysicalVolume * world /* the detector to scan */,
        int axis,
        Vector3D<Precision> origin,
        Vector3D<Precision> bboxscreen,
        Vector3D<Precision> dir,
        double axis1_start, double axis1_end,
        double axis2_start, double axis2_end,
        int data_size_x,
        int data_size_y,
        double pixel_axis,
        int * image) {


    // ATTENTION: THERE IS A (OR MIGHT BE) UNIT MISSMATCH HERE BETWEEN ROOT AND GEANT
     // ROOT = cm and GEANT4 = mm; basically a factor of 10 in all dimensions

     // const double UNITCONV=10.;
     G4Navigator * nav = new G4Navigator();

     // now start XRay procedure
     nav->SetWorldVolume( world );

     G4ThreeVector d(dir.x(),dir.y(),dir.z());

     G4ThreeVector p(0,0,0);

                   // false == locate from top
   G4VPhysicalVolume const * vol
    = nav->LocateGlobalPointAndSetup( p, &d, false );


   double distancetravelled=0.;
   int crossedvolumecount=0;
    int counter=0;
   while( vol!=NULL ) {
       crossedvolumecount++;
       double safety;
       // do one step ( this will internally adjust the current point and so on )
       // also calculates safety

       double step = nav->ComputeStep( p, d, vecgeom::kInfinity, safety );

//                       std::cerr << " STEP " << step << " ENTERING " << nav->EnteredDaughterVolume() << "\n";

       // calculate next point ( do transportation ) and volume ( should go across boundary )
       G4ThreeVector next = p + (step) * d;
       distancetravelled+=step;
       nav->SetGeometricallyLimitedStep();
       vol = nav->LocateGlobalPointAndSetup( next, &d, true);

       if(VERBOSE) {
         if( vol != NULL )
         std::cout << " *G4 " << counter++ << " * point" << p/10. << " goes to " << " VolumeName: " << vol->GetName();
       else
         std::cout << "  NULL: ";

         std::cout << " step[" << step/10. << "]"<< std::endl;
       }
       p=next;
   } // end while

   if(VERBOSE) {
       std::cout << " PassedVolume:" << "<"<< crossedvolumecount << " ";
       std::cout << " Distance: " << distancetravelled/10.<< std::endl;
   }
   return 0;
}
#endif

//////////////////////////////////
// main function
int main(int argc, char * argv[])
{
  int axis= 0;

  double axis1_start= 0.;
  double axis1_end= 0.;

  double axis2_start= 0.;
  double axis2_end= 0.;

  double pixel_axis= 1.;

  if( argc < 5 )
  {
    std::cerr<< std::endl;
    std::cerr<< "Need to give rootfile, volumename, direction phi and direction theta (in degrees)"<< std::endl;
    return 1;
  }

  TGeoManager::Import( argv[1] );
  std::string testvolume( argv[2] );

  //double directionphi   = atof(argv[3])*vecgeom::kDegToRad;
  //double directiontheta = atof(argv[4])*vecgeom::kDegToRad;

  for(auto i= 5; i< argc; i++)
  {
    if( ! strcmp(argv[i], "--usolids") )
      usolids= true;
    if( ! strcmp(argv[i], "--vecgeom") )
      usolids= false;
    if( ! strcmp(argv[i], "--novoxel") )
      voxelize = false;
  }

  int found = 0;
  TGeoVolume * foundvolume = NULL;
  // now try to find shape with logical volume name given on the command line
  TObjArray *vlist = gGeoManager->GetListOfVolumes( );
  for( auto i = 0; i < vlist->GetEntries(); ++i )
  {
    TGeoVolume * vol = reinterpret_cast<TGeoVolume*>(vlist->At( i ));
    std::string fullname(vol->GetName());
    
    std::size_t founds = fullname.compare(testvolume);
    if ( founds==0 ){
      found++;
      foundvolume = vol;

      std::cerr << "("<< i<< ")found matching volume " << foundvolume->GetName()
        << " of type " << foundvolume->GetShape()->ClassName() << "\n";
    }
  }

  std::cerr << "volume found " << found << " times \n\n";

  // if volume not found take world
  if( ! foundvolume ) {
      std::cerr << "specified volume not found; xraying complete detector\n";
      foundvolume = gGeoManager->GetTopVolume();
  }

  if( foundvolume ) {
    foundvolume->GetShape()->InspectShape();
    std::cerr << "volume capacity " 
          << foundvolume->GetShape()->Capacity() << "\n";

    // get bounding box to generate x-ray start positions
    double dx = ((TGeoBBox*)foundvolume->GetShape())->GetDX()*1.5;
    double dy = ((TGeoBBox*)foundvolume->GetShape())->GetDY()*1.5;
    double dz = ((TGeoBBox*)foundvolume->GetShape())->GetDZ()*1.5;
    double origin[3]= {0., };
    origin[0]= ((TGeoBBox*)foundvolume->GetShape())->GetOrigin()[0];
    origin[1]= ((TGeoBBox*)foundvolume->GetShape())->GetOrigin()[1];
    origin[2]= ((TGeoBBox*)foundvolume->GetShape())->GetOrigin()[2];
    
    TGeoMaterial * matVacuum = new TGeoMaterial("Vacuum",0,0,0);
    TGeoMedium * vac = new TGeoMedium("Vacuum",1,matVacuum);

    TGeoVolume* boundingbox= gGeoManager->MakeBox("BoundingBox", vac,
            std::abs(origin[0]) + dx,
            std::abs(origin[1]) + dy,
            std::abs(origin[2]) + dz );

    // TGeoManager * geom = boundingbox->GetGeoManager();
    std::cout << gGeoManager->CountNodes() << "\n";

    if(! voxelize ) DeleteROOTVoxels();

//    TGeoManager * mg1 = gGeoManager;
    gGeoManager = 0;

    TGeoManager * mgr2 = new TGeoManager();

//    delete gGeoManager;
//    gGeoManager = new TGeoManager();
    boundingbox->AddNode( foundvolume, 1);
    mgr2->SetTopVolume( boundingbox );
    mgr2->CloseGeometry();
    gGeoManager = mgr2;
    gGeoManager->Export("DebugGeom.root");

    mgr2->GetTopNode()->GetMatrix()->Print();

    std::cout << gGeoManager->CountNodes() << "\n";
    //delete world->GetVoxels();
    //world->SetVoxelFinder(0);
    
    std::cout<< std::endl;
    std::cout<< "BoundingBoxDX: "<< dx<< std::endl;
    std::cout<< "BoundingBoxDY: "<< dy<< std::endl;
    std::cout<< "BoundingBoxDZ: "<< dz<< std::endl;
    
    std::cout<< std::endl;
    std::cout<< "BoundingBoxOriginX: "<< origin[0]<< std::endl;
    std::cout<< "BoundingBoxOriginY: "<< origin[1]<< std::endl;
    std::cout<< "BoundingBoxOriginZ: "<< origin[2]<< std::endl<< std::endl;
  
    Vector3D<Precision> p;
    // Vector3D<Precision> dir( std::cos(directionphi)*std::sin(directiontheta), std::sin(directionphi)*std::sin(directiontheta),  std::cos(directiontheta) );

    //
    Vector3D<Precision> dir( -0.00366952650659481318523580384294 , 0.00101412421199570282163981982393 , 0.999991248519344400058628252737 );

    //Vector3D<Precision> dir( 1 , 0. , 0. );

    dir.FixZeroes();
    
    // init data for image
    int data_size_x= 1;//(axis1_end-axis1_start)/pixel_axis;
    int data_size_y= 1;//(axis2_end-axis2_start)/pixel_axis;
    int *volume_result= (int*) new int[data_size_y * data_size_x*3];


    Stopwatch timer;
    timer.Start();
    XRayWithROOT( axis,
            Vector3D<Precision>(origin[0],origin[1],origin[2]),
            Vector3D<Precision>(dx,dy,dz),
            dir,
            axis1_start, axis1_end,
            axis2_start, axis2_end,
            data_size_x, data_size_y,
            pixel_axis,
            volume_result );
    timer.Stop();

    std::cout << std::endl;
    std::cout << " ROOT Elapsed time : "<< timer.Elapsed() << std::endl;

#ifdef VECGEOM_GEANT4

    G4VPhysicalVolume * world = SetupGeant4Geometry( testvolume, Vector3D<Precision>( std::abs(origin[0]) + dx,
            std::abs(origin[1]) + dy,
            std::abs(origin[2]) + dz ) );
    G4GeoManager::Instance().LoadG4Geometry( world );

    timer.Start();

    XRayWithGeant4( world, axis,
            Vector3D<Precision>(origin[0],origin[1],origin[2]),
            Vector3D<Precision>(dx,dy,dz),
            dir,
            axis1_start, axis1_end,
            axis2_start, axis2_end,
            data_size_x, data_size_y,
            pixel_axis,
            volume_result );
    timer.Stop();
    std::cout << " Geant4 Elapsed time : "<< timer.Elapsed() << std::endl;

#endif

    // convert current gGeoManager to a VecGeom geometry
    RootGeoManager::Instance().LoadRootGeometry();
    std::cout << "Detector loaded " << "\n";
    timer.Start();
    XRayWithVecGeom( axis,
               Vector3D<Precision>(origin[0],origin[1],origin[2]),
               Vector3D<Precision>(dx,dy,dz),
               dir,
               axis1_start, axis1_end,
               axis2_start, axis2_end,
               data_size_x, data_size_y,
               pixel_axis,
               volume_result );
    timer.Stop();

    std::cout << " VecGeom Elapsed time : "<< timer.Elapsed() << std::endl;

    // use the vector interface
    timer.Start();
    XRayWithVecGeom_VecNav( axis,
                   Vector3D<Precision>(origin[0],origin[1],origin[2]),
                   Vector3D<Precision>(dx,dy,dz),
                   dir,
                   axis1_start, axis1_end,
                   axis2_start, axis2_end,
                   data_size_x, data_size_y,
                   pixel_axis,
                   volume_result );
    timer.Stop();
    std::cout << std::endl;
    std::cout << " VecGeom Vector Interface Elapsed time : "<< timer.Elapsed() << std::endl;


    delete[] volume_result;
  }
  return 0;
}
