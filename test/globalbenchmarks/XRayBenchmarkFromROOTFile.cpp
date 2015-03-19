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
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <map>
#include <cassert>

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
#define WRITE_FILE_NAME "volumeImage.bmp" // output image name

using namespace vecgeom;

#pragma pack(push, 1)

typedef struct tFILE_HEADER
{
  unsigned short bfType;
  unsigned long bfSize;
  unsigned short bfReserved1;
  unsigned short bfReserved2;
  unsigned long bfOffBits;
} FILE_HEADER;

#pragma pack(pop)

typedef struct tINFO_HEADER
{
   unsigned long biSize;
   unsigned long biWidth;
   unsigned long biHeight;
   unsigned short biPlanes;
   unsigned short biBitCount;
   unsigned long biCompression;
   unsigned long biSizeImage;
   unsigned long biXPelsPerMeter;
   unsigned long biYPelsPerMeter;
   unsigned long biClrUsed;
   unsigned long biClrImportant;
} INFO_HEADER;

typedef struct tMY_BITMAP
{
  FILE_HEADER  bmpFileHeader;
  INFO_HEADER  bmpInfoHeader;
  unsigned char* bmpPalette;
  unsigned char* bmpRawData;
} MY_BITMAP;

bool usolids= true;

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

if(VERBOSE){
    std::cout << "from [" << axis1_start << ";" << axis2_start
              << "] to [" << axis1_end   << ";" << axis2_end << "]\n";
    std::cout << "Xpixels " << data_size_x << " YPixels " << data_size_y << "\n";

    std::cout << pixel_axis << "\n";
}

double pixel_width_1 = (axis1_end-axis1_start)/data_size_x;
double pixel_width_2 = (axis2_end-axis2_start)/data_size_y;

    if(VERBOSE){
    std::cout << pixel_width_1 << "\n";
    std::cout << pixel_width_2 << "\n";
    }

    for( int pixel_count_2 = 0; pixel_count_2 < data_size_y; ++pixel_count_2 ){
        for( int pixel_count_1 = 0; pixel_count_1 < data_size_x; ++pixel_count_1 )
        {
            double axis1_count = axis1_start + pixel_count_1 * pixel_width_1 + 1E-6;
            double axis2_count = axis2_start + pixel_count_2 * pixel_width_2 + 1E-6;

            if(VERBOSE) {
                std::cout << "\n OutputPoint("<< axis1_count<< ", "<< axis2_count<< ")\n";
            }
            // set start point of XRay
            Vector3D<Precision> p;

            if( axis== 1 )
              p.Set( origin[0]-bbox[0], axis1_count, axis2_count);
            else if( axis== 2)
              p.Set( axis1_count, origin[1]-bbox[1], axis2_count);
            else if( axis== 3)
              p.Set( axis1_count, axis2_count, origin[2]-bbox[2]);

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

          //  std::cout << pixel_count_1 << " " << pixel_count_2 << " " << dir << "\t" << p << "\t";
          //  std::cout << "IN|OUT" << nav->IsOutside() << "\n";
          //  if( node ) std::cout <<    node->GetVolume()->GetName() << "\t";
            while( node !=NULL ) {
                node = nav->FindNextBoundaryAndStep( vecgeom::kInfinity );
                distancetravelled+=nav->GetStep();

                if(VERBOSE) {
                    if( node != NULL ){
                        std::cout << "  VolumeName: "<< node->GetVolume()->GetName();
                    }
                    else
                       std::cout << "  NULL: ";

                    std::cout << " step[" << nav->GetStep()<< "]"<< std::endl;
                    double const * pROOT = nav->GetCurrentPoint();
                    p = Vector3D<Precision>(pROOT[0],pROOT[1],pROOT[2]);
                    std::cout << " point(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
                }
                // Increase passed_volume
                // TODO: correct counting of travel in "world" bounding box
                crossedvolumecount++;
              } // end while
            // std::cout << crossedvolumecount << "\n";

            ///////////////////////////////////
            // Store the number of passed volume at 'volume_result'
            *(image+pixel_count_2*data_size_x+pixel_count_1)= crossedvolumecount;

            if(VERBOSE) {
                std::cout << "  EndOfBoundingBox:";
                std::cout << " PassedVolume:" << "<"<< crossedvolumecount << " ";
                std::cout << " step[" << nav->GetStep()<< "]";
                std::cout << " Distance: " << distancetravelled<< std::endl;
            }
      } // end inner loop
    } // end outer loop
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



if(VERBOSE){
    std::cout << "from [" << axis1_start << ";" << axis2_start
              << "] to [" << axis1_end   << ";" << axis2_end << "]\n";
    std::cout << "Xpixels " << data_size_x << " YPixels " << data_size_y << "\n";

    std::cout << pixel_axis << "\n";
}
    double pixel_width_1 = (axis1_end-axis1_start)/data_size_x;
    double pixel_width_2 = (axis2_end-axis2_start)/data_size_y;

    if(VERBOSE){
        std::cout << pixel_width_1 << "\n";
        std::cout << pixel_width_2 << "\n";
    }

    NavigationState * newnavstate = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );
    NavigationState * curnavstate = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );

    for( int pixel_count_2 = 0; pixel_count_2 < data_size_y; ++pixel_count_2 ){
        for( int pixel_count_1 = 0; pixel_count_1 < data_size_x; ++pixel_count_1 )
        {
            double axis1_count = axis1_start + pixel_count_1 * pixel_width_1 + 1E-6;
            double axis2_count = axis2_start + pixel_count_2 * pixel_width_2 + 1E-6;

            if(VERBOSE) {
                std::cout << "\n OutputPoint("<< axis1_count<< ", "<< axis2_count<< ")\n";
            }
         //   std::cout << pixel_count_1 << " " << pixel_count_2 << "\n";

            // set start point of XRay
            Vector3D<Precision> p;
            if( axis== 1 )
              p.Set( origin[0]-bbox[0], axis1_count, axis2_count);
            else if( axis== 2)
              p.Set( axis1_count, origin[1]-bbox[1], axis2_count);
            else if( axis== 3)
              p.Set( axis1_count, axis2_count, origin[2]-bbox[2]);

            SimpleNavigator nav;
            curnavstate->Clear();
            nav.LocatePoint( GeoManager::Instance().GetWorld(), p, *curnavstate, true );

#ifdef VECGEOM_DISTANCE_DEBUG
            gGeoManager->GetCurrentNavigator()->FindNode( p.x(), p.y(), p.z() );
#endif

//          curnavstate->Print();

            double distancetravelled=0.;
            int crossedvolumecount=0;

            if(VERBOSE) {
              std::cout << " StartPoint(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
              std::cout << " Direction <" << dir[0] << ", " << dir[1] << ", " << dir[2] << ">"<< std::endl;
            }

            while( ! curnavstate->IsOutside() ) {
                double step = 0;
                newnavstate->Clear();
                nav.FindNextBoundaryAndStep( p,
                        dir,
                        *curnavstate,
                        *newnavstate,
                        vecgeom::kInfinity, step);

                //std::cout << "step " << step << "\n";
                distancetravelled+=step;
//
//              std::cout << "GOING FROM "
//                       << curnavstate->Top()->GetLabel() << "(";
//                        curnavstate->Top()->PrintType();
//                     std::cout << ") to ";
//
//                if( newnavstate->Top() ){
//                    std::cout << newnavstate->Top()->GetLabel() << "(";
//                    newnavstate->Top()->PrintType();
//                    std::cout << ")";
//                }
//                else
//                    std::cout << "outside ";
//
//                if ( curnavstate->Top() == newnavstate->Top() ) {
//                    std::cout << " CROSSING PROBLEM \n";
//                    curnavstate->Print();
//                    newnavstate->Print();
//                }
//
//                std::cout << "# step " << step << " crossed "<< crossedvolumecount << "\n";

                // here we have to propagate particle ourselves and adjust navigation state
                p = p + dir*(step + 1E-6);

//                std::cout << p << "\n";

                newnavstate->CopyTo(curnavstate);

                // Increase passed_volume
                // TODO: correct counting of travel in "world" bounding box
                if(step>0) crossedvolumecount++;

              //  if(crossedvolumecount > 1000){
                    //std::cerr << "OOPS: Problem for pixel " << pixel_count_1 << " " << pixel_count_2 << " \n";
                    //break;}
             } // end while

             ///////////////////////////////////
             // Store the number of passed volume at 'volume_result'
             *(image+pixel_count_2*data_size_x+pixel_count_1) = crossedvolumecount;

      } // end inner loop
    } // end outer loop

    NavigationState::ReleaseInstance( curnavstate );
    NavigationState::ReleaseInstance( newnavstate );

} // end XRayWithVecGeom

#ifdef VECGEOM_GEANT4
G4VPhysicalVolume * SetupGeant4Geometry( std::string volumename,
        Vector3D<Precision> worldbbox)
{

    // ATTENTION: THERE IS A (OR MIGHT BE) UNIT MISSMATCH HERE BETWEEN ROOT AND GEANT
    // ROOT = cm and GEANT4 = mm; basically a factor of 10 in all dimensions

     const double UNITCONV=10.;

//       // take G4 geometry from gdml file
       G4GDMLParser parser;
       parser.Read( "cms2015.gdml" );

       G4LogicalVolumeStore * store = G4LogicalVolumeStore::GetInstance();
//
       int found=0;
       G4LogicalVolume * foundvolume;
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
        G4PVPlacement * xrayedpl = new G4PVPlacement(
                 NULL, /* rotation */
                 G4ThreeVector(0,0,0), /* translation */
                 foundvolume, /* current logical */
                 "xrayedpl",
                 worldlv, /* this is where it is placed */
                 0,0);

        // voxelize
      // G4GeometryManager::GetInstance()->CloseGeometry( worldpv );

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

     const double UNITCONV=10.;
     G4Navigator * nav = new G4Navigator();

        // now start XRay procedure
        nav->SetWorldVolume( world );

        double pixel_width_1 = (axis1_end-axis1_start)/data_size_x;
        double pixel_width_2 = (axis2_end-axis2_start)/data_size_y;

        G4ThreeVector d(dir.x(),dir.y(),dir.z());
        for( int pixel_count_2 = 0; pixel_count_2 < data_size_y; ++pixel_count_2 ){
               for( int pixel_count_1 = 0; pixel_count_1 < data_size_x; ++pixel_count_1 )
               {
                   double axis1_count = axis1_start + pixel_count_1 * pixel_width_1 + 1E-6;
                   double axis2_count = axis2_start + pixel_count_2 * pixel_width_2 + 1E-6;

                   // set start point of XRay
                   G4ThreeVector p;
                   if( axis== 1 )
                     p = UNITCONV*G4ThreeVector( origin[0]-bboxscreen[0], axis1_count, axis2_count );
                   else if( axis== 2)
                     p = UNITCONV*G4ThreeVector( axis1_count, origin[1]-bboxscreen[1], axis2_count );
                   else if( axis== 3)
                     p = UNITCONV*G4ThreeVector( axis1_count, axis2_count, origin[2]-bboxscreen[2] );

                   // false == locate from top
                   G4VPhysicalVolume const * vol
                    = nav->LocateGlobalPointAndSetup( p, &d, false );

             //      std::cerr << p << " in vol " << vol->GetName() << " N D "
               //            << vol->GetLogicalVolume()->GetNoDaughters() << "\n";

                   double distancetravelled=0.;
                   int crossedvolumecount=0;

                   while( vol!=NULL ) {
                       crossedvolumecount++;
                       double safety;
                       // do one step ( this will internally adjust the current point and so on )
                       // also calculates safety

                       double step = nav->ComputeStep( p, d, vecgeom::kInfinity, safety );

//                       std::cerr << " STEP " << step << " ENTERING " << nav->EnteredDaughterVolume() << "\n";

                       // calculate next point ( do transportation ) and volume ( should go across boundary )
                       G4ThreeVector next = p + (step + 1E-6) * d;

                       nav->SetGeometricallyLimitedStep();
                       vol = nav->LocateGlobalPointAndSetup( next, &d, true);
                       p=next;
                   }
//                ///////////////////////////////////
//                // Store the number of passed volume at 'volume_result'
                  *(image+pixel_count_2*data_size_x+pixel_count_1) = crossedvolumecount;
         } // end inner loop
       } // end outer loop
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

  double pixel_width= 0;
  double pixel_axis= 1.;

  if( argc < 5 )
  {
    std::cerr<< std::endl;
    std::cerr<< "Need to give rootfile, volumename, axis and number of axis"<< std::endl;
    std::cerr<< "USAGE : ./XRayBenchmarkFromROOTFile [rootfile] [VolumeName] [ViewDirection(Axis)] [PixelWidth(OutputImageSize)] [--usolids|--vecgeom(Default:usolids)]"<< std::endl;
    std::cerr<< "  ex) ./XRayBenchmarkFromROOTFile cms2015.root BSCTrap y 95"<< std::endl;
    std::cerr<< "      ./XRayBenchmarkFromROOTFile cms2015.root PLT z 500 --vecgeom"<< std::endl<< std::endl;
    return 1;
  }

  TGeoManager::Import( argv[1] );
  std::string testvolume( argv[2] );

  if( strcmp(argv[3], "x")==0 )
    axis= 1;
  else if( strcmp(argv[3], "y")==0 )
    axis= 2;
  else if( strcmp(argv[3], "z")==0 )
    axis= 3;
  else
  {
    std::cerr<< "Incorrect axis"<< std::endl<< std::endl;
    return 1;
  }

  pixel_width= atof(argv[4]);

  for(auto i= 5; i< argc; i++)
  {
    if( ! strcmp(argv[i], "--usolids") )
      usolids= true;
    
    if( ! strcmp(argv[i], "--vecgeom") )
      usolids= false;
  }

  int found = 0;
  TGeoVolume * foundvolume = NULL;
  // now try to find shape with logical volume name given on the command line
  TObjArray *vlist = gGeoManager->GetListOfVolumes( );
  for( auto i = 0; i < vlist->GetEntries(); ++i )
  {
    TGeoVolume * vol = reinterpret_cast<TGeoVolume*>(vlist->At( i ));
    std::string fullname(vol->GetName());
    
    // strip off pointer information
    std::string strippedname(fullname, 0, fullname.length()-4);
    
    std::size_t founds = strippedname.compare(testvolume);
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

    DeleteROOTVoxels();

    TGeoManager * mg1 = gGeoManager;
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
    Vector3D<Precision> dir;
    
    if(axis== 1)
    {
      //dir.Set(1., 0., 0.);
      Transformation3D trans( 0, 0, 0, 5, 5, 5);
      trans.Print();
      dir = trans.TransformDirection( Vector3D<Precision> (1,0,0));

      axis1_start= origin[1]- dy;
      axis1_end= origin[1]+ dy;
      axis2_start= origin[2]- dz;
      axis2_end= origin[2]+ dz;
      pixel_axis= (dy*2)/pixel_width;
    }
    else if(axis== 2)
    {
      dir.Set(0., 1., 0.);
      vecgeom::Transformation3D trans( 0, 0, 0, 5, 5, 5);
      dir = trans.TransformDirection(dir);
      axis1_start= origin[0]- dx;
      axis1_end= origin[0]+ dx;
      axis2_start= origin[2]- dz;
      axis2_end= origin[2]+ dz;
      pixel_axis= (dx*2)/pixel_width;
    }
    else if(axis== 3)
    {
      dir.Set(0., 0., 1.);
      vecgeom::Transformation3D trans( 0, 0, 0, 5, 5, 5);
      dir = trans.TransformDirection(dir);
      axis1_start= origin[0]- dx;
      axis1_end= origin[0]+ dx;
      axis2_start= origin[1]- dy;
      axis2_end= origin[1]+ dy;
      pixel_axis= (dx*2)/pixel_width;
    }

    // init data for image
    int data_size_x= (axis1_end-axis1_start)/pixel_axis;
    int data_size_y= (axis2_end-axis2_start)/pixel_axis;
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

    // Make bitmap file
    make_bmp(volume_result, "volumeImage_ROOT.bmp", data_size_x, data_size_y);

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

    make_bmp(volume_result, "volumeImage_Geant4.bmp", data_size_x, data_size_y);
    std::cout << std::endl;
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

    make_bmp(volume_result, "volumeImage_VecGeom.bmp", data_size_x, data_size_y);

    std::cout << std::endl;
    std::cout << " VecGeom Elapsed time : "<< timer.Elapsed() << std::endl;

    delete[] volume_result;
  }
  return 0;
}


int make_bmp(int const * volume_result, char const * name, int data_size_x, int data_size_y)
{

  MY_BITMAP* pBitmap= new MY_BITMAP;
  FILE *pBitmapFile;
  int width_4= (data_size_x+ 3)&~3;
  unsigned char* bmpBuf;

  bmpBuf = (unsigned char*) new unsigned char[data_size_y* width_4* 3+ 54];
  printf("\n Write bitmap...\n");

  unsigned int len= 0;

  // bitmap file header
  pBitmap->bmpFileHeader.bfType=0x4d42;
  pBitmap->bmpFileHeader.bfSize=data_size_y* width_4* 3+ 54;
  pBitmap->bmpFileHeader.bfReserved1= 0;
  pBitmap->bmpFileHeader.bfReserved2= 0;
  pBitmap->bmpFileHeader.bfOffBits= 54;
  
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfType, 2);
  len+= 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfSize, 4);
  len+= 4;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfReserved1, 2);
  len+= 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfReserved2, 2);
  len+= 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfOffBits, 4);
  len+= 4;

  // bitmap information header
  pBitmap->bmpInfoHeader.biSize= 40;
  pBitmap->bmpInfoHeader.biWidth= width_4;
  pBitmap->bmpInfoHeader.biHeight= data_size_y;
  pBitmap->bmpInfoHeader.biPlanes= 1;
  pBitmap->bmpInfoHeader.biBitCount= 24;
  pBitmap->bmpInfoHeader.biCompression= 0;
  pBitmap->bmpInfoHeader.biSizeImage= data_size_y* width_4* 3;
  pBitmap->bmpInfoHeader.biXPelsPerMeter= 0;
  pBitmap->bmpInfoHeader.biYPelsPerMeter= 0;
  pBitmap->bmpInfoHeader.biClrUsed= 0;
  pBitmap->bmpInfoHeader.biClrImportant=0;


  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biSize, 4); 
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biWidth, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biHeight, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biPlanes, 2);
  len+= 2;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biBitCount, 2);
  len+= 2;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biCompression, 4); 
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biSizeImage, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biXPelsPerMeter, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biYPelsPerMeter, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biClrUsed, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biClrImportant, 4);
  len+= 4;


  int x= 0;
  int y= 0;
  int origin_x= 0;

  int padding= width_4- data_size_x;
  int padding_idx= padding;
  unsigned char *imgdata= (unsigned char*) new unsigned char[data_size_y*width_4*3];

  int totalcount = 0;
  int maxcount = 0;

  while( y< data_size_y )
  {
    while( origin_x< data_size_x )
    {
      int value = *(volume_result+y*data_size_x+origin_x);
      totalcount += value;
      maxcount = ( value > maxcount )? value : maxcount;

      *(imgdata+y*width_4*3+x*3+0)= (value *50) % 256;
      *(imgdata+y*width_4*3+x*3+1)= (value *40) % 256;
      *(imgdata+y*width_4*3+x*3+2)= (value *30) % 256;
      
      x++;
      origin_x++;

      while( origin_x== data_size_x && padding_idx)
      {
// padding 4-byte at bitmap image
        *(imgdata+y*width_4*3+x*3+0)= 0;
        *(imgdata+y*width_4*3+x*3+1)= 0;
        *(imgdata+y*width_4*3+x*3+2)= 0;
        x++;
        padding_idx--;
      }
      padding_idx= padding;
    }
    y++;
    x= 0;
    origin_x= 0;
  }
  
  memcpy(bmpBuf + 54, imgdata, width_4* data_size_y* 3);

  pBitmapFile = fopen(name, "wb");
  fwrite(bmpBuf, sizeof(char), width_4*data_size_y*3+54, pBitmapFile);


  fclose(pBitmapFile);
  delete[] imgdata;
  delete[] bmpBuf;
  delete pBitmap;

  std::cout << " wrote image file " << name <<  "\n";
  std::cout << " total count " << totalcount << "\n";
  std::cout << " max count " << maxcount << "\n";
  return 0;
}
