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
#include "navigation/ABBoxNavigator.h"
#include "base/Transformation3D.h"
#include "base/SOA3D.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <map>
#include <vector>
#include <cassert>
#include <sstream>

#ifdef CALLGRIND
#include "base/callgrind.h"
#endif

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoNavigator.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoVoxelFinder.h"
#include "TGeoMaterial.h"

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

// a global variable to switch voxels on or off
bool voxelize = true;

// produce a bmp image out of pixel information given in volume_results
int make_bmp_header( );
int make_bmp(int const * image_result, char const *, int data_size_x, int data_size_y, bool linear = true);
int make_diff_bmp(int const * image1, int const * image2, char const *, int sizex, int sizey);


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
            double accumulateddensity =0.;

            if(VERBOSE) {
              std::cout << " StartPoint(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
              std::cout << " Direction <" << dir[0] << ", " << dir[1] << ", " << dir[2] << ">"<< std::endl;
            }

            // propagate until we leave detector
            TGeoNode const * node = nav->FindNode();
            TGeoMaterial const * curmat = node->GetVolume()->GetMaterial();

          //  std::cout << pixel_count_1 << " " << pixel_count_2 << " " << dir << "\t" << p << "\t";
          //  std::cout << "IN|OUT" << nav->IsOutside() << "\n";
          //  if( node ) std::cout <<    node->GetVolume()->GetName() << "\t";
            while( node !=NULL ) {
                node = nav->FindNextBoundaryAndStep( vecgeom::kInfinity );
                distancetravelled+=nav->GetStep();
                accumulateddensity+=curmat->GetDensity() * distancetravelled;

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
                if(nav->GetStep()>0.) crossedvolumecount++;
                curmat = (node!=0) ? node->GetVolume()->GetMaterial() : 0;
            } // end while
            // std::cout << crossedvolumecount << "\n";

            ///////////////////////////////////
            // Store the number of passed volume at 'volume_result'
            *(image+pixel_count_2*data_size_x+pixel_count_1) = crossedvolumecount;// accumulateddensity ;// crossedvolumecount;

            if(VERBOSE) {
                std::cout << "  EndOfBoundingBox:";
                std::cout << " PassedVolume:" << "<"<< crossedvolumecount << " ";
                std::cout << " step[" << nav->GetStep()<< "]";
                std::cout << " Distance: " << distancetravelled<< std::endl;
            }
      } // end inner loop
    } // end outer loop
} // end XRayWithROOT


template<typename Nav_t = SimpleNavigator>
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
                Nav_t navigator;
                navigator.FindNextBoundaryAndStep( p,
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

    for( int pixel_count_2 = 0; pixel_count_2 < data_size_y; ++pixel_count_2 ){
        for( int pixel_count_1 = 0; pixel_count_1 < data_size_x; ++pixel_count_1 )
        {
            double axis1_count = axis1_start + pixel_count_1 * pixel_width_1 + 1E-6;
            double axis2_count = axis2_start + pixel_count_2 * pixel_width_2 + 1E-6;

            if(VERBOSE) {
                std::cout << "\n OutputPoint("<< axis1_count<< ", "<< axis2_count<< ")\n";
            }
            // std::cout << pixel_count_1 << " " << pixel_count_2 << "\n";
            // set start points of XRay; points should be in a SOA/AOS

            SimpleNavigator nav;
            // initialize points and locate them is serialized
            for( unsigned int j=0; j<N; ++j ){

                if( axis== 1 )
                  points.set( j, origin[0]-bbox[0], axis1_count, axis2_count );
                else if( axis== 2)
                  points.set( j, axis1_count, origin[1]-bbox[1], axis2_count );
                else if( axis== 3)
                  points.set( j, axis1_count, axis2_count, origin[2]-bbox[2] );

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
                // propagate points
                for(unsigned int j=0;j<N;++j){
                    points.set(j, points[j] + dirs[j]*(steps[0] + 1E-6));
                    newnavstates[j]->CopyTo(curnavstates[j]);
                }

                // Increase passed_volume
                // TODO: correct counting of travel in "world" bounding box
                if(steps[0]>0) crossedvolumecount++;
             } // end while

             ///////////////////////////////////
             // Store the number of passed volume at 'volume_result'
             *(image+pixel_count_2*data_size_x+pixel_count_1) = crossedvolumecount;

      } // end inner loop
    } // end outer loop

    for( unsigned int j=0; j<N ; ++j ) {
        NavigationState::ReleaseInstance( curnavstates[j] );
        NavigationState::ReleaseInstance( newnavstates[j] );
    }
} // end XRayWithVecGeomVectorInterface



#ifdef VECGEOM_GEANT4
// a function allowing to clip geometry branches deeper than a certain level from the given volume
void DeleteAllG4VolumesDeeperThan( G4LogicalVolume * vol, unsigned int level ){
  if (level == 0) {
    // deletes daughters of this volume
    // vol->ClearDaughters() has a bug !!

    std::vector<G4VPhysicalVolume *> daughters;
    for (auto d = 0; d < vol->GetNoDaughters(); ++d)
      daughters.push_back(vol->GetDaughter(d));

    // now remove them
    for (unsigned int d = 0; d < daughters.size(); ++d)
      vol->RemoveDaughter(daughters[d]);
    return;
  } else {
    // recurse down
    for (auto d = 0; d < vol->GetNoDaughters(); d++) {
      auto node = vol->GetDaughter(d);
      if (node) {
        DeleteAllG4VolumesDeeperThan(node->GetLogicalVolume(), level - 1);
      }
    }
  }
}

G4VPhysicalVolume *SetupGeant4Geometry(std::string volumename, Vector3D<Precision> worldbbox, bool cutgeometry,
                                       unsigned int cutlevel) {

  // ATTENTION: THERE IS A (OR MIGHT BE) UNIT MISSMATCH HERE BETWEEN ROOT AND GEANT
  // ROOT = cm and GEANT4 = mm; basically a factor of 10 in all dimensions

  const double UNITCONV = 10.;

  G4GDMLParser parser;
  parser.Read("cms2015.gdml", false); // false == do not validate

  G4LogicalVolumeStore *store = G4LogicalVolumeStore::GetInstance();
  //
  int found = 0;
  G4LogicalVolume *foundvolume = NULL;
  for (auto v : *store) {
    std::size_t founds = volumename.compare(v->GetName());
    if (founds == 0) {
      found++;
      foundvolume = v;
    }
  }
  std::cerr << " found logical volume " << volumename << " " << found << " times "
            << "\n";

  if (cutgeometry)
    DeleteAllG4VolumesDeeperThan(foundvolume, cutlevel);

  // embed logical volume in a Box
  // create box first
  G4Box *worldb =
      new G4Box("BoundingBox", UNITCONV * worldbbox.x(), UNITCONV * worldbbox.y(), UNITCONV * worldbbox.z());
  G4LogicalVolume *worldlv = new G4LogicalVolume(worldb, 0, "world", 0, 0, 0);
  G4PVPlacement *worldpv = new G4PVPlacement(0, G4ThreeVector(0, 0, 0), "BoundingBox", worldlv, 0, false, 0, 0);

  // embed found logical volume "foundvolume" into world bounding box
  new G4PVPlacement(NULL,                   /* rotation */
                    G4ThreeVector(0, 0, 0), /* translation */
                    foundvolume,            /* current logical */
                    "xrayedpl", worldlv,    /* this is where it is placed */
                    0, 0);

  G4GeometryManager::GetInstance()->CloseGeometry(voxelize);

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

//                   double distancetravelled=0.;
                   int crossedvolumecount=0;

                   while( vol!=NULL ) {

                       double safety;
                       // do one step ( this will internally adjust the current point and so on )
                       // also calculates safety

                       double step = nav->ComputeStep( p, d, vecgeom::kInfinity, safety );
                       if(step>0.) crossedvolumecount++;
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
    return 0;
}
#endif

// a function allowing to clip geometry branches deeper than a certain level from the given volume
void DeleteAllNodesDeeperThan( TGeoVolume * vol, unsigned int level ){
    if( level == 0 ){
        std::cerr << " deleting daughters " << vol->GetNdaughters() << "\n";
        // deletes daughters of this volume
        vol->SetNodes( nullptr );
        std::cerr << " size is now " << vol->GetNdaughters() << "\n";
        return;
    }
    // recurse down
    for (auto d = 0; d < vol->GetNdaughters(); ++d) {
      TGeoNode *node = vol->GetNode(d);
      if (node) {
        DeleteAllNodesDeeperThan(node->GetVolume(), level - 1);
      }
    }
}

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
    std::cerr<< "USAGE : ./XRayBenchmarkFromROOTFile [rootfile] [VolumeName] [ViewDirection(Axis)]"
             << "[PixelWidth(OutputImageSize)] [--usolids|--vecgeom(Default:usolids)] [--novoxel(Default:voxel)]"
             << std::endl;
    std::cerr<< "  ex) ./XRayBenchmarkFromROOTFile cms2015.root BSCTrap y 95"<< std::endl;
    std::cerr<< "      ./XRayBenchmarkFromROOTFile cms2015.root PLT z 500 --vecgeom --novoxel"<< std::endl<< std::endl;
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

  unsigned int cutatlevel = 1000;
  bool cutlevel = false;
  for(auto i= 5; i< argc; i++)
  {
    if( ! strcmp(argv[i], "--usolids") )
      usolids= true;
    if( ! strcmp(argv[i], "--vecgeom") )
      usolids= false;
    if( ! strcmp(argv[i], "--novoxel") )
      voxelize = false;
    if( ! strcmp(argv[i], "--tolevel") ){
      cutlevel = true;
      cutatlevel = atoi(argv[i+1]);
      std::cout << "Cutting geometry at level " << cutatlevel << "\n";
    }

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
    double dx = ((TGeoBBox*)foundvolume->GetShape())->GetDX()*1.05;
    double dy = ((TGeoBBox*)foundvolume->GetShape())->GetDY()*1.05;
    double dz = ((TGeoBBox*)foundvolume->GetShape())->GetDZ()*1.05;
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

    gGeoManager = 0;

    TGeoManager * mgr2 = new TGeoManager();

    // do some surgery
    //delete foundvolume->GetNodes();
    //foundvolume->SetNodes(nullptr);
    if( cutlevel ){
        DeleteAllNodesDeeperThan(foundvolume, cutatlevel);
    }
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

    unsigned long long data_size_x;
    unsigned long long data_size_y;
    do {    
    if(axis== 1)
    {
      dir.Set(1., 0., 0.);
      //Transformation3D trans( 0, 0, 0, 5, 5, 5);
      //trans.Print();
     // dir = trans.TransformDirection( Vector3D<Precision> (1,0,0));

      axis1_start= origin[1]- dy;
      axis1_end= origin[1]+ dy;
      axis2_start= origin[2]- dz;
      axis2_end= origin[2]+ dz;
      pixel_axis= (dy*2)/pixel_width;
    }
    else if(axis== 2)
    {
      dir.Set(0., 1., 0.);
      //vecgeom::Transformation3D trans( 0, 0, 0, 5, 5, 5);
      //dir = trans.TransformDirection(dir);
      axis1_start= origin[0]- dx;
      axis1_end= origin[0]+ dx;
      axis2_start= origin[2]- dz;
      axis2_end= origin[2]+ dz;
      pixel_axis= (dx*2)/pixel_width;
    }
    else if(axis== 3)
    {
      dir.Set(0., 0., 1.);
      //vecgeom::Transformation3D trans( 0, 0, 0, 5, 5, 5);
      //dir = trans.TransformDirection(dir);
      axis1_start= origin[0]- dx;
      axis1_end= origin[0]+ dx;
      axis2_start= origin[1]- dy;
      axis2_end= origin[1]+ dy;
      pixel_axis= (dx*2)/pixel_width;
    }

    // init data for image
   data_size_x= (axis1_end-axis1_start)/pixel_axis;
   data_size_y= (axis2_end-axis2_start)/pixel_axis;

    if( data_size_x * data_size_y > 1E7L ){
      pixel_width/=2;
      std::cerr << data_size_x * data_size_y << "\n";
      std::cerr << "warning: image to big " << pixel_width << " " << pixel_axis << "\n";
    }
    else{
      std::cerr << "size ok " <<  data_size_x * data_size_y << "\n";

    }
} while ( data_size_x * data_size_y > 1E7L );
std::cerr << "allocating image" << "\n";
int *volume_result= (int*) new int[data_size_y * data_size_x*3];

#ifdef VECGEOM_GEANT4
    int *volume_result_Geant4= (int*) new int[data_size_y * data_size_x*3];
#endif
    int *volume_result_VecGeom= (int*) new int[data_size_y * data_size_x*3];
    int *volume_result_VecGeomABB= (int*) new int[data_size_y * data_size_x*3];

    Stopwatch timer;
    timer.Start();
#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif
    XRayWithROOT( axis,
            Vector3D<Precision>(origin[0],origin[1],origin[2]),
            Vector3D<Precision>(dx,dy,dz),
            dir,
            axis1_start, axis1_end,
            axis2_start, axis2_end,
            data_size_x, data_size_y,
            pixel_axis,
            volume_result );
#ifdef CALLGRIND
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif
    timer.Stop();

    std::cout << std::endl;
    std::cout << " ROOT Elapsed time : "<< timer.Elapsed() << std::endl;

    // Make bitmap file; generate filename
    std::stringstream imagenamebase;
    imagenamebase << "volumeImage_" << testvolume;
    if(axis==1) imagenamebase << "x";
    if(axis==2) imagenamebase << "y";
    if(axis==3) imagenamebase << "z";
    if(voxelize) imagenamebase << "_VOXELIZED_";
    std::stringstream ROOTimage;
    ROOTimage << imagenamebase.str();
    ROOTimage << "_ROOT.bmp";

    make_bmp(volume_result, ROOTimage.str().c_str(), data_size_x, data_size_y);
#ifdef VECGEOM_GEANT4

    G4VPhysicalVolume * world = SetupGeant4Geometry( testvolume, Vector3D<Precision>( std::abs(origin[0]) + dx,
            std::abs(origin[1]) + dy,
            std::abs(origin[2]) + dz ), cutlevel, cutatlevel );
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
            volume_result_Geant4 );
    timer.Stop();

    std::stringstream G4image;
    G4image << imagenamebase.str();
    G4image << "_Geant4.bmp";
    make_bmp(volume_result_Geant4, G4image.str().c_str(), data_size_x, data_size_y);
    std::cout << std::endl;
    std::cout << " Geant4 Elapsed time : "<< timer.Elapsed() << std::endl;
    std::stringstream G4diffimage;
    G4diffimage << imagenamebase.str();
    G4diffimage << "_diffROOTG4.bmp";
    make_diff_bmp(volume_result, volume_result_Geant4, G4diffimage.str().c_str(), data_size_x, data_size_y);
#endif

    // convert current gGeoManager to a VecGeom geometry
    RootGeoManager::Instance().LoadRootGeometry();
    std::cout << "Detector loaded " << "\n";
    ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
    std::cout << "voxelized " << "\n";
    timer.Start();
#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif
    XRayWithVecGeom<SimpleNavigator>( axis,
               Vector3D<Precision>(origin[0],origin[1],origin[2]),
               Vector3D<Precision>(dx,dy,dz),
               dir,
               axis1_start, axis1_end,
               axis2_start, axis2_end,
               data_size_x, data_size_y,
               pixel_axis,
               volume_result_VecGeom );
#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif
    timer.Stop();

    std::stringstream VecGeomimage;
    VecGeomimage << imagenamebase.str();
    VecGeomimage << "_VecGeom.bmp";
    make_bmp(volume_result_VecGeom, VecGeomimage.str().c_str(), data_size_x, data_size_y);

    std::stringstream VGRdiffimage;
    VGRdiffimage << imagenamebase.str();
    VGRdiffimage << "_diffROOTVG.bmp";
    make_diff_bmp(volume_result, volume_result_VecGeom, VGRdiffimage.str().c_str(), data_size_x, data_size_y);
#ifdef VECGEOM_GEANT4
    std::stringstream VGG4diffimage;
    VGG4diffimage << imagenamebase.str();
    VGG4diffimage << "_diffG4VG.bmp";
    make_diff_bmp(volume_result_Geant4, volume_result_VecGeom, VGG4diffimage.str().c_str(), data_size_x, data_size_y);
#endif

    std::cout << std::endl;
    std::cout << " VecGeom Elapsed time : "<< timer.Elapsed() << std::endl;

    timer.Start();
#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif
    XRayWithVecGeom<ABBoxNavigator>( axis,
    Vector3D<Precision>(origin[0],origin[1],origin[2]),
    Vector3D<Precision>(dx,dy,dz),
    dir,
    axis1_start, axis1_end,
    axis2_start, axis2_end,
    data_size_x, data_size_y,
    pixel_axis,
    volume_result_VecGeomABB );
#ifdef CALLGRIND
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif
    timer.Stop();

    return 0.;

    std::stringstream VecGeomABBimage;
    VecGeomABBimage << imagenamebase.str();
    VecGeomABBimage << "_VecGeomABB.bmp";
    make_bmp(volume_result_VecGeomABB, VecGeomABBimage.str().c_str(), data_size_x, data_size_y);

    make_diff_bmp(volume_result_VecGeom, volume_result_VecGeomABB, "diffVecGeomSimplevsABB.bmp", data_size_x, data_size_y);
    make_diff_bmp(volume_result, volume_result_VecGeomABB, "diffROOTVecGeomABB.bmp", data_size_x, data_size_y);


    std::cout << std::endl;
    std::cout << " VecGeom ABB Elapsed time : "<< timer.Elapsed() << std::endl;

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

    std::stringstream VecGeomimagevec;
    VecGeomimagevec << imagenamebase.str();
    VecGeomimagevec << "_VecGeomVecNav.bmp";
    make_bmp(volume_result, VecGeomimagevec.str().c_str(), data_size_x, data_size_y);



    delete[] volume_result;
  }
  return 0;
}


void make_bmp_header( MY_BITMAP * pBitmap, unsigned char * bmpBuf, int sizex, int sizey )
{
  int width_4= (sizex+ 3)&~3;
  unsigned int len= 0;

  // bitmap file header
  pBitmap->bmpFileHeader.bfType=0x4d42;
  pBitmap->bmpFileHeader.bfSize=sizey* width_4* 3+ 54;
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
  pBitmap->bmpInfoHeader.biHeight= sizey;
  pBitmap->bmpInfoHeader.biPlanes= 1;
  pBitmap->bmpInfoHeader.biBitCount= 24;
  pBitmap->bmpInfoHeader.biCompression= 0;
  pBitmap->bmpInfoHeader.biSizeImage= sizey* width_4* 3;
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
}


int make_bmp(int const * volume_result, char const * name, int data_size_x, int data_size_y, bool linear )
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

  // find out maxcount before doing the picture
  int maxcount = 0;
  int x=0,y=0,origin_x=0;
  while( y< data_size_y )
  {
     while( origin_x< data_size_x )
     {
       int value = *(volume_result+y*data_size_x+origin_x);
       maxcount = ( value > maxcount )? value : maxcount;

       x++;
       origin_x++;
     }
     y++;
     x = 0;
     origin_x = 0;
  }
//  maxcount = std::log(maxcount + 1);

  x= 0;
  y= 0;
  origin_x= 0;

  int padding= width_4- data_size_x;
  int padding_idx= padding;
  unsigned char *imgdata= (unsigned char*) new unsigned char[data_size_y*width_4*3];

  int totalcount = 0;

  while( y< data_size_y )
  {
    while( origin_x< data_size_x )
    {
      int value = *(volume_result+y*data_size_x+origin_x);
      totalcount += value;

      //*(imgdata+y*width_4*3+x*3+0)= (value *50) % 256;
      //*(imgdata+y*width_4*3+x*3+1)= (value *40) % 256;
      //*(imgdata+y*width_4*3+x*3+2)= (value *30) % 256;

      //*(imgdata+y*width_4*3+x*3+0)= (std::log(value)/(1.*maxcount)) * 256;
      //*(imgdata+y*width_4*3+x*3+1)= (std::log(value)/(1.2*maxcount)) * 256;
      //*(imgdata+y*width_4*3+x*3+2)= (std::log(value)/(1.4*maxcount)) * 256;
if( linear ){
      *(imgdata+y*width_4*3+x*3+0)= (value/(1.*maxcount)) * 256;
      *(imgdata+y*width_4*3+x*3+1)= (value/(1.*maxcount)) * 256;
      *(imgdata+y*width_4*3+x*3+2)= (value/(1.*maxcount)) * 256;
}
else {
          *(imgdata+y*width_4*3+x*3+0)= (log(value+1))/(1.*(log(1+maxcount))) * 256;
          *(imgdata+y*width_4*3+x*3+1)= (log(value+1))/(1.*(log(1+maxcount))) * 256;
          *(imgdata+y*width_4*3+x*3+2)= (log(value+1))/(1.*(log(1+maxcount))) * 256;
}
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



int make_diff_bmp(int const * image1, int const * image2, char const * name, int data_size_x, int data_size_y )
{

  MY_BITMAP* pBitmap= new MY_BITMAP;
  FILE *pBitmapFile;
  int width_4 = (data_size_x + 3)&~3;
  unsigned char* bmpBuf = (unsigned char*) new unsigned char[data_size_y* width_4* 3+ 54];

  // init buffer and write header
  make_bmp_header(pBitmap, bmpBuf, data_size_x, data_size_y);

  // TODO: verify the 2 images have same dimensions

  // find out maxcount before doing the picture
  int maxdiff = 0;
  int mindiff = 0;
  int x=0,y=0,origin_x=0;
  while( y< data_size_y )
  {
     while( origin_x< data_size_x )
     {
       int value = *(image1+y*data_size_x+origin_x) - *(image2+y*data_size_x + origin_x);
       maxdiff = ( value > maxdiff )? value : maxdiff;
       mindiff = ( value < mindiff )? value : mindiff;
       x++;
       origin_x++;
     }
     y++;
     x = 0;
     origin_x = 0;
  }


  x= 0;
  y= 0;
  origin_x= 0;

  int padding= width_4 - data_size_x;
  int padding_idx= padding;
  unsigned char *imgdata= (unsigned char*) new unsigned char[data_size_y*width_4*3];

  while( y< data_size_y )
  {
    while( origin_x< data_size_x )
    {
      int value = *(image1 + y*data_size_x + origin_x) - *(image2 + y*data_size_x + origin_x);

      if( value >=0 ){
          *(imgdata+y*width_4*3+x*3+0)= 255 - (value/(1.*maxdiff)) * 255;
          *(imgdata+y*width_4*3+x*3+1)= 255 - 0;// (value/(1.*maxcount)) * 256;
          *(imgdata+y*width_4*3+x*3+2)= 255 - 0;//(value/(1.*maxcount)) * 256;}
      }
      else
      {
          *(imgdata+y*width_4*3+x*3+0)= 255 - 0;
          *(imgdata+y*width_4*3+x*3+1)= 255 - 0;// (value/(1.*maxcount)) * 255;
          *(imgdata+y*width_4*3+x*3+2)= 255 - (value/(1.*mindiff)) * 255;//(value/(1.*maxcount)) * 255;}
      }
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
  return 0;
}
