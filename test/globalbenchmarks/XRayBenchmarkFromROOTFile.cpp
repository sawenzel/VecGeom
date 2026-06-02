/*
 * XRayBenchmarkFromROOTFile.cpp
 *
 * this benchmark performs an X-Ray scan of a (logical volume
 * in a) detector
 *
 * the benchmark stresses the distance functions of the volumes as well as
 * the basic higher level navigation functionality
 */

#include "VecGeomTest/RootGeoManager.h"

#include "VecGeom/volumes/LogicalVolume.h"

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/SOA3D.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <map>
#include <vector>
#include <sstream>

#include "VecGeom/navigation/VNavigator.h"
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"
#include "VecGeom/navigation/HybridNavigator2.h"

#include "VecGeom/volumes/utilities/ResultComparator.h"

// #define CALLGRIND
#ifdef CALLGRIND
#include <valgrind/callgrind.h>
#endif

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoNavigator.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoVoxelFinder.h"
#include "TGeoMaterial.h"
#include "TList.h"
#include "TROOT.h"

#ifdef VECGEOM_GEANT4
#include "G4Navigator.hh"
#include "G4VPhysicalVolume.hh"
#include "VecGeomTest/G4GeoManager.h"
#endif

#define VERBOSE false                     // true or false
#define WRITE_FILE_NAME "volumeImage.bmp" // output image name

using namespace vecgeom;

#pragma pack(push, 1)

typedef struct tFILE_HEADER {
  unsigned short bfType;
  unsigned long bfSize;
  unsigned short bfReserved1;
  unsigned short bfReserved2;
  unsigned long bfOffBits;
} FILE_HEADER;

#pragma pack(pop)

typedef struct tINFO_HEADER {
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

typedef struct tMY_BITMAP {
  FILE_HEADER bmpFileHeader;
  INFO_HEADER bmpInfoHeader;
  unsigned char *bmpPalette;
  unsigned char *bmpRawData;
} MY_BITMAP;

// a global variable to switch voxels on or off
bool voxelize     = true;
bool trackverbose = false;

// a global variable to switch on/off assemblies
// (if off .. assemblies will be flattened)
bool assemblies = true;

// configurable threshold for zero step detection
double kZeroStepLimit(1E-7);

// produce a bmp image out of pixel information given in volume_results
int make_bmp_header();
int make_bmp(int const *image_result, char const *, int data_size_x, int data_size_y, bool linear = false);

// compares 2 images and produces a difference image
// returns 0 if images are same
// returns !=0 if images different (according to some criterion)
int make_diff_bmp(int const *image1, int const *image2, char const *, int sizex, int sizey);

typedef Vector3D<Precision> Vec3_t;
typedef std::pair<Vec3_t, Vec3_t> PointAndDir_t;
typedef std::pair<PointAndDir_t, NavigationState *> TrackAndState_t;
typedef std::map<LogicalVolume const *, std::vector<TrackAndState_t> *> VolumeTracksMap_t;
VolumeTracksMap_t gVolumeTrackMap;
VolumeTracksMap_t gVolumeTrackForLevelLocate;

Vector3D<Precision> GetStartPoint(Vector3D<Precision> const &origin, Vector3D<Precision> const &bbox,
                                  double axis1_count, double axis2_count, int axis)
{
  Vector3D<Precision> tmp;
  // a shift to make sure the ray starts inside the world bounding box
  double shift(1.E-6);
  if (axis == 1) {
    tmp.Set(origin[0] - (bbox[0] - shift), axis1_count, axis2_count);
  } else if (axis == 2) {
    tmp.Set(axis1_count, origin[1] - (bbox[1] - shift), axis2_count);
  } else if (axis == 3) {
    tmp.Set(axis1_count, axis2_count, origin[2] - (bbox[2] - shift));
  }
  return tmp;
}

// #define LOGDATA

void AddTrack(LogicalVolume const *lvol, Vec3_t p, Vec3_t d, NavigationState const *state)
{
  NavigationState *newstate = new NavigationState(*state);
  if (gVolumeTrackMap.find(lvol) == gVolumeTrackMap.end()) {
    std::vector<TrackAndState_t> *v = new std::vector<TrackAndState_t>();
    gVolumeTrackMap[lvol]           = v;
    v->push_back(TrackAndState_t(PointAndDir_t(p, d), newstate));
  } else {
    auto &v = gVolumeTrackMap[lvol];
    v->push_back(TrackAndState_t(PointAndDir_t(p, d), newstate));
  }
}

void PrintTracks()
{
  for (auto &key : gVolumeTrackMap) {
    std::cerr << key.first->GetName() << " " << key.second->size() << "\n";
  }
}

void BenchNavigationUsingLoggedTracks(LogicalVolume const *lvol, std::vector<VNavigator const *> const &navs,
                                      std::vector<TrackAndState_t> const &tracks)
{
  NavigationState *newstate = new NavigationState();
  std::cerr << "lvol " << lvol->GetName() << " CURRENT NAV " << lvol->GetNavigator()->GetName() << "\n";
  int i        = 1000;
  int j        = 0;
  double bestT = vecgeom::kInfLength;

  for (auto &nav : navs) {
    double step = 0;
    Stopwatch timer;
    timer.Start();
    for (size_t rep = 0; rep < 3; rep++) {
      for (auto &t : tracks) {
        step += nav->ComputeStepAndPropagatedState(t.first.first, t.first.second, vecgeom::kInfLength, *t.second,
                                                   *newstate);
      }
    }
    timer.Stop();
    double T = timer.Elapsed();
    if (T < bestT) {
      bestT = T;
      i     = j;
    }
    std::cerr << "lvol " << lvol->GetName() << " NAV " << nav->GetName() << "t: " << T << "i: " << i << "step: " << step
              << "\n";
    j++;
  }
  std::cerr << "SETTING lvol " << lvol->GetName() << " to NAV " << navs[i]->GetName() << "\n";
  const_cast<LogicalVolume *>(lvol)->SetNavigator(navs[i]);
}

void BenchTracks()
{

  for (auto &key : gVolumeTrackMap) {
    std::vector<VNavigator const *> navs = {NewSimpleNavigator<>::Instance(), SimpleABBoxNavigator<>::Instance()};
    navs.push_back(key.first->GetNavigator());
    //  std::cerr << key.first->GetName() << " " << key.second->size() << "\n";
    BenchNavigationUsingLoggedTracks(key.first, navs, *key.second);
  }
}

void InitNavigators()
{
  for (auto &lvol : GeoManager::Instance().GetLogicalVolumesMap()) {
    auto ndaughters = lvol.second->GetDaughtersp()->size();

    if (ndaughters <= 2) {
      lvol.second->SetNavigator(NewSimpleNavigator<>::Instance());
    } else if (ndaughters <= 10) {
      lvol.second->SetNavigator(SimpleABBoxNavigator<>::Instance());
    } else { // ndaughters > 10
      lvol.second->SetNavigator(HybridNavigator<>::Instance());
      HybridManager2::Instance().InitStructure((lvol.second));
    }
  }
}

__attribute__((noinline)) void DeleteROOTVoxels()
{
  std::cout << " IN DELETE VOXEL METHOD \n";
  int counter       = 0;
  TObjArray *volist = gGeoManager->GetListOfVolumes();

  std::cout << " entries " << volist->GetEntries() << "\n";

  for (int i = 0; i < volist->GetEntries(); ++i) {
    TGeoVolume *vol = (TGeoVolume *)volist->At(i);
    if (vol != NULL && vol->GetVoxels() != 0) {
      counter++;
      delete vol->GetVoxels();
      vol->SetVoxelFinder(0);
    }
  }
  std::cout << " deleted " << counter << " Voxels \n";
}

template <bool DoVerbose = false>
void XRayWithROOT(int axis, Vector3D<Precision> origin, Vector3D<Precision> bbox, Vector3D<Precision> dir,
                  double axis1_start, double axis1_end, double axis2_start, double axis2_end, int data_size_x,
                  int data_size_y, double pixel_axis, int *image)
{

  if (DoVerbose) {
    std::cout << "from [" << axis1_start << ";" << axis2_start << "] to [" << axis1_end << ";" << axis2_end << "]\n";
    std::cout << "Xpixels " << data_size_x << " YPixels " << data_size_y << "\n";

    std::cout << pixel_axis << "\n";
  }

  double pixel_width_1 = (axis1_end - axis1_start) / data_size_x;
  double pixel_width_2 = (axis2_end - axis2_start) / data_size_y;

  if (DoVerbose) {
    std::cout << pixel_width_1 << "\n";
    std::cout << pixel_width_2 << "\n";
  }

  size_t zerosteps_accum(0);
  for (int pixel_count_2 = 0; pixel_count_2 < data_size_y; ++pixel_count_2) {
    for (int pixel_count_1 = 0; pixel_count_1 < data_size_x; ++pixel_count_1) {
      double axis1_count = axis1_start + pixel_count_1 * pixel_width_1 + 1E-6;
      double axis2_count = axis2_start + pixel_count_2 * pixel_width_2 + 1E-6;

      if (DoVerbose) {
        std::cout << "\n PIXEL(" << pixel_count_1 << ", " << pixel_count_2 << ")\n";
      }
      // set start point of XRay
      Vector3D<Precision> p(GetStartPoint(origin, bbox, axis1_count, axis2_count, axis));

      TGeoNavigator *nav = gGeoManager->GetCurrentNavigator();
      nav->SetCurrentPoint(p.x(), p.y(), p.z());
      nav->SetCurrentDirection(dir.x(), dir.y(), dir.z());

      double distancetravelled  = 0.;
      int crossedvolumecount    = 0;
      double accumulateddensity = 0.;

      // avoid set but unused compiler warnings
      (void)distancetravelled;
      (void)crossedvolumecount;
      (void)accumulateddensity;

      if (DoVerbose) {
        std::cout << " StartPoint(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
        std::cout << " Direction <" << dir[0] << ", " << dir[1] << ", " << dir[2] << ">" << std::endl;
      }

      // propagate until we leave detector
      TGeoNode const *node       = nav->FindNode();
      TGeoMaterial const *curmat = node->GetVolume()->GetMaterial();
      std::string currnodename;
      if (DoVerbose) {
        currnodename = (node) ? node->GetName() : "NULL";
      }

      //  std::cout << pixel_count_1 << " " << pixel_count_2 << " " << dir << "\t" << p << "\t";
      //  std::cout << "IN|OUT" << nav->IsOutside() << "\n";
      //  if( node ) std::cout <<    node->GetVolume()->GetName() << "\t";
      while (node != NULL) {
        node = nav->FindNextBoundaryAndStep(vecgeom::kInfLength);
        distancetravelled += nav->GetStep();
        if (nav->GetStep() < 0.) {
          std::cout << "NEGATIVE STEP IN ROOT DETECTED ... ABORTING PIXEL\n";
          node = nullptr;
          continue;
        }
        accumulateddensity += curmat->GetDensity() * distancetravelled;

        if (DoVerbose) {
          std::string nextnodename;
          nextnodename = (node) ? node->GetVolume()->GetName() : "NULL";
          std::cout << "R FROM p(" << p[0] << ", " << p[1] << ", " << p[2] << ") ; " << currnodename << "  -->  "
                    << nextnodename << " with step [" << nav->GetStep() << "]\n";
          currnodename = nextnodename;
        }
        // Increase passed_volume
        // TODO: correct counting of travel in "world" bounding box
        if (nav->GetStep() > kZeroStepLimit) {
          crossedvolumecount++;
        } else {
          zerosteps_accum++;
        }
        curmat = (node != 0) ? node->GetVolume()->GetMaterial() : 0;
      } // end while
      // std::cout << crossedvolumecount << "\n";

      ///////////////////////////////////
      // Store the number of passed volume at 'volume_result'
      *(image + pixel_count_2 * data_size_x + pixel_count_1) =
          crossedvolumecount; // accumulateddensity ;// crossedvolumecount;

      if (DoVerbose) {
        std::cout << "PIXEL STEP ROOT:(" << pixel_count_1 << ", " << pixel_count_2 << ") : " << crossedvolumecount
                  << "\n";
      }
    } // end inner loop
  } // end outer loop
  std::cout << "ZERO STEPS ROOT " << zerosteps_accum << "\n";
} // end XRayWithROOT

template <bool DoVerbose = false>
void XRayWithVecGeom_PolymorphicNavigationFramework(int axis, Vector3D<Precision> origin, Vector3D<Precision> bbox,
                                                    Vector3D<Precision> dir, double axis1_start, double axis1_end,
                                                    double axis2_start, double axis2_end, int data_size_x,
                                                    int data_size_y, double pixel_axis, int *image)
{

  if (DoVerbose) {
    std::cout << "from [" << axis1_start << ";" << axis2_start << "] to [" << axis1_end << ";" << axis2_end << "]\n";
    std::cout << "Xpixels " << data_size_x << " YPixels " << data_size_y << "\n";

    std::cout << pixel_axis << "\n";
  }
  double pixel_width_1 = (axis1_end - axis1_start) / data_size_x;
  double pixel_width_2 = (axis2_end - axis2_start) / data_size_y;

  if (DoVerbose) {
    std::cout << pixel_width_1 << "\n";
    std::cout << pixel_width_2 << "\n";
  }

  NavigationState *newnavstate = new NavigationState();
  NavigationState *curnavstate = new NavigationState();

  size_t zerosteps(0);
  size_t zerosteps_accum(0);
  auto world = GeoManager::Instance().GetWorld();
  for (int pixel_count_2 = 0; pixel_count_2 < data_size_y; ++pixel_count_2) {
    for (int pixel_count_1 = 0; pixel_count_1 < data_size_x; ++pixel_count_1) {
      double axis1_count = axis1_start + pixel_count_1 * pixel_width_1 + 1E-6;
      double axis2_count = axis2_start + pixel_count_2 * pixel_width_2 + 1E-6;

      if (DoVerbose) {
        std::cout << "\n PIXEL(" << pixel_count_1 << ", " << pixel_count_2 << ")\n";
      }

      // set start point of XRay
      Vector3D<Precision> p(GetStartPoint(origin, bbox, axis1_count, axis2_count, axis));

      curnavstate->Clear();
      GlobalLocator::LocateGlobalPoint(world, p, *curnavstate, true);

#ifdef VECGEOM_DISTANCE_DEBUG
      gGeoManager->GetCurrentNavigator()->FindNode(p.x(), p.y(), p.z());
#endif
      double distancetravelled = 0.;
      int crossedvolumecount   = 0;

      // avoid set but unused compiler warnings
      (void)distancetravelled;
      (void)crossedvolumecount;

      std::string currnodename;
      if (DoVerbose) {
        std::cout << " StartPoint(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
        std::cout << " Direction <" << dir[0] << ", " << dir[1] << ", " << dir[2] << ">" << std::endl;
        currnodename = (curnavstate->Top()) ? curnavstate->Top()->GetName() : "NULL";
      }

      zerosteps = 0;
      while (!curnavstate->IsOutside()) {
#ifdef LOGDATA
        AddTrack(curnavstate->Top()->GetLogicalVolume(), p, dir, curnavstate);
#endif
        double step = 0;
        newnavstate->Clear();
        VNavigator const *navigator = curnavstate->Top()->GetLogicalVolume()->GetNavigator();
        step = navigator->ComputeStepAndPropagatedState(p, dir, vecgeom::kInfLength, *curnavstate, *newnavstate);

        distancetravelled += step;
        if (DoVerbose) {
          std::string nextnodename((newnavstate->Top()) ? newnavstate->Top()->GetName() : "NULL");
          std::cout << "VG FROM p(" << p[0] << ", " << p[1] << ", " << p[2] << ") ; " << currnodename << "  -->  "
                    << nextnodename << " with step [" << step << "]\n";
          currnodename = nextnodename;
        }
        // here we have to propagate particle ourselves and adjust navigation state
        p = p + dir * (step + 1E-6);

        // pointer swap should do
        newnavstate->CopyTo(curnavstate);

        // Increase passed_volume
        // TODO: correct counting of travel in "world" bounding box
        if (step > kZeroStepLimit) {
          crossedvolumecount++;
        } else {
          zerosteps++;
          if (zerosteps > 1000) {
            std::cerr << "Too many zero steps in VecGeom .. Aborting\n";
            return;
          }
        }
        zerosteps_accum += zerosteps;
      } // end while

      ///////////////////////////////////
      // Store the number of passed volume at 'volume_result'
      *(image + pixel_count_2 * data_size_x + pixel_count_1) = crossedvolumecount;

      if (DoVerbose) {
        std::cout << "PIXEL STEP VGM:(" << pixel_count_1 << ", " << pixel_count_2 << ") : " << crossedvolumecount
                  << "\n";
      }

    } // end inner loop
  } // end outer loop
  std::cout << "ZERO STEPS VG " << zerosteps_accum << "\n";

  delete curnavstate;
  delete newnavstate;

} // end XRayWithVecGeom

// performs the XRay scan using Geant4
#ifdef VECGEOM_GEANT4
template <bool DoVerbose = false>
int XRayWithGeant4(G4VPhysicalVolume *world /* the detector to scan */, int axis, Vector3D<Precision> origin,
                   Vector3D<Precision> bboxscreen, Vector3D<Precision> dir, double axis1_start, double axis1_end,
                   double axis2_start, double axis2_end, int data_size_x, int data_size_y, double pixel_axis,
                   int *image)
{

  // ATTENTION: THERE IS A (OR MIGHT BE) UNIT MISMATCH HERE BETWEEN ROOT AND GEANT
  // ROOT = cm and GEANT4 = mm; basically a factor of 10 in all dimensions

  const double UNITCONV = 10.;
  G4Navigator *nav      = new G4Navigator();

  // now start XRay procedure
  nav->SetWorldVolume(world);

  double pixel_width_1 = (axis1_end - axis1_start) / data_size_x;
  double pixel_width_2 = (axis2_end - axis2_start) / data_size_y;

  size_t zerosteps_accum(0);
  G4ThreeVector d(dir.x(), dir.y(), dir.z());
  for (int pixel_count_2 = 0; pixel_count_2 < data_size_y; ++pixel_count_2) {
    for (int pixel_count_1 = 0; pixel_count_1 < data_size_x; ++pixel_count_1) {
      double axis1_count = axis1_start + pixel_count_1 * pixel_width_1 + 1E-6;
      double axis2_count = axis2_start + pixel_count_2 * pixel_width_2 + 1E-6;

      if (DoVerbose) {
        std::cout << "\n PIXEL(" << pixel_count_1 << ", " << pixel_count_2 << ")\n";
      }

      // set start point of XRay
      G4ThreeVector p;
      if (axis == 1)
        p = UNITCONV * G4ThreeVector(origin[0] - bboxscreen[0], axis1_count, axis2_count);
      else if (axis == 2)
        p = UNITCONV * G4ThreeVector(axis1_count, origin[1] - bboxscreen[1], axis2_count);
      else if (axis == 3)
        p = UNITCONV * G4ThreeVector(axis1_count, axis2_count, origin[2] - bboxscreen[2]);

      G4ThreeVector s(p);
      // false == locate from top
      G4VPhysicalVolume const *vol = nav->LocateGlobalPointAndSetup(p, &d, false);
      std::string currnodename;
      if (DoVerbose) {
        std::cout << " StartPoint(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
        std::cout << " Direction <" << dir[0] << ", " << dir[1] << ", " << dir[2] << ">" << std::endl;
        currnodename = (vol) ? vol->GetName() : "NULL";
      }

      int crossedvolumecount = 0;
      while (vol != NULL) {
        double safety;
        // do one step ( this will internally adjust the current point and so on )
        // also calculates safety

        double step = nav->ComputeStep(p, d, vecgeom::kInfLength, safety);
        if (step > kZeroStepLimit) {
          crossedvolumecount++;
        } else {
          zerosteps_accum++;
        }

        // calculate next point ( do transportation ) and volume ( should go across boundary )
        G4ThreeVector next = p + (step + 1E-6) * d;
        nav->SetGeometricallyLimitedStep();
        vol = nav->LocateGlobalPointAndSetup(next, &d, true);

        if (DoVerbose) {
          std::string nextnodename((vol) ? vol->GetName() : "NULL");
          std::cout << "G4 FROM p(" << p[0] << ", " << p[1] << ", " << p[2] << ") ; " << currnodename << "  -->  "
                    << nextnodename << " with step [" << step << "]\n";
          currnodename = nextnodename;
        }
        p = next;
      }
      //                ///////////////////////////////////
      //                // Store the number of passed volume at 'volume_result'
      *(image + pixel_count_2 * data_size_x + pixel_count_1) = crossedvolumecount;

      if (DoVerbose) {
        std::cout << "PIXEL STEP G4:(" << pixel_count_1 << ", " << pixel_count_2 << ") : " << crossedvolumecount
                  << "\n";
      }

    } // end inner loop
  } // end outer loop
  std::cout << "ZERO STEPS G4 " << zerosteps_accum << "\n";
  return 0;
}
#endif

// a function allowing to clip geometry branches deeper than a certain level from the given volume
void DeleteAllNodesDeeperThan(TGeoVolume *vol, unsigned int level)
{
  if (level == 0) {
    std::cerr << " deleting daughters " << vol->GetNdaughters() << "\n";
    // deletes daughters of this volume
    vol->SetNodes(nullptr);
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
int main(int argc, char *argv[])
{
  int axis = 0;

  double axis1_start = 0.;
  double axis1_end   = 0.;

  double axis2_start = 0.;
  double axis2_end   = 0.;

  double pixel_width = 0;
  double pixel_axis  = 1.;

  if (argc < 5) {
    std::cerr << std::endl;
    std::cerr << "Need to give rootfile, volumename, axis and number of axis" << std::endl;
    std::cerr << "USAGE : ./XRayBenchmarkFromROOTFile [rootfile] [VolumeName] [ViewDirection(Axis)] "
              << "[PixelWidth(OutputImageSize)] [--novoxel(Default:voxel)]" << std::endl;
    std::cerr << "  ex) ./XRayBenchmarkFromROOTFile cms2015.root BSCTrap y 95" << std::endl;
    std::cerr << "      ./XRayBenchmarkFromROOTFile cms2015.root PLT z 500 --vecgeom --novoxel" << std::endl
              << std::endl;
    return 1;
  }

  TGeoManager::Import(argv[1]);
  std::string testvolume(argv[2]);

  if (strcmp(argv[3], "x") == 0)
    axis = 1;
  else if (strcmp(argv[3], "y") == 0)
    axis = 2;
  else if (strcmp(argv[3], "z") == 0)
    axis = 3;
  else {
    std::cerr << "Incorrect axis" << std::endl << std::endl;
    return 1;
  }

  pixel_width = atof(argv[4]);

  unsigned int cutatlevel = 1000;
  bool cutlevel           = false;
  std::string imagePrefix{};
  for (auto i = 5; i < argc; i++) {
    if (!strcmp(argv[i], "--novoxel")) voxelize = false;
    if (!strcmp(argv[i], "--noassembly")) assemblies = false;
    if (!strcmp(argv[i], "--trackverbose")) trackverbose = true;
    if (!strcmp(argv[i], "--tolevel")) {
      cutlevel   = true;
      cutatlevel = atoi(argv[i + 1]);
      std::cout << "Cutting geometry at level " << cutatlevel << "\n";
    }
    if (!strcmp(argv[i], "--prefix")) {
      imagePrefix = std::string(argv[i + 1]);
      std::cout << "imagePrefix set to " << imagePrefix << "\n";
    }
    if (!strcmp(argv[i], "--zerosteplimit")) {
      kZeroStepLimit = atof(argv[i + 1]);
      std::cout << "Setting zero step limit to " << kZeroStepLimit << "\n";
    }
  }

  int found               = 0;
  TGeoVolume *foundvolume = NULL;
  // now try to find shape with logical volume name given on the command line
  TObjArray *vlist = gGeoManager->GetListOfVolumes();
  for (auto i = 0; i < vlist->GetEntries(); ++i) {
    TGeoVolume *vol = reinterpret_cast<TGeoVolume *>(vlist->At(i));
    std::string fullname(vol->GetName());

    std::size_t founds = fullname.compare(testvolume);
    if (founds == 0) {
      found++;
      foundvolume = vol;

      std::cerr << "(" << i << ")found matching volume " << foundvolume->GetName() << " of type "
                << foundvolume->GetShape()->ClassName() << "\n";
    }
  }

  std::cerr << "volume found " << found << " times \n\n";

  // if volume not found take world
  if (!foundvolume) {
    std::cerr << "specified volume not found; xraying complete detector\n";
    foundvolume = gGeoManager->GetTopVolume();
  }

  if (foundvolume) {
    foundvolume->GetShape()->InspectShape();
    std::cerr << "volume capacity " << foundvolume->GetShape()->Capacity() << "\n";

    // get bounding box to generate x-ray start positions
    double dx        = ((TGeoBBox *)foundvolume->GetShape())->GetDX() * 1.05;
    double dy        = ((TGeoBBox *)foundvolume->GetShape())->GetDY() * 1.05;
    double dz        = ((TGeoBBox *)foundvolume->GetShape())->GetDZ() * 1.05;
    double origin[3] = {
        0.,
    };
    origin[0] = ((TGeoBBox *)foundvolume->GetShape())->GetOrigin()[0];
    origin[1] = ((TGeoBBox *)foundvolume->GetShape())->GetOrigin()[1];
    origin[2] = ((TGeoBBox *)foundvolume->GetShape())->GetOrigin()[2];

    // TGeoMaterial *matAl = new TGeoMaterial("Al", 0, 0, 0);
    // TGeoMedium *vac         = new TGeoMedium("Al", 1, matAl);
    TGeoMedium *medium = gGeoManager->GetTopVolume()->GetMedium();

    TGeoVolume *boundingbox = gGeoManager->MakeBox("BoundingBox", medium, std::abs(origin[0]) + dx,
                                                   std::abs(origin[1]) + dy, std::abs(origin[2]) + dz);

    // TGeoManager * geom = boundingbox->GetGeoManager();
    std::cout << gGeoManager->CountNodes() << "\n";

    if (!voxelize) DeleteROOTVoxels();

    auto materiallist = gGeoManager->GetListOfMaterials();
    auto volumelist   = gGeoManager->GetListOfVolumes();

    gGeoManager = nullptr;

    TGeoManager *mgr2 = new TGeoManager();
    // fix materials first of all
    for (int i = 0; i < materiallist->GetEntries(); ++i) {
      mgr2->AddMaterial((TGeoMaterial *)materiallist->At(i));
    }
    for (int i = 0; i < volumelist->GetEntries(); ++i) {
      mgr2->AddVolume((TGeoVolume *)volumelist->At(i));
    }

    // do some surgery
    // delete foundvolume->GetNodes();
    // foundvolume->SetNodes(nullptr);
    if (cutlevel) {
      DeleteAllNodesDeeperThan(foundvolume, cutatlevel);
    }
    //    delete gGeoManager;
    //    gGeoManager = new TGeoManager();
    boundingbox->AddNode(foundvolume, 1);
    mgr2->SetTopVolume(boundingbox);
    mgr2->CloseGeometry();
    gGeoManager = mgr2;
    gGeoManager->CloseGeometry();
    gGeoManager->Export("DebugGeom.root");

    // FixVolumeList

    mgr2->GetTopNode()->GetMatrix()->Print();

    std::cout << gGeoManager->CountNodes() << "\n";
    // delete world->GetVoxels();
    // world->SetVoxelFinder(0);

    std::cout << std::endl;
    std::cout << "BoundingBoxDX: " << dx << std::endl;
    std::cout << "BoundingBoxDY: " << dy << std::endl;
    std::cout << "BoundingBoxDZ: " << dz << std::endl;

    std::cout << std::endl;
    std::cout << "BoundingBoxOriginX: " << origin[0] << std::endl;
    std::cout << "BoundingBoxOriginY: " << origin[1] << std::endl;
    std::cout << "BoundingBoxOriginZ: " << origin[2] << std::endl << std::endl;

    Vector3D<Precision> p;
    Vector3D<Precision> dir;

    unsigned long long data_size_x;
    unsigned long long data_size_y;
    do {
      if (axis == 1) {
        dir.Set(1., 0., 0.);
        // Transformation3D trans( 0, 0, 0, 5, 5, 5);
        // trans.Print();
        // dir = trans.TransformDirection( Vector3D<Precision> (1,0,0));

        axis1_start = origin[1] - dy;
        axis1_end   = origin[1] + dy;
        axis2_start = origin[2] - dz;
        axis2_end   = origin[2] + dz;
        pixel_axis  = (dy * 2) / pixel_width;
      } else if (axis == 2) {
        dir.Set(0., 1., 0.);
        // vecgeom::Transformation3D trans( 0, 0, 0, 5, 5, 5);
        // dir = trans.TransformDirection(dir);
        axis1_start = origin[0] - dx;
        axis1_end   = origin[0] + dx;
        axis2_start = origin[2] - dz;
        axis2_end   = origin[2] + dz;
        pixel_axis  = (dx * 2) / pixel_width;
      } else if (axis == 3) {
        dir.Set(0., 0., 1.);
        // vecgeom::Transformation3D trans( 0, 0, 0, 5, 5, 5);
        // dir = trans.TransformDirection(dir);
        axis1_start = origin[0] - dx;
        axis1_end   = origin[0] + dx;
        axis2_start = origin[1] - dy;
        axis2_end   = origin[1] + dy;
        pixel_axis  = (dx * 2) / pixel_width;
      }

      // init data for image
      data_size_x = (axis1_end - axis1_start) / pixel_axis;
      data_size_y = (axis2_end - axis2_start) / pixel_axis;

      if (data_size_x * data_size_y > 1E7L) {
        pixel_width /= 2;
        std::cerr << data_size_x * data_size_y << "\n";
        std::cerr << "warning: image to big " << pixel_width << " " << pixel_axis << "\n";
      } else {
        std::cerr << "size ok " << data_size_x * data_size_y << "\n";
      }
    } while (data_size_x * data_size_y > 1E7L);
    std::cerr << "allocating image" << "\n";
    int *volume_result = (int *)new int[data_size_y * data_size_x * 3];

#ifdef VECGEOM_GEANT4
    int *volume_result_Geant4 = (int *)new int[data_size_y * data_size_x * 3];
#endif
    int *volume_result_VecGeom = (int *)new int[data_size_y * data_size_x * 3];

    Stopwatch timer;
    timer.Start();
#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif
    if (trackverbose) {
      XRayWithROOT<true>(axis, Vector3D<Precision>(origin[0], origin[1], origin[2]), Vector3D<Precision>(dx, dy, dz),
                         dir, axis1_start, axis1_end, axis2_start, axis2_end, data_size_x, data_size_y, pixel_axis,
                         volume_result);
    } else {
      XRayWithROOT<false>(axis, Vector3D<Precision>(origin[0], origin[1], origin[2]), Vector3D<Precision>(dx, dy, dz),
                          dir, axis1_start, axis1_end, axis2_start, axis2_end, data_size_x, data_size_y, pixel_axis,
                          volume_result);
    }

#ifdef CALLGRIND
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif
    timer.Stop();

    std::cout << std::endl;
    std::cout << " ROOT Elapsed time : " << timer.Elapsed() << std::endl;

    // Make bitmap file; generate filename
    std::stringstream imagenamebase;
    imagenamebase << (imagePrefix.size() ? imagePrefix : "volumeImage_") << testvolume;
    if (axis == 1) imagenamebase << "x";
    if (axis == 2) imagenamebase << "y";
    if (axis == 3) imagenamebase << "z";
    if (voxelize) imagenamebase << "_VOXELIZED_";
    std::stringstream ROOTimage;
    ROOTimage << imagenamebase.str();
    ROOTimage << "_ROOT.bmp";
    make_bmp(volume_result, ROOTimage.str().c_str(), data_size_x, data_size_y);

#ifdef VECGEOM_GEANT4
    if (getenv("XRAY_GEANT4")) {
      // int errorROOTG4(0);
      G4VPhysicalVolume *world(vecgeom::G4GeoManager::Instance().GetG4GeometryFromROOT());
      if (world != nullptr) G4GeoManager::Instance().LoadG4Geometry(world);

      timer.Start();
      if (world != nullptr) {
        if (trackverbose) {
          XRayWithGeant4<true>(world, axis, Vector3D<Precision>(origin[0], origin[1], origin[2]),
                               Vector3D<Precision>(dx, dy, dz), dir, axis1_start, axis1_end, axis2_start, axis2_end,
                               data_size_x, data_size_y, pixel_axis, volume_result_Geant4);
        } else {
          XRayWithGeant4<false>(world, axis, Vector3D<Precision>(origin[0], origin[1], origin[2]),
                                Vector3D<Precision>(dx, dy, dz), dir, axis1_start, axis1_end, axis2_start, axis2_end,
                                data_size_x, data_size_y, pixel_axis, volume_result_Geant4);
        }
      }
      timer.Stop();

      std::stringstream G4image;
      G4image << imagenamebase.str();
      G4image << "_Geant4.bmp";
      make_bmp(volume_result_Geant4, G4image.str().c_str(), data_size_x, data_size_y);
      std::cout << std::endl;
      std::cout << " Geant4 Elapsed time : " << timer.Elapsed() << std::endl;
      std::stringstream G4diffimage;
      G4diffimage << imagenamebase.str();
      G4diffimage << "_diffROOTG4.bmp";
      // errorROOTG4 =
      make_diff_bmp(volume_result, volume_result_Geant4, G4diffimage.str().c_str(), data_size_x, data_size_y);
    }
#endif

    // convert current gGeoManager to a VecGeom geometry
    // RootGeoManager::Instance().set_verbose(true);
    RootGeoManager::Instance().SetFlattenAssemblies(!assemblies);
    RootGeoManager::Instance().LoadRootGeometry();
    std::cout << "Detector loaded " << "\n";
    ABBoxManager<Precision>::Instance().InitABBoxesForCompleteGeometry();
    std::cout << "voxelized " << "\n";

    // gGeoManager->Export("extractedgeom.root");
    // GeoManager::Instance().GetWorld()->PrintContent();
    std::cerr << "total number of nodes " << GeoManager::Instance().GetWorld()->GetLogicalVolume()->GetNTotal() << "\n";

    InitNavigators();

    timer.Start();
#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif
    if (trackverbose) {
      XRayWithVecGeom_PolymorphicNavigationFramework<true>(
          axis, Vector3D<Precision>(origin[0], origin[1], origin[2]), Vector3D<Precision>(dx, dy, dz), dir, axis1_start,
          axis1_end, axis2_start, axis2_end, data_size_x, data_size_y, pixel_axis, volume_result_VecGeom);
    } else {
      XRayWithVecGeom_PolymorphicNavigationFramework<false>(
          axis, Vector3D<Precision>(origin[0], origin[1], origin[2]), Vector3D<Precision>(dx, dy, dz), dir, axis1_start,
          axis1_end, axis2_start, axis2_end, data_size_x, data_size_y, pixel_axis, volume_result_VecGeom);
    }
#ifdef CALLGRIND
    CALLGRIND_START_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif
    timer.Stop();
    std::cout << " VG (NEW) Elapsed time : " << timer.Elapsed() << std::endl;
    //      PrintTracks();
    BenchTracks();
    std::stringstream VecGeomimage;
    VecGeomimage.str("");
    VecGeomimage.clear();
    VecGeomimage << imagenamebase.str();
    VecGeomimage << "_VecGeomNEW.bmp";
    make_bmp(volume_result_VecGeom, VecGeomimage.str().c_str(), data_size_x, data_size_y);

    std::stringstream VGRdiffimage;
    VGRdiffimage.str("");
    VGRdiffimage.clear();
    VGRdiffimage << imagenamebase.str();
    VGRdiffimage << "_diffROOTVGNEW.bmp";
    int errorROOTVG =
        make_diff_bmp(volume_result, volume_result_VecGeom, VGRdiffimage.str().c_str(), data_size_x, data_size_y);
    int errorG4VG(0);
#ifdef VECGEOM_GEANT4
    std::stringstream VGG4diffimage;
    VGG4diffimage.str("");
    VGG4diffimage.clear();
    VGG4diffimage << imagenamebase.str();
    VGG4diffimage << "_diffG4VGNEW.bmp";
    errorG4VG = make_diff_bmp(volume_result_Geant4, volume_result_VecGeom, VGG4diffimage.str().c_str(), data_size_x,
                              data_size_y);
#endif

    // these are ugly hacks to prevent ROOT from crashing at cleanup
    // the problem is due to the fact that we messed with the gGeoManager (which is not directly foreseen)
    gGeoManager = nullptr;
    mgr2        = nullptr;
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 0, 0)
    ROOT::Internal::gROOTLocal = nullptr;
#else
    gROOT = nullptr;
#endif

    return ((errorROOTVG == 0) || (errorG4VG == 0)) ? 0 : 1;
  }
  return 0;
}

void make_bmp_header(MY_BITMAP *pBitmap, unsigned char *bmpBuf, int sizex, int sizey)
{
  int width_4      = (sizex + 3) & ~3;
  unsigned int len = 0;

  // bitmap file header
  pBitmap->bmpFileHeader.bfType      = 0x4d42;
  pBitmap->bmpFileHeader.bfSize      = sizey * width_4 * 3 + 54;
  pBitmap->bmpFileHeader.bfReserved1 = 0;
  pBitmap->bmpFileHeader.bfReserved2 = 0;
  pBitmap->bmpFileHeader.bfOffBits   = 54;

  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfType, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfSize, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfReserved1, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfReserved2, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfOffBits, 4);
  len += 4;

  // bitmap information header
  pBitmap->bmpInfoHeader.biSize          = 40;
  pBitmap->bmpInfoHeader.biWidth         = width_4;
  pBitmap->bmpInfoHeader.biHeight        = sizey;
  pBitmap->bmpInfoHeader.biPlanes        = 1;
  pBitmap->bmpInfoHeader.biBitCount      = 24;
  pBitmap->bmpInfoHeader.biCompression   = 0;
  pBitmap->bmpInfoHeader.biSizeImage     = sizey * width_4 * 3;
  pBitmap->bmpInfoHeader.biXPelsPerMeter = 0;
  pBitmap->bmpInfoHeader.biYPelsPerMeter = 0;
  pBitmap->bmpInfoHeader.biClrUsed       = 0;
  pBitmap->bmpInfoHeader.biClrImportant  = 0;

  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biSize, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biWidth, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biHeight, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biPlanes, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biBitCount, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biCompression, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biSizeImage, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biXPelsPerMeter, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biYPelsPerMeter, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biClrUsed, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biClrImportant, 4);
  len += 4;
}

int make_bmp(int const *volume_result, char const *name, int data_size_x, int data_size_y, bool linear)
{

  MY_BITMAP *pBitmap = new MY_BITMAP;
  FILE *pBitmapFile;
  int width_4 = (data_size_x + 3) & ~3;
  unsigned char *bmpBuf;

  bmpBuf = (unsigned char *)new unsigned char[data_size_y * width_4 * 3 + 54];
  printf("\n Write bitmap...\n");

  unsigned int len = 0;

  // bitmap file header
  pBitmap->bmpFileHeader.bfType      = 0x4d42;
  pBitmap->bmpFileHeader.bfSize      = data_size_y * width_4 * 3 + 54;
  pBitmap->bmpFileHeader.bfReserved1 = 0;
  pBitmap->bmpFileHeader.bfReserved2 = 0;
  pBitmap->bmpFileHeader.bfOffBits   = 54;

  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfType, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfSize, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfReserved1, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfReserved2, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfOffBits, 4);
  len += 4;

  // bitmap information header
  pBitmap->bmpInfoHeader.biSize          = 40;
  pBitmap->bmpInfoHeader.biWidth         = width_4;
  pBitmap->bmpInfoHeader.biHeight        = data_size_y;
  pBitmap->bmpInfoHeader.biPlanes        = 1;
  pBitmap->bmpInfoHeader.biBitCount      = 24;
  pBitmap->bmpInfoHeader.biCompression   = 0;
  pBitmap->bmpInfoHeader.biSizeImage     = data_size_y * width_4 * 3;
  pBitmap->bmpInfoHeader.biXPelsPerMeter = 0;
  pBitmap->bmpInfoHeader.biYPelsPerMeter = 0;
  pBitmap->bmpInfoHeader.biClrUsed       = 0;
  pBitmap->bmpInfoHeader.biClrImportant  = 0;

  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biSize, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biWidth, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biHeight, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biPlanes, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biBitCount, 2);
  len += 2;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biCompression, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biSizeImage, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biXPelsPerMeter, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biYPelsPerMeter, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biClrUsed, 4);
  len += 4;
  memcpy(bmpBuf + len, &pBitmap->bmpInfoHeader.biClrImportant, 4);
  len += 4;

  // find out maxcount before doing the picture
  int maxcount = 0;
  int x = 0, y = 0, origin_x = 0;
  while (y < data_size_y) {
    while (origin_x < data_size_x) {
      int value = *(volume_result + y * data_size_x + origin_x);
      maxcount  = (value > maxcount) ? value : maxcount;

      x++;
      origin_x++;
    }
    y++;
    x        = 0;
    origin_x = 0;
  }
  //  maxcount = std::log(maxcount + 1);

  x        = 0;
  y        = 0;
  origin_x = 0;

  int padding            = width_4 - data_size_x;
  int padding_idx        = padding;
  unsigned char *imgdata = (unsigned char *)new unsigned char[data_size_y * width_4 * 3];

  int totalcount = 0;

  while (y < data_size_y) {
    while (origin_x < data_size_x) {
      int value = *(volume_result + y * data_size_x + origin_x);
      totalcount += value;

      //*(imgdata+y*width_4*3+x*3+0)= (value *50) % 256;
      //*(imgdata+y*width_4*3+x*3+1)= (value *40) % 256;
      //*(imgdata+y*width_4*3+x*3+2)= (value *30) % 256;

      //*(imgdata+y*width_4*3+x*3+0)= (std::log(value)/(1.*maxcount)) * 256;
      //*(imgdata+y*width_4*3+x*3+1)= (std::log(value)/(1.2*maxcount)) * 256;
      //*(imgdata+y*width_4*3+x*3+2)= (std::log(value)/(1.4*maxcount)) * 256;
      if (linear) {
        *(imgdata + y * width_4 * 3 + x * 3 + 0) = (value / (1. * maxcount)) * 256;
        *(imgdata + y * width_4 * 3 + x * 3 + 1) = (value / (1. * maxcount)) * 256;
        *(imgdata + y * width_4 * 3 + x * 3 + 2) = (value / (1. * maxcount)) * 256;
      } else {
        *(imgdata + y * width_4 * 3 + x * 3 + 0) = (log(value + 1)) / (1. * (log(1 + maxcount))) * 256;
        *(imgdata + y * width_4 * 3 + x * 3 + 1) = (log(value + 1)) / (1. * (log(1 + maxcount))) * 256;
        *(imgdata + y * width_4 * 3 + x * 3 + 2) = (log(value + 1)) / (1. * (log(1 + maxcount))) * 256;
      }
      x++;
      origin_x++;

      while (origin_x == data_size_x && padding_idx) {
        // padding 4-byte at bitmap image
        *(imgdata + y * width_4 * 3 + x * 3 + 0) = 0;
        *(imgdata + y * width_4 * 3 + x * 3 + 1) = 0;
        *(imgdata + y * width_4 * 3 + x * 3 + 2) = 0;
        x++;
        padding_idx--;
      }
      padding_idx = padding;
    }
    y++;
    x        = 0;
    origin_x = 0;
  }

  memcpy(bmpBuf + 54, imgdata, width_4 * data_size_y * 3);

  pBitmapFile = fopen(name, "wb");
  fwrite(bmpBuf, sizeof(char), width_4 * data_size_y * 3 + 54, pBitmapFile);

  fclose(pBitmapFile);
  delete[] imgdata;
  delete[] bmpBuf;
  delete pBitmap;

  std::cout << " wrote image file " << name << "\n";
  std::cout << " total count " << totalcount << "\n";
  std::cout << " max count " << maxcount << "\n";
  return 0;
}

int make_diff_bmp(int const *image1, int const *image2, char const *name, int data_size_x, int data_size_y)
{

  MY_BITMAP *pBitmap = new MY_BITMAP;
  FILE *pBitmapFile;
  int width_4           = (data_size_x + 3) & ~3;
  unsigned char *bmpBuf = (unsigned char *)new unsigned char[data_size_y * width_4 * 3 + 54];

  // init buffer and write header
  make_bmp_header(pBitmap, bmpBuf, data_size_x, data_size_y);

  // TODO: verify the 2 images have same dimensions

  // find out maxcount before doing the picture
  int maxdiff = 0;
  int mindiff = 0;
  int x = 0, y = 0, origin_x = 0;
  size_t differentpixels = 0;
  size_t samepixels      = 0;
  while (y < data_size_y) {
    while (origin_x < data_size_x) {
      int value = *(image1 + y * data_size_x + origin_x) - *(image2 + y * data_size_x + origin_x);
      maxdiff   = (value > maxdiff) ? value : maxdiff;
      mindiff   = (value < mindiff) ? value : mindiff;
      x++;
      origin_x++;
      if (value == 0) {
        samepixels++;
      } else {
        differentpixels++;
      }
    }
    y++;
    x        = 0;
    origin_x = 0;
  }

  x        = 0;
  y        = 0;
  origin_x = 0;

  int padding            = width_4 - data_size_x;
  int padding_idx        = padding;
  unsigned char *imgdata = (unsigned char *)new unsigned char[data_size_y * width_4 * 3];

  while (y < data_size_y) {
    while (origin_x < data_size_x) {
      int value = *(image1 + y * data_size_x + origin_x) - *(image2 + y * data_size_x + origin_x);

      if (value >= 0) {
        *(imgdata + y * width_4 * 3 + x * 3 + 0) = 255 - (value / (1. * maxdiff)) * 255;
        *(imgdata + y * width_4 * 3 + x * 3 + 1) = 255 - 0; // (value/(1.*maxcount)) * 256;
        *(imgdata + y * width_4 * 3 + x * 3 + 2) = 255 - 0; //(value/(1.*maxcount)) * 256;}
      } else {
        *(imgdata + y * width_4 * 3 + x * 3 + 0) = 255 - 0;
        *(imgdata + y * width_4 * 3 + x * 3 + 1) = 255 - 0;                              // (value/(1.*maxcount)) * 255;
        *(imgdata + y * width_4 * 3 + x * 3 + 2) = 255 - (value / (1. * mindiff)) * 255; //(value/(1.*maxcount)) * 255;}
      }
      x++;
      origin_x++;

      while (origin_x == data_size_x && padding_idx) {
        // padding 4-byte at bitmap image
        *(imgdata + y * width_4 * 3 + x * 3 + 0) = 0;
        *(imgdata + y * width_4 * 3 + x * 3 + 1) = 0;
        *(imgdata + y * width_4 * 3 + x * 3 + 2) = 0;
        x++;
        padding_idx--;
      }
      padding_idx = padding;
    }
    y++;
    x        = 0;
    origin_x = 0;
  }

  memcpy(bmpBuf + 54, imgdata, width_4 * data_size_y * 3);
  pBitmapFile = fopen(name, "wb");
  fwrite(bmpBuf, sizeof(char), width_4 * data_size_y * 3 + 54, pBitmapFile);

  fclose(pBitmapFile);
  delete[] imgdata;
  delete[] bmpBuf;
  delete pBitmap;

  std::cout << " wrote image file " << name << "\n";
  auto diffpercentage = differentpixels / (1. * (samepixels + differentpixels));
  std::cout << " Different pixel percentage " << diffpercentage << "\n";
  return (diffpercentage < 0.01) ? 0 : 1;
}
