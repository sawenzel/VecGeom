/*
 * CppExporter.cpp
 *
 *  Created on: 23.03.2015
 *      Author: swenzel
 */

#include "VecGeom/management/CppExporter.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Array.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/PlacedBooleanVolume.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/volumes/UnplacedTube.h"
#include "VecGeom/volumes/UnplacedCutTube.h"
#include "VecGeom/volumes/UnplacedCone.h"
#include "VecGeom/volumes/UnplacedTrapezoid.h"
#include "VecGeom/volumes/UnplacedTorus2.h"
#include "VecGeom/volumes/UnplacedPolycone.h"
#include "VecGeom/volumes/UnplacedPolyhedron.h"
#include "VecGeom/volumes/UnplacedParallelepiped.h"
#include "VecGeom/volumes/UnplacedTrd.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include "VecGeom/volumes/ScaledShape.h"
#include "VecGeom/volumes/GenTrap.h"
#include <sstream>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <list>
#include <vector>
#include <iostream>
#include <iomanip>

namespace vecgeom {
inline namespace cxx {

template <typename IterableContainer, typename ElementType>
bool ContainerContains(IterableContainer const &c, ElementType const &e)
{
  return std::find(c.cbegin(), c.cend(), e) != c.cend();
}

// this function should live in GeoManager and to be used by various exporters
// function returns
// a sorted list of logical volumes -- if some logical volume A depends on another volume B
// (as for boolean volumes); then B
// should come in that list before A
// a list of transformations
void GeomCppExporter::ScanGeometry(VPlacedVolume const *const volume, std::list<LogicalVolume const *> &lvlist,
                                   std::list<LogicalVolume const *> &boollvlist,
                                   std::list<Transformation3D const *> &tlist)
{
  // if not yet treated
  if (std::find(lvlist.cbegin(), lvlist.cend(), volume->GetLogicalVolume()) == lvlist.cend() &&
      std::find(boollvlist.cbegin(), boollvlist.cend(), volume->GetLogicalVolume()) == boollvlist.cend()) {
    if (auto v = dynamic_cast<PlacedBooleanVolume<kUnion> const *>(volume)) {
      boollvlist.push_front(volume->GetLogicalVolume());
      ScanGeometry(v->GetUnplacedVolume()->GetLeft(), lvlist, boollvlist, tlist);
      ScanGeometry(v->GetUnplacedVolume()->GetRight(), lvlist, boollvlist, tlist);
    } else if (auto v = dynamic_cast<PlacedBooleanVolume<kIntersection> const *>(volume)) {
      boollvlist.push_front(volume->GetLogicalVolume());
      ScanGeometry(v->GetUnplacedVolume()->GetLeft(), lvlist, boollvlist, tlist);
      ScanGeometry(v->GetUnplacedVolume()->GetRight(), lvlist, boollvlist, tlist);
    } else if (auto v = dynamic_cast<PlacedBooleanVolume<kSubtraction> const *>(volume)) {
      boollvlist.push_front(volume->GetLogicalVolume());
      ScanGeometry(v->GetUnplacedVolume()->GetLeft(), lvlist, boollvlist, tlist);
      ScanGeometry(v->GetUnplacedVolume()->GetRight(), lvlist, boollvlist, tlist);
    } else if (dynamic_cast<PlacedScaledShape const *>(volume)) {
      boollvlist.push_front(volume->GetLogicalVolume());
      PlacedScaledShape const *v = dynamic_cast<PlacedScaledShape const *>(volume);
      ScanGeometry(v->GetUnplacedVolume()->fScaled.fPlaced, lvlist, boollvlist, tlist);
    } else {
      // ordinary logical volume
      lvlist.push_back(volume->GetLogicalVolume());
    }

    for (size_t d = 0; d < volume->GetDaughters().size(); ++d)
      ScanGeometry(volume->GetDaughters()[d], lvlist, boollvlist, tlist);
  }

  if (std::find(tlist.cbegin(), tlist.cend(), volume->GetTransformation()) == tlist.cend()) {
    tlist.push_back(volume->GetTransformation());
  }
}

void static PushAndReset(std::stringstream &stream, std::vector<std::string> &output)
{
  output.push_back(stream.str());
  stream.str(""); // Remove accumulated information
  stream.clear(); // reset the ios (error) flags.
}

void GeomCppExporter::DumpTransformations(std::vector<std::string> &trafoconstrlist, std::stringstream &trafoexterndecl,
                                          std::vector<std::string> &trafodecllist,
                                          std::list<Transformation3D const *> const &tvlist)
{

  // loop over all transformations
  unsigned int counter = 0;
  for (auto t : tvlist) {
    // register transformation
    if (fTrafoToStringMap.find(t) == fTrafoToStringMap.cend()) {
      // many transformation are identity: we can filter them out and allocate only one
      // identity
      // TODO: such reduction can be applied for other transformations
      if (t->IsIdentity()) {
        fTrafoToStringMap[t] = "idtrans";
      } else {
        // create a variable name
        std::stringstream s;
        s << "transf" << counter;
        // do mapping from existing pointer value to variable name
        fTrafoToStringMap[t] = s.str();
        counter++;
      }
    }
  }

  // we will split the transformation constructions into different groups
  // of compiler translation units for faster and parallel compilation
  unsigned int group = 0;
  std::stringstream trafoconstr;
  std::stringstream trafodecl;

  // generate function that instantiates the transformations
  int groupcounter = 0;
  trafoconstr << "void GenerateTransformations_part" << group << "(){\n";
  bool iddone = false;
  for (auto t : fTrafoToStringMap) {
    Transformation3D const *tp = t.first;
    if (tp->IsIdentity() && iddone) continue;
    if (tp->IsIdentity()) iddone = true;

    // we take a limit if 5000 transformations per translation unit
    // which compiles reasonably fast
    if (++groupcounter > 5000) {
      group++;
      // close old function
      trafoconstr << "}\n";

      // create a new stream
      PushAndReset(trafoconstr, trafoconstrlist);
      PushAndReset(trafodecl, trafodecllist);

      // init new function
      trafoconstr << "void GenerateTransformations_part" << group << "(){\n";

      // reset counter
      groupcounter = 0;
    }

    std::stringstream line;

    // extern declaration line
    trafoexterndecl << "extern Transformation3D *" << t.second << ";\n";
    trafodecl << "Transformation3D * " << t.second << " = nullptr;\n";

    // instantiation line
    line << std::setprecision(15);
    line << t.second << " = new Transformation3D(";
    line << tp->Translation(0) << " , ";
    line << tp->Translation(1) << " , ";
    line << tp->Translation(2);
    if (tp->HasRotation()) {
      line << " , ";
      for (auto i = 0; i < 8; ++i)
        line << tp->Rotation(i) << " , ";
      line << tp->Rotation(8);
    }
    line << ");\n";
    trafoconstr << line.str();
  }
  trafoconstr << "}\n";

  PushAndReset(trafoconstr, trafoconstrlist);
  PushAndReset(trafodecl, trafodecllist);
}

template <typename VectorContainer>
void DumpVector(VectorContainer const &v, std::ostream &dumps)
{
  dumps << "&std::vector<double>{";
  for (int j = 0, n = v.size() - 1; j < n; ++j)
    dumps << v[j] << " , ";
  dumps << v[v.size() - 1] << "}[0]";
}

// function which dumps the logical volumes
void GeomCppExporter::DumpLogicalVolumes(std::ostream &dumps, std::ostream &externdeclarations,
                                         std::ostream &lvoldefinitions, std::list<LogicalVolume const *> const &lvlist)
{

  static unsigned int counter = 0;
  for (auto l : lvlist) {
    // register logical volume
    if (fLVolumeToStringMap.find(l) == fLVolumeToStringMap.cend()) {
      // create a variable name
      std::stringstream s;
      s << "lvol" << counter;
      // do mapping from existing pointer value to variable name
      fLVolumeToStringMap[l] = s.str();
      counter++;
    }
  }

  // generate code that instantiates LogicalVolumes
  for (auto l : lvlist) {
    // Some shapes need to pre-build arrays
    std::stringstream line;
    line << std::setprecision(15);
    if (dynamic_cast<UnplacedGenTrap const *>(l->GetUnplacedVolume())) {
      UnplacedGenTrap const *shape = dynamic_cast<UnplacedGenTrap const *>(l->GetUnplacedVolume());
      line << std::setprecision(15);
      line << "std::vector<Vector3D<Precision> > ";
      line << fLVolumeToStringMap[l] << "_arr;\n";
      for (auto ivert = 0; ivert < 8; ++ivert) {
        Vector3D<Precision> vert = shape->GetVertex(ivert);
        line << fLVolumeToStringMap[l] << "_arr.push_back(Vector3D<Precision>(";
        line << vert.x() << ", " << vert.y() << ", 0.) );\n";
      }
    }
    line << fLVolumeToStringMap[l];
    line << " = new LogicalVolume ( \"" << l->GetLabel() << "\" , ";

    // now we need to distinguish types
    // use here dynamic casting ( alternatives might exist )
    // ******* TREAT THE BOX *********
    if (dynamic_cast<UnplacedBox const *>(l->GetUnplacedVolume())) {
      UnplacedBox const *box = dynamic_cast<UnplacedBox const *>(l->GetUnplacedVolume());

      line << " new UnplacedBox( ";
      line << box->dimensions().x() << " , ";
      line << box->dimensions().y() << " , ";
      line << box->dimensions().z();
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedBox.h");
    }

    // ******* TREAT THE TUBE *********
    else if (dynamic_cast<UnplacedTube const *>(l->GetUnplacedVolume())) {
      UnplacedTube const *shape = dynamic_cast<UnplacedTube const *>(l->GetUnplacedVolume());

      line << " new UnplacedTube( ";
      line << shape->rmin() << " , ";
      line << shape->rmax() << " , ";
      line << shape->z() << " , ";
      line << shape->sphi() << " , ";
      line << shape->dphi();
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedTube.h");
    }

    // ******* TREAT THE CUT TUBE *********
    else if (dynamic_cast<UnplacedCutTube const *>(l->GetUnplacedVolume())) {
      UnplacedCutTube const *shape = dynamic_cast<UnplacedCutTube const *>(l->GetUnplacedVolume());

      line << " new UnplacedCutTube( ";
      line << shape->rmin() << " , ";
      line << shape->rmax() << " , ";
      line << shape->z() << " , ";
      line << shape->sphi() << " , ";
      line << shape->dphi() << " , ";
      line << "Vector3D<Precision>(";
      line << shape->BottomNormal().x() << ", ";
      line << shape->BottomNormal().y() << ", ";
      line << shape->BottomNormal().z() << ") , ";
      line << "Vector3D<Precision>(";
      line << shape->TopNormal().x() << ", ";
      line << shape->TopNormal().y() << ", ";
      line << shape->TopNormal().z() << ") , ";
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedCutTube.h");
    }

    // ******* TREAT THE CONE *********
    else if (dynamic_cast<UnplacedCone const *>(l->GetUnplacedVolume())) {
      UnplacedCone const *shape = dynamic_cast<UnplacedCone const *>(l->GetUnplacedVolume());

      line << " new UnplacedCone( ";
      line << shape->GetRmin1() << " , ";
      line << shape->GetRmax1() << " , ";
      line << shape->GetRmin2() << " , ";
      line << shape->GetRmax2() << " , ";
      line << shape->GetDz() << " , ";
      line << shape->GetSPhi() << " , ";
      line << shape->GetDPhi();
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedCone.h");
    }

    // ******* TREAT THE TRAPEZOID *********
    else if (dynamic_cast<UnplacedTrapezoid const *>(l->GetUnplacedVolume())) {
      UnplacedTrapezoid const *shape = dynamic_cast<UnplacedTrapezoid const *>(l->GetUnplacedVolume());
      line << " new UnplacedTrapezoid( ";

      line << shape->dz() << " , ";
      line << shape->theta() << " , ";
      line << shape->phi() << " , ";
      line << shape->dy1() << " , ";
      line << shape->dx1() << " , ";
      line << shape->dx2() << " , ";
      line << shape->tanAlpha1() << " , ";
      line << shape->dy2() << " , ";
      line << shape->dx3() << " , ";
      line << shape->dx4() << " , ";
      line << shape->tanAlpha2();
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedTrapezoid.h");
    }

    // ******* TREAT THE TORUS 2 **********
    else if (dynamic_cast<UnplacedTorus2 const *>(l->GetUnplacedVolume())) {
      UnplacedTorus2 const *shape = dynamic_cast<UnplacedTorus2 const *>(l->GetUnplacedVolume());

      line << " new UnplacedTorus2( ";
      line << shape->rmin() << " , ";
      line << shape->rmax() << " , ";
      line << shape->rtor() << " , ";
      line << shape->sphi() << " , ";
      line << shape->dphi();
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedTorus2.h");
    }

    // ******* TREAT THE PARALLELEPIPED **********
    else if (dynamic_cast<UnplacedParallelepiped const *>(l->GetUnplacedVolume())) {
      UnplacedParallelepiped const *shape = dynamic_cast<UnplacedParallelepiped const *>(l->GetUnplacedVolume());

      line << " new UnplacedParallelepiped( ";
      line << shape->GetX() << " , ";
      line << shape->GetY() << " , ";
      line << shape->GetZ() << " , ";
      line << shape->GetAlpha() << " , ";
      line << shape->GetTheta() << " , ";
      line << shape->GetPhi();
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedParallelepiped.h");
    }

    // ******* TREAT THE GENERAL TRAP **********
    else if (dynamic_cast<UnplacedGenTrap const *>(l->GetUnplacedVolume())) {
      UnplacedGenTrap const *shape = dynamic_cast<UnplacedGenTrap const *>(l->GetUnplacedVolume());

      line << " new UnplacedGenTrap( &" << fLVolumeToStringMap[l] << "_arr[0],";
      line << shape->GetDZ();
      line << " )";

      fNeededHeaderFiles.insert("volumes/GenTrap.h");
    }

    // ********* TREAT THE PCON **********
    else if (dynamic_cast<UnplacedPolycone const *>(l->GetUnplacedVolume())) {
      UnplacedPolycone const *shape = dynamic_cast<UnplacedPolycone const *>(l->GetUnplacedVolume());

      line << " new UnplacedPolycone( ";
      line << shape->GetStartPhi() << " , ";
      line << shape->GetDeltaPhi() << " , ";

      std::vector<double> rmin, rmax, z;
      // serialize the arrays as temporary std::vector
      shape->GetStruct().ReconstructSectionArrays(z, rmin, rmax);
#ifndef NDEBUG
      for (auto element : rmin) {
        VECGEOM_ASSERT(element >= 0.);
      }
      for (auto element : rmax) {
        VECGEOM_ASSERT(element >= 0.);
      }
#endif
      if (shape->GetNz() != z.size()) {
        VECGEOM_LOG(warning) << "Volume " << l->GetLabel()
                             << " has a mismatch in the number of z-planes (possible duplication)";
      }
      line << z.size() << " , ";

      // put z vector
      DumpVector(z, line);
      line << " ,";
      // put rmin vector
      DumpVector(rmin, line);
      line << " , ";
      // put rmax vector
      DumpVector(rmax, line);
      line << " ) ";

      fNeededHeaderFiles.insert("volumes/UnplacedPolycone.h");
    }

    // ********* TREAT THE PGON **********
    else if (dynamic_cast<UnplacedPolyhedron const *>(l->GetUnplacedVolume())) {
      UnplacedPolyhedron const *shape = dynamic_cast<UnplacedPolyhedron const *>(l->GetUnplacedVolume());
      line << " new UnplacedPolyhedron( ";
      line << shape->GetPhiStart() << " , ";
      line << shape->GetPhiDelta() << " , ";
      line << shape->GetSideCount() << " , ";
      line << shape->GetZSegmentCount() + 1 << " , ";
      //                std::vector<double> rmin, rmax, z;
      //                // serialize the arrays as tempary std::vector
      //                shape->ReconstructSectionArrays( z,rmin,rmax );
      //
      //                if( z.size() != rmax.size() || rmax.size() != rmin.size() ){
      //                    std::cerr << "different vector sizes\n";
      //                    std::cerr << l->GetLabel() << "\n";
      //                }
      //                if( shape->GetZSegmentCount()+1 != z.size() ){
      //                    std::cerr << "problem with dimensions\n";
      //                    std::cerr << l->GetLabel() << "\n";
      //                }
      auto z    = shape->GetZPlanes();
      auto rmin = shape->GetRMin();
      auto rmax = shape->GetRMax();

      // put z vector
      DumpVector(z, line);
      line << " , ";

      // put rmin vector
      DumpVector(rmin, line);
      line << " , ";

      // put rmax vector
      DumpVector(rmax, line);
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedPolyhedron.h");
    }

    // *** BOOLEAN SOLIDS NEED A SPECIAL TREATMENT *** //
    // their constituents are  not already a part of the logical volume list
    else if (auto shape = dynamic_cast<UnplacedBooleanVolume<kUnion> const *>(l->GetUnplacedVolume())) {
      VPlacedVolume const *left  = shape->GetLeft();
      VPlacedVolume const *right = shape->GetRight();

      // CHECK IF THIS BOOLEAN VOLUME DEPENDS ON OTHER BOOLEAN VOLUMES NOT YET DUMPED
      // THIS SOLUTION IS POTENTIALLY SLOW; MIGHT CONSIDER DIFFERENT TYPE OF CONTAINER
      if (!ContainerContains(fListofTreatedLogicalVolumes, left->GetLogicalVolume()) ||
          !ContainerContains(fListofTreatedLogicalVolumes, right->GetLogicalVolume())) {
        // we need to defer the treatment of this logical volume
        fListofDeferredLogicalVolumes.push_back(l);
        continue;
      }
      line << " new UnplacedBooleanVolume( ";
      line << " kUnion ";
      line << " , ";
      // placed versions of left and right volume
      line << fLVolumeToStringMap[left->GetLogicalVolume()] << "->Place( "
           << fTrafoToStringMap[left->GetTransformation()] << " )";
      line << " , ";
      line << fLVolumeToStringMap[right->GetLogicalVolume()] << "->Place( "
           << fTrafoToStringMap[right->GetTransformation()] << " )";
      line << " )";
      fNeededHeaderFiles.insert("volumes/UnplacedBooleanVolume.h");
    }

    else if (auto shape = dynamic_cast<UnplacedBooleanVolume<kIntersection> const *>(l->GetUnplacedVolume())) {
      VPlacedVolume const *left  = shape->GetLeft();
      VPlacedVolume const *right = shape->GetRight();

      // CHECK IF THIS BOOLEAN VOLUME DEPENDS ON OTHER BOOLEAN VOLUMES NOT YET DUMPED
      // THIS SOLUTION IS POTENTIALLY SLOW; MIGHT CONSIDER DIFFERENT TYPE OF CONTAINER
      if (!ContainerContains(fListofTreatedLogicalVolumes, left->GetLogicalVolume()) ||
          !ContainerContains(fListofTreatedLogicalVolumes, right->GetLogicalVolume())) {
        // we need to defer the treatment of this logical volume
        fListofDeferredLogicalVolumes.push_back(l);
        continue;
      }
      line << " new UnplacedBooleanVolume( ";
      line << " kIntersection ";
      line << " , ";
      // placed versions of left and right volume
      line << fLVolumeToStringMap[left->GetLogicalVolume()] << "->Place( "
           << fTrafoToStringMap[left->GetTransformation()] << " )";
      line << " , ";
      line << fLVolumeToStringMap[right->GetLogicalVolume()] << "->Place( "
           << fTrafoToStringMap[right->GetTransformation()] << " )";
      line << " )";
      fNeededHeaderFiles.insert("volumes/UnplacedBooleanVolume.h");
    }

    else if (auto shape = dynamic_cast<UnplacedBooleanVolume<kSubtraction> const *>(l->GetUnplacedVolume())) {
      VPlacedVolume const *left  = shape->GetLeft();
      VPlacedVolume const *right = shape->GetRight();

      // CHECK IF THIS BOOLEAN VOLUME DEPENDS ON OTHER BOOLEAN VOLUMES NOT YET DUMPED
      // THIS SOLUTION IS POTENTIALLY SLOW; MIGHT CONSIDER DIFFERENT TYPE OF CONTAINER
      if (!ContainerContains(fListofTreatedLogicalVolumes, left->GetLogicalVolume()) ||
          !ContainerContains(fListofTreatedLogicalVolumes, right->GetLogicalVolume())) {
        // we need to defer the treatment of this logical volume
        fListofDeferredLogicalVolumes.push_back(l);
        continue;
      }
      line << " new UnplacedBooleanVolume( ";
      line << " kSubtraction ";
      line << " , ";
      // placed versions of left and right volume
      line << fLVolumeToStringMap[left->GetLogicalVolume()] << "->Place( "
           << fTrafoToStringMap[left->GetTransformation()] << " )";
      line << " , ";
      line << fLVolumeToStringMap[right->GetLogicalVolume()] << "->Place( "
           << fTrafoToStringMap[right->GetTransformation()] << " )";
      line << " )";
      fNeededHeaderFiles.insert("volumes/UnplacedBooleanVolume.h");
    }

    else if (dynamic_cast<UnplacedTrd const *>(l->GetUnplacedVolume())) {
      UnplacedTrd const *shape = dynamic_cast<UnplacedTrd const *>(l->GetUnplacedVolume());

      line << " new UnplacedTrd( ";
      line << shape->dx1() << " , ";
      line << shape->dx2() << " , ";
      line << shape->dy1() << " , ";
      line << shape->dy2() << " , ";
      line << shape->dz();
      line << " )";

      fNeededHeaderFiles.insert("volumes/UnplacedTrd.h");
    } else {
      line << " = new UNSUPPORTEDSHAPE()";
      line << l->GetLabel() << "\n";
    }

    line << " );\n";

    lvoldefinitions << "LogicalVolume *" << fLVolumeToStringMap[l] << "= nullptr;\n";
    externdeclarations << "extern LogicalVolume *" << fLVolumeToStringMap[l] << ";\n";

    // if we came here, we dumped this logical volume; so register it as beeing treated
    fListofTreatedLogicalVolumes.push_back(l);
  } // end loop over logical volumes
}

// now recreate geometry hierarchy
// the mappings fLogicalVolToStringMap and fTrafoToStringMap need to be initialized
void GeomCppExporter::DumpGeomHierarchy(std::vector<std::string> &dumps, std::list<LogicalVolume const *> const &lvlist)
{
  static unsigned int group = -1;
  group++;
  unsigned int groupcounter = 0;
  std::stringstream output;
  output << " void GeneratePlacedVolumes_part" << group << "(){\n";

  for (auto l : lvlist) {
    // map daughters for logical volume l
    std::string thisvolumevariable = fLVolumeToStringMap[l];

    for (size_t d = 0; d < l->GetDaughters().size(); ++d) {
      VPlacedVolume const *daughter = l->GetDaughters()[d];

      // get transformation and logical volume for this daughter
      Transformation3D const *t       = daughter->GetTransformation();
      LogicalVolume const *daughterlv = daughter->GetLogicalVolume();

      std::string tvariable = fTrafoToStringMap[t];
      std::string lvariable = fLVolumeToStringMap[daughterlv];

      // only allow 5000 lines per function to speed up compilation
      if (groupcounter++ > 5000) {
        output << "}";
        // new output
        PushAndReset(output, dumps);
        group++;
        output << " void GeneratePlacedVolumes_part" << group << "(){\n";
        groupcounter = 0;
      }

      // build the C++ code
      std::stringstream line;
      line << thisvolumevariable << "->PlaceDaughter( ";
      line << lvariable << " , ";
      line << tvariable << " );\n";

      output << line.str();
    }
  }
  // close the last output
  output << "}\n";
  // no need to reset here.
  dumps.push_back(output.str());
}

void GeomCppExporter::DumpHeader(std::ostream &dumps)
{
  // put some disclaimer ( to be extended )
  dumps << "// THIS IS AN AUTOMATICALLY GENERATED FILE -- DO NOT MODIFY\n";
  dumps << "// FILE SHOULD BE COMPILED INTO A SHARED LIBRARY FOR REUSE\n";
  // put standard headers
  dumps << "#include \"base/Global.h\"\n";
  dumps << "#include \"volumes/PlacedVolume.h\"\n";
  dumps << "#include \"volumes/LogicalVolume.h\"\n";
  dumps << "#include \"base/Transformation3D.h\"\n";
  dumps << "#include <vector>\n";

  // put shape specific headers
  for (auto headerfile : fNeededHeaderFiles) {
    dumps << "#include \"" << headerfile << "\"\n";
  }
}

void GeomCppExporter::DumpGeometry(std::ostream &s)
{
  // stringstreams to assemble code in parts
  std::vector<std::string> transformations;
  std::stringstream transexterndecl;
  std::vector<std::string> transdecl;

  std::stringstream logicalvolumes;
  std::stringstream lvoldefinitions;
  std::stringstream lvoldeclarations;

  std::stringstream header;
  std::vector<std::string> geomhierarchy;

  // create list of transformations, simple logical volumes and boolean logical volumes
  std::list<Transformation3D const *> tlist;
  std::list<LogicalVolume const *> lvlist;
  std::list<LogicalVolume const *> boollvlist;

  ScanGeometry(GeoManager::Instance().GetWorld(), lvlist, boollvlist, tlist);

  // generate code that instantiates the transformations
  DumpTransformations(transformations, transexterndecl, transdecl, tlist);
  // generate code that instantiates ordinary logical volumes
  DumpLogicalVolumes(logicalvolumes, lvoldeclarations, lvoldefinitions, lvlist);

  // generate code that instantiates complex logical volumes ( for the moment only booleans )
  // do a first pass
  DumpLogicalVolumes(logicalvolumes, lvoldeclarations, lvoldefinitions, boollvlist);
  int counter = 0;
  // do more passes to resolve dependencies between logical volumes
  // doing max 10 passes to protect against infinite loop ( which should never occur )
  while (fListofDeferredLogicalVolumes.size() > 0 && counter < 10) {
    std::list<LogicalVolume const *> remainingvolumes = fListofDeferredLogicalVolumes;
    fListofDeferredLogicalVolumes.clear();
    DumpLogicalVolumes(logicalvolumes, lvoldeclarations, lvoldefinitions, remainingvolumes);
    counter++;
  }

  // generate more header; this has to be done here since
  // headers are determined from the logical volumes used !!
  DumpHeader(header);

  // generate code that reproduces the geometry hierarchy
  DumpGeomHierarchy(geomhierarchy, lvlist);
  // dito for the booleans
  DumpGeomHierarchy(geomhierarchy, boollvlist);

  s << header.str();
  s << "using namespace vecgeom;\n";
  s << "\n";

  // write translation units for transformations
  for (unsigned int i = 0; i < transdecl.size(); ++i) {
    std::ofstream outfile;
    std::stringstream name;
    name << "geomconstr_trans_part" << i << ".cpp";
    outfile.open(name.str());
    outfile << "#include \"base/Transformation3D.h\"\n";
    outfile << "using namespace vecgeom;\n";
    outfile << transdecl[i];
    outfile << transformations[i];
    outfile.close();
  }

  // write translation unit for logical volumes
  {
    std::ofstream outfile;
    std::stringstream name;
    name << "geomconstr_lvol_part" << 0 << ".cpp";
    outfile.open(name.str());
    outfile << header.str();
    outfile << "using namespace vecgeom;\n";

    // we need external declarations for transformations
    outfile << transexterndecl.str();
    outfile << lvoldefinitions.str();
    outfile << "void CreateLogicalVolumes(){\n";
    outfile << logicalvolumes.str();
    outfile << "}\n";
    outfile.close();
  }

  { // write translation units for placed volumes
    for (unsigned int i = 0; i < geomhierarchy.size(); ++i) {
      std::ofstream outfile;
      std::stringstream name;
      name << "geomconstr_placedvol_part" << i << ".cpp";
      outfile.open(name.str());
      outfile << header.str();
      outfile << "using namespace vecgeom;\n";
      outfile << transexterndecl.str();
      outfile << lvoldeclarations.str();
      outfile << geomhierarchy[i];
      outfile.close();
    }
  }

  // create file that connects everything up
  {
    std::ofstream outfile;
    std::stringstream name;
    name << "geomconstr_createdetector.cpp";
    outfile.open(name.str());
    outfile << "#include \"base/Global.h\"\n";
    outfile << "#include \"volumes/PlacedVolume.h\"\n";
    outfile << "#include \"volumes/LogicalVolume.h\"\n";
    outfile << "#include \"base/Transformation3D.h\"\n";
    outfile << "#include \"management/GeoManager.h\"\n";
    outfile << "#include \"base/Stopwatch.h\"\n";
    outfile << "#include <iostream>\n";
    outfile << "using namespace vecgeom;";
    VPlacedVolume const *world   = GeoManager::Instance().GetWorld();
    LogicalVolume const *worldlv = world->GetLogicalVolume();
    // extern declarations
    for (unsigned int i = 0; i < transdecl.size(); ++i) {
      outfile << "extern void GenerateTransformations_part" << i << "();\n";
    }
    outfile << "extern void CreateLogicalVolumes();\n";
    for (unsigned int i = 0; i < geomhierarchy.size(); ++i) {
      outfile << "extern void GeneratePlacedVolumes_part" << i << "();\n";
    }
    outfile << "extern LogicalVolume * " << fLVolumeToStringMap[worldlv] << ";\n";
    outfile << "extern Transformation3D * " << fTrafoToStringMap[world->GetTransformation()] << ";\n";

    outfile << "VPlacedVolume const * generateDetector() {\n";
    // call all the functions from other translation units
    // ...  start with the transformations
    for (unsigned int i = 0; i < transdecl.size(); ++i) {
      outfile << "  GenerateTransformations_part" << i << "();\n";
    }
    outfile << "CreateLogicalVolumes();\n";
    for (unsigned int i = 0; i < geomhierarchy.size(); ++i) {
      outfile << "  GeneratePlacedVolumes_part" << i << "();\n";
    }
    outfile << "VPlacedVolume const * world = " << fLVolumeToStringMap[worldlv] << "->Place( "
            << fTrafoToStringMap[world->GetTransformation()] << " ); \n";
    outfile << "return world;\n}\n";
    outfile << "int main(){\n";
    outfile << "// function could be used like this \n";
    outfile << " GeoManager & geom = GeoManager::Instance();\n";
    outfile << " Stopwatch timer;\n";
    outfile << " timer.Start();\n";
    outfile << " geom.SetWorld( generateDetector() );\n";
    outfile << " geom.CloseGeometry();\n";
    outfile << " timer.Stop();\n";
    outfile << " std::cerr << \"loading took  \" << timer.Elapsed() << \" s \" << std::endl;\n";
    outfile << " std::cerr << \"loaded geometry has \" << geom.getMaxDepth() << \" levels \" << std::endl;\n";
    outfile << " return 0;}\n";

    outfile.close();
  }

  // create hint on how to use the generated function
  s << "// function could be used like this \n";
  s << "// int main(){\n";
  s << "// GeoManager & geom = GeoManager::Instance();\n";
  s << "// Stopwatch timer;\n";
  s << "// timer.Start();\n";
  s << "//geom.SetWorld( generateDetector() );\n";
  s << "//geom.CloseGeometry();\n";
  s << "//timer.Stop();\n";
  s << "//std::cerr << \"loading took  \" << timer.Elapsed() << \" s \" << std::endl;\n";
  s << "//std::cerr << \"loaded geometry has \" << geom.getMaxDepth() << \" levels \" << std::endl;\n";
  s << "// return 0;}\n";
}
} // namespace cxx
} // namespace vecgeom
