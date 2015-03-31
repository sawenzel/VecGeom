/*
 * CppExporter.cpp
 *
 *  Created on: 23.03.2015
 *      Author: swenzel
 */

#include "management/CppExporter.h"
#include "management/GeoManager.h"
#include "base/Transformation3D.h"
#include "base/Array.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/PlacedBooleanVolume.h"
#include "volumes/UnplacedBox.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedTrapezoid.h"
#include "volumes/UnplacedTorus.h"
#include "volumes/UnplacedPolycone.h"
#include "volumes/UnplacedPolyhedron.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/UnplacedBooleanVolume.h"
#include <sstream>
#include <ostream>
#include <algorithm>
#include <list>
#include <iomanip>

namespace vecgeom {
inline namespace cxx {

template<typename IterableContainer, typename ElementType>
bool ContainerContains( IterableContainer const & c, ElementType const & e ){
    return std::find( c.cbegin(), c.cend(), e ) != c.cend();
}


// this function should live in GeoManager and to be used by various exporters
// function returns
// a sorted list of logical volumes -- if some logical volume A depends on another volume B
// (as for boolean volumes); then B
// should come in that list before A
// a list of transformations
void ScanGeometry( VPlacedVolume const *const volume,
                   std::list<LogicalVolume const *> & lvlist,
                   std::list<LogicalVolume const *> & boollvlist,
                   std::list<Transformation3D const *> & tlist ) {
    // if not yet treated
    if( std::find( lvlist.cbegin(), lvlist.cend(), volume->GetLogicalVolume() ) == lvlist.cend()
      && std::find( boollvlist.cbegin(), boollvlist.cend(), volume->GetLogicalVolume() ) == boollvlist.cend() ) {

      if( dynamic_cast<PlacedBooleanVolume const*>(volume) ){
          boollvlist.push_front( volume->GetLogicalVolume() );
          PlacedBooleanVolume const* v =  dynamic_cast<PlacedBooleanVolume const*>(volume);
          ScanGeometry(v->GetUnplacedVolume()->fLeftVolume, lvlist, boollvlist, tlist );
          ScanGeometry(v->GetUnplacedVolume()->fRightVolume, lvlist, boollvlist, tlist );
      }
      else{
          // ordinary logical volume
          lvlist.push_back( volume->GetLogicalVolume() );
      }

      for( auto d = 0; d < volume->daughters().size(); ++d )
        ScanGeometry(volume->daughters()[d], lvlist, boollvlist, tlist);
  }

  if ( std::find( tlist.cbegin(),tlist.cend(), volume->GetTransformation()) == tlist.cend() ){
      tlist.push_back(volume->GetTransformation());
  }
}



    void GeomCppExporter::DumpTransformations( std::ostream & dumps, std::list<Transformation3D const *> const & tvlist ){

        // loop over all transformations
        unsigned int counter=0;
        for( auto t : tvlist ){
            // register transformation
            if( fTrafoToStringMap.find(t) == fTrafoToStringMap.cend() ){

                // many transformation are identity: we can filter them out and allocate only one
                // identity
                // TODO: such reduction can be applied for other transformations
                if( t->IsIdentity() ){
                    fTrafoToStringMap[ t ] = "idtrans";
                }
                else {
                    // create a variable name
                    std::stringstream s;
                    s << "transf" << counter;
                    // do mapping from existing pointer value to variable name
                    fTrafoToStringMap[ t ] = s.str();
                    counter++;
                }

            }
        }

        // generate code that instantiates transformations
        bool iddone = false;
        for ( auto t : fTrafoToStringMap ){
            Transformation3D const * tp = t.first;
            if ( tp->IsIdentity() && iddone ) continue;
            if ( tp->IsIdentity() ) iddone = true;
            std::stringstream line;
            line << std::setprecision(15);
            line << "Transformation3D * " << t.second << " = new Transformation3D(";
            line << tp->Translation(0) << " , ";
            line << tp->Translation(1) << " , ";
            line << tp->Translation(2);
            if( tp->HasRotation() ){
                line << " , ";
                for( auto i=0; i<8; ++i )
                    line << tp->Rotation(i) << " , ";
                line << tp->Rotation(8);
            }
            line << ");\n";
            dumps << line.str();
        }
    }

    template<typename VectorContainer>
    void DumpVector( VectorContainer const & v, std::ostream & dumps ){
        dumps << "&std::vector<double>{";
        for(auto j=0;j<v.size()-1;++j)
           dumps << v[j] << " , ";
        dumps << v[v.size()-1] << "}[0]";
    }

// function which dumps the logical volumes
void GeomCppExporter::DumpLogicalVolumes( std::ostream & dumps,
            std::list<LogicalVolume const *> const & lvlist ) {

    static unsigned int counter=0;
    for( auto l : lvlist ){
        // register logical volume
        if( fLVolumeToStringMap.find(l) == fLVolumeToStringMap.cend() ){
             // create a variable name
             std::stringstream s;
             s << "lvol" << counter;
             // do mapping from existing pointer value to variable name
             fLVolumeToStringMap[ l ] = s.str();
             counter++;
        }
    }

    // generate code that instantiates LogicalVolumes
    for ( auto l : lvlist ){
        std::stringstream line;
        line << std::setprecision(15);
        line << "LogicalVolume * " << fLVolumeToStringMap[l];
        line << " = new LogicalVolume ( \"" << l->GetLabel() << "\" , ";

        // now we need to distinguish types
        // use here dynamic casting ( alternatives might exist )
        // ******* TREAT THE BOX *********
        if( dynamic_cast<UnplacedBox const *>( l->unplaced_volume() ) ){
            UnplacedBox const * box = dynamic_cast<UnplacedBox const *>( l->unplaced_volume() );

            line << " new UnplacedBox( " ;
            line << box->dimensions().x() << " , ";
            line << box->dimensions().y() << " , ";
            line << box->dimensions().z();
            line << " )";

            fNeededHeaderFiles.insert("volumes/UnplacedBox.h");
        }

        // ******* TREAT THE TUBE *********
        else if( dynamic_cast<UnplacedTube const *>( l->unplaced_volume() ) ){
            UnplacedTube const * shape
                = dynamic_cast<UnplacedTube const *>( l->unplaced_volume() );

            line << " new UnplacedTube( " ;
            line << shape->rmin() << " , ";
            line << shape->rmax() << " , ";
            line << shape->z() << " , ";
            line << shape->sphi() << " , ";
            line << shape->dphi();
            line << " )";

            fNeededHeaderFiles.insert("volumes/UnplacedTube.h");
        }

        // ******* TREAT THE CONE *********
        else if( dynamic_cast<UnplacedCone const *>( l->unplaced_volume() ) ){
             UnplacedCone const * shape
                 = dynamic_cast<UnplacedCone const *>( l->unplaced_volume() );

             line << " new UnplacedCone( " ;
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
        else if( dynamic_cast<UnplacedTrapezoid const *>( l->unplaced_volume() ) ){
             UnplacedTrapezoid const * shape
                 = dynamic_cast<UnplacedTrapezoid const *>( l->unplaced_volume() );
             line << " new UnplacedTrapezoid( " ;

             line << shape->GetDz() << " , ";
             line << shape->GetTheta() << " , ";
             line << shape->GetPhi() << " , ";
             line << shape->GetDy1() << " , ";
             line << shape->GetDx1() << " , ";
             line << shape->GetDx2() << " , ";
             line << shape->GetAlpha1() << " , ";
             line << shape->GetDy2() << " , ";
             line << shape->GetDx3() << " , ";
             line << shape->GetDx4() << " , ";
             line << shape->GetAlpha2();
             line << " )";

             fNeededHeaderFiles.insert("volumes/UnplacedTrapezoid.h");
        }

        // ******* TREAT THE TORUS **********
        else if( dynamic_cast<UnplacedTorus const *>( l->unplaced_volume() ) ){
             UnplacedTorus const * shape
                 = dynamic_cast<UnplacedTorus const *>( l->unplaced_volume() );

             line << " new UnplacedTorus( " ;
             line << shape->rmin() << " , ";
             line << shape->rmax() << " , ";
             line << shape->rtor() << " , ";
             line << shape->sphi() << " , ";
             line << shape->dphi();
             line << " )";

             fNeededHeaderFiles.insert("volumes/UnplacedTorus.h");
        }

        // ********* TREAT THE PCON **********
        else if( dynamic_cast<UnplacedPolycone const *>( l->unplaced_volume() ) ){
              UnplacedPolycone const * shape
                = dynamic_cast<UnplacedPolycone const *>( l->unplaced_volume() );

             line << " new UnplacedPolycone( " ;
             line << shape->GetStartPhi() << " , ";
             line << shape->GetDeltaPhi() << " , ";
             line << shape->GetNz() << " , ";

             std::vector<double> rmin, rmax, z;
             // serialize the arrays as tempary std::vector
             shape->ReconstructSectionArrays(z,rmin,rmax);

             // put rmin vector
             DumpVector( rmin, line );
             line << " , ";
             // put rmax vector
             DumpVector( rmax, line );
             line << " , ";

             // put z vector
             DumpVector( z, line );
             line << " )";

            fNeededHeaderFiles.insert("volumes/UnplacedPolycone.h");
       }

        // ********* TREAT THE PGON **********
        else if( dynamic_cast<UnplacedPolyhedron const *>( l->unplaced_volume() ) ){
              UnplacedPolyhedron const * shape
                = dynamic_cast<UnplacedPolyhedron const *>( l->unplaced_volume() );
              line << " new UnplacedPolyhedron( " ;
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
               auto z = shape->GetZPlanes();
               auto rmin = shape->GetRMin();
               auto rmax = shape->GetRMax();

                // put z vector
                DumpVector( z, line );
                line << " , ";

                // put rmin vector
                DumpVector( rmin, line );
                line << " , ";

                // put rmax vector
                DumpVector( rmax, line );
                line << " )";

                fNeededHeaderFiles.insert("volumes/UnplacedPolyhedron.h");
          }

       // *** BOOLEAN SOLIDS NEED A SPECIAL TREATMENT *** //
        // their constituents are  not already a part of the logical volume list
        else if( dynamic_cast<UnplacedBooleanVolume const *>( l->unplaced_volume() ) ){
            UnplacedBooleanVolume const * shape
               = dynamic_cast<UnplacedBooleanVolume const *>( l->unplaced_volume() );

            VPlacedVolume const * left = shape->fLeftVolume;
            VPlacedVolume const * right = shape->fRightVolume;

            // CHECK IF THIS BOOLEAN VOLUME DEPENDS ON OTHER BOOLEAN VOLUMES NOT YET DUMPED
            // THIS SOLUTION IS POTENTIALLY SLOW; MIGHT CONSIDER DIFFERENT TYPE OF CONTAINER
            if( ! ContainerContains(fListofTreatedLogicalVolumes, left->GetLogicalVolume())
                    || ! ContainerContains(fListofTreatedLogicalVolumes, right->GetLogicalVolume()) ) {
                    // we need to defer the treatment of this logical volume
                    fListofDeferredLogicalVolumes.push_back( l );
                    continue;
            }

            line << " new UnplacedBooleanVolume( " ;
            if( shape->GetOp() == kUnion ){
                line << " kUnion ";
            }
            if( shape->GetOp() == kSubtraction ){
                line << " kSubtraction ";
            }
            if( shape->GetOp() == kIntersection ){
                line << " kIntersection ";
            }
            line << " , ";
            // placed versions of left and right volume
            line << fLVolumeToStringMap[ left->GetLogicalVolume() ]
                 << "->Place( "
                 << fTrafoToStringMap[ left->GetTransformation() ]
                 << " )";
            line << " , ";
            line << fLVolumeToStringMap[ right->GetLogicalVolume() ]
                 << "->Place( "
                 << fTrafoToStringMap[ right->GetTransformation() ]
                 <<  " )";
            line << " )";

            fNeededHeaderFiles.insert("volumes/UnplacedBooleanVolume.h");
        }

        else if( dynamic_cast<UnplacedTrd const *>( l->unplaced_volume() ) ){
            UnplacedTrd const * shape
                 = dynamic_cast<UnplacedTrd const *>( l->unplaced_volume() );

            line << " new UnplacedTrd( " ;
            line << shape->dx1() << " , ";
            line << shape->dx2() << " , ";
            line << shape->dy1() << " , ";
            line << shape->dy2() << " , ";
            line << shape->dz();
            line << " )";

            fNeededHeaderFiles.insert("volumes/UnplacedTrd.h");
        }
        else{
                line << " = new UNSUPPORTEDSHAPE()";
                line << l->GetLabel() << "\n";
        }

    line << " );\n";
    dumps << line.str();
    // if we came here, we dumped this logical volume; so register it as beeing treated
    fListofTreatedLogicalVolumes.push_back( l );
   } // end loop over logical volumes
}


// now recreate geometry hierarchy
// the mappings fLogicalVolToStringMap and fTrafoToStringMap need to be initialized
void GeomCppExporter::DumpGeomHierarchy( std::ostream & dumps, std::list<LogicalVolume const *> const & lvlist ){

    for( auto l : lvlist ){
        // map daughters for logical volume l
        std::string thisvolumevariable = fLVolumeToStringMap[l];

        for( auto d = 0; d < l->daughters().size(); ++d ){
            VPlacedVolume const * daughter = l->daughters()[d];

            // get transformation and logical volume for this daughter
            Transformation3D const * t = daughter->GetTransformation();
            LogicalVolume const * daughterlv = daughter->GetLogicalVolume();

            std::string tvariable = fTrafoToStringMap[t];
            std::string lvariable = fLVolumeToStringMap[daughterlv];
//                // build the C++ code
            std::stringstream line;
            line << thisvolumevariable << "->PlaceDaughter( ";
            line << lvariable << " , ";
            line << tvariable << " );\n";

            dumps << line.str();
       }
    }
}

void GeomCppExporter::DumpHeader( std::ostream & dumps ){
    // put some disclaimer ( to be extended )
    dumps << "// THIS IS AN AUTOMATICALLY GENERATED FILE -- DO NOT MODIFY\n";
    dumps << "// FILE SHOULD BE COMPILED INTO A SHARED LIBRARY FOR REUSE\n";
    // put standard headers
    dumps << "#include \"base/Global.h\"\n";
    dumps << "#include \"volumes/PlacedVolume.h\"\n";
    dumps << "#include \"volumes/LogicalVolume.h\"\n";
    dumps << "#include \"base/Transformation3D.h\"\n";
    dumps << "#include \"management/GeoManager.h\"\n";
    dumps << "#include \"base/Stopwatch.h\"\n";
    dumps << "#include <vector>\n";

    // put shape specific headers
    for( auto headerfile : fNeededHeaderFiles ){
        dumps << "#include \"" << headerfile << "\"\n";
    }
}


void GeomCppExporter::DumpGeometry( std::ostream & s ) {
  // stringstreams to assemble code in parts
    std::stringstream transformations;
    std::stringstream logicalvolumes;
    std::stringstream header;
    std::stringstream geomhierarchy;

    // create list of transformations, simple logical volumes and boolean logical volumes
    std::list<Transformation3D const *> tlist;
    std::list<LogicalVolume const *> lvlist;
    std::list<LogicalVolume const *> boollvlist;
    ScanGeometry( GeoManager::Instance().GetWorld(), lvlist, boollvlist, tlist );

    // generate code that instantiates the transformations
    DumpTransformations( transformations, tlist );
    // generate code that instantiates ordinary logical volumes
    DumpLogicalVolumes( logicalvolumes, lvlist );

    // generate code that instantiates complex logical volumes ( for the moment only booleans )
    // do a first pass
    DumpLogicalVolumes( logicalvolumes, boollvlist );
    int counter=0;
    // do more passes to resolve dependencies between logical volumes
    // doing max 10 passes to protect against infinite loop ( which should never occur )
    while( fListofDeferredLogicalVolumes.size() > 0 && counter < 10){
        std::list< LogicalVolume const *> remainingvolumes = fListofDeferredLogicalVolumes;
        fListofDeferredLogicalVolumes.clear();
        DumpLogicalVolumes( logicalvolumes, remainingvolumes );
        counter++;
    }

    // generate more header; this has to be done here since
    // headers are determined from the logical volumes used !!
    DumpHeader( header );

    // generate code that reproduces the geometry hierarchy
    DumpGeomHierarchy( geomhierarchy, lvlist );
    // dito for the booleans
    DumpGeomHierarchy( geomhierarchy, boollvlist );

    s << header.str();
    s << "using namespace vecgeom;\n";
    s << "\n";

    // create function start; body and end
    s << "VPlacedVolume const * generateDetector() {\n";
    s << transformations.str();
    s << logicalvolumes.str();
    s << geomhierarchy.str();
    // return placed world Volume
    // now define world
    VPlacedVolume const * world = GeoManager::Instance().GetWorld();
    Transformation3D const * t = world->GetTransformation();
    LogicalVolume const * worldlv = world->GetLogicalVolume();
    s << "VPlacedVolume const * world = " << fLVolumeToStringMap[worldlv] << "->Place( "
                                             << fTrafoToStringMap[t] <<  " ); \n";
    s << "return world;\n}\n";

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

}} // end namespace


