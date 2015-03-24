/*
 * CppExporter.cpp
 *
 *  Created on: 23.03.2015
 *      Author: swenzel
 */

#include "management/CppExporter.h"
#include "management/GeoManager.h"
#include "base/Transformation3D.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedBox.h"
#include <sstream>
#include <ostream>
#include <vector>

namespace vecgeom {
inline namespace cxx {


    void GeomCppExporter::DumpTransformations( std::ostream & dumps ){

        std::vector<VPlacedVolume const *> allplacedvolumes;
        GeoManager::Instance().getAllPlacedVolumes( allplacedvolumes );

        // loop over all transformations
        unsigned int counter=0;
        for( auto p : allplacedvolumes ){
            Transformation3D const * t = p->transformation();
            // register transformation
            if( fTrafoToStringMap.find(t) == fTrafoToStringMap.cend() ){
                // create a variable name
                std::stringstream s;
                s << "transf" << counter;
                // do mapping from existing pointer value to variable name
                fTrafoToStringMap[ t ] = s.str();
                counter++;
            }
        }

        // generate code that instantiates transformations
        for ( auto t : fTrafoToStringMap ){
            Transformation3D const * tp = t.first;
            std::stringstream line;
            line << "Transformation3D * " << t.second << " = new Transformation3D(";
            line << tp->Translation(0) << " , ";
            line << tp->Translation(1) << " , ";
            line << tp->Translation(2) << " , ";
            for( auto i=0; i<8; ++i )
                line << tp->Rotation(i) << " , ";
            line << tp->Rotation(8);
            line << ");\n";
            dumps << line.str();
        }
    }

    // function which dumps the logical volumes
    void GeomCppExporter::DumpLogicalVolumes( std::ostream & dumps ) {
        std::vector<LogicalVolume const *> alllogicalvolumes;
        GeoManager::Instance().getAllLogicalVolumes( alllogicalvolumes );

        unsigned int counter=0;
        for( auto l : alllogicalvolumes ){
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
        for ( auto iter : fLVolumeToStringMap ){
            LogicalVolume const * l = iter.first;
            std::stringstream line;
            line << "LogicalVolume * " << iter.second;
            line << " = new LogicalVolume ( \"" << l->GetLabel() << "\" , ";

            // now we need to distinguish types
            // use here dynamic casting ( alternatives might exist )
            if( dynamic_cast<UnplacedBox const *>( l->unplaced_volume() ) ){
                UnplacedBox const * box = dynamic_cast<UnplacedBox const *>( l->unplaced_volume() );
                line << " new UnplacedBox( " ;
                   line << box->dimensions().x() << " , ";
                   line << box->dimensions().y() << " , ";
                   line << box->dimensions().z();
                   line << " )";

                fNeededHeaderFiles.insert("volumes/UnplacedBox.h");
            }
            else{
                line << " = new UNSUPPORTEDSHAPE()\n";
            }
            line << " );\n";
            dumps << line.str();
        }
    }


    // now recreate geometry hierarchy
    // the mappings fLogicalVolToStringMap and fTrafoToStringMap need to be initialized
    void GeomCppExporter::DumpGeomHierarchy( std::ostream & dumps ){
        // idea recreate hierarchy by adding daughters
        std::vector<LogicalVolume const *> alllogicalvolumes;
        GeoManager::Instance().getAllLogicalVolumes( alllogicalvolumes );

        for( auto l : alllogicalvolumes ){
            // map daughters for logical volume l

            std::string thisvolumevariable = fLVolumeToStringMap[l];

            for( auto d = 0; d < l->daughters().size(); ++d ){
                VPlacedVolume const * daughter = l->daughters()[d];

                // get transformation and logical volume for this daughter
                Transformation3D const * t = daughter->transformation();
                LogicalVolume const * daughterlv = daughter->logical_volume();

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
        // now define world
        VPlacedVolume const * world = GeoManager::Instance().GetWorld();
        Transformation3D const * t = world->transformation();
        LogicalVolume const * worldlv = world->logical_volume();
        dumps << "VPlacedVolume const * world = " << fLVolumeToStringMap[worldlv] << "->Place( "
                                          << fTrafoToStringMap[t] <<  " ); \n";
    }

    void GeomCppExporter::DumpHeader( std::ostream & dumps ){
        // put standard headers
        dumps << "#include \"base/Global.h\"\n";
        dumps << "#include \"volumes/PlacedVolume.h\"\n";
        dumps << "#include \"volumes/LogicalVolume.h\"\n";
        dumps << "#include \"base/Transformation3D.h\"\n";

        // put shape specific headers
        for( auto headerfile : fNeededHeaderFiles ){
            dumps << "#include \"" << headerfile << "\"\n";
        }
    }


//    void GeomCppExporter::DumpLogicalV( stream & dumps,  LogicalVolume const & lvol ){
//        // common part create variable name
//        std::stringstream s;
//        s << lvol.GetLabel();
//
//        // if GetLabelIsEmpty; need to create a variable name
//
//        // register pointer to variablename
//        fLVolumeToNameMap.add( &lvol, s.str() );
//
//        DumpSpecific()
//    }
//    void DumpSpecific( UnplacedBox const & ){
//    //
//
//    }

void GeomCppExporter::DumpGeometry( std::ostream & s ) {
    std::stringstream transformations;
    std::stringstream logicalvolumes;
    std::stringstream header;
    std::stringstream geomhierarchy;

    DumpTransformations( transformations );
    DumpLogicalVolumes( logicalvolumes );
    DumpHeader( header );
    DumpGeomHierarchy( geomhierarchy );

    s << header.str();
    s << "using namespace vecgeom;\n";
    s << "\n";

    // create function start; body and end
    s << "VPlacedVolume const * generateDetector() {\n";
    s << transformations.str();
    s << logicalvolumes.str();
    s << geomhierarchy.str();
    // return placed world Volume
    s << "return world;\n}\n";
}

}} // end namespace


