/*
 * CppExporter.h
 *
 *  Created on: 23.03.2015
 *      Author: swenzel
 */

#ifndef VECGEOM_CPPEXPORTER_H_
#define VECGEOM_CPPEXPORTER_H_

#include "base/Global.h"
#include <map>
#include <ostream>
#include <set>

// Compile for vecgeom namespace to work as interface
namespace vecgeom {

#ifndef VECGEOM_NVCC
inline
#endif
namespace cxx {

class Transformation3D;
class LogicalVolume;

// a class to provide serialization functionality of an existing
// geometry hierarchy to C++ code; This code can then be compiled into a library
// from which the geometry can be reloaded quickly
// This export is useful in situations where other export/import functionality can be used or is not convenient

// NOTE: The class is implemented in isolation of any shape functionality (but as a friend of geometry classes)
// in order not to bloat the functionality and interfaces of geometry classes with serialization specific code
// the obvious drawback is that the Exporter has to be updated whenever new classes are added or changed

// idea: could also export in specialized mode

class GeomCppExporter {
    // declare the friend classes
    friend Transformation3D;


private:
    // mapping pointer to variable names for Transformations
    std::map< Transformation3D const *, std::string > fTrafoToStringMap;
    // mapping pointer to variable names for logical volumes
    std::map< LogicalVolume const *, std::string > fLVolumeToStringMap;
    std::set< std::string > fNeededHeaderFiles;

    void DumpTransformations( std::ostream & );
    void DumpLogicalVolumes( std::ostream & );
    void DumpGeomHierarchy( std::ostream & );
    void DumpHeader( std::ostream & );

    void DumpEntryFunction();

    // private Constructor
    GeomCppExporter() : fTrafoToStringMap(), fLVolumeToStringMap(), fNeededHeaderFiles() {}

public:
    static GeomCppExporter & Instance(){
        static GeomCppExporter instance;
        return instance;
    }


    void DumpGeometry( std::ostream & );
};

}} // end namespace


#endif /* CPPEXPORTER_H_ */
