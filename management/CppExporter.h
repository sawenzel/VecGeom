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
#include <list>

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

// TODO:
// a) use more efficient containers for this treatment -- revise export algorithms; improve structure of code
// b) look for possibilities to reduce memory ( if we see that a transformation will be instantiated twice )
// c) optimized memory allocation in the construction of the detector; we could allocate only once and
//      then use placement new to place the transformations, logical volumes etc.
// d) ultimately we could try to bring together things in memory which are used together:
//       certain transformation - volume combinations
// e) split the output into various smaller functions which could be compiled in parallel
//    ( currently compiling the cms detector from the produced source takes quite a while ... )
// f) dump numeric values according to their precision needed ( the default precision will likely lead to errors )
// first version: Sandro Wenzel 26.3.2015

class GeomCppExporter {
    // declare the friend classes
    friend Transformation3D;


private:
    // mapping pointer to variable names for Transformations
    std::map< Transformation3D const *, std::string > fTrafoToStringMap;
    // mapping pointer to variable names for logical volumes
    std::map< LogicalVolume const *, std::string > fLVolumeToStringMap;
    std::set< std::string > fNeededHeaderFiles;

    std::list< LogicalVolume const * > fListofTreatedLogicalVolumes;
    // container to keep track of logical volumes which need to be coded in C++
    // at a later stage because a dependency is not satisfied
    std::list< LogicalVolume const * > fListofDeferredLogicalVolumes;


    void DumpTransformations( std::ostream &, std::list< Transformation3D const * > const & );
    void DumpLogicalVolumes( std::ostream &, std::list< LogicalVolume const * >  const & );
    void DumpGeomHierarchy( std::ostream &, std::list< LogicalVolume const * > const & );
    void DumpHeader( std::ostream & );

    void DumpEntryFunction();

    // private Constructor
    GeomCppExporter() : fTrafoToStringMap(), fLVolumeToStringMap(), fNeededHeaderFiles(),
                        fListofTreatedLogicalVolumes(), fListofDeferredLogicalVolumes() {}

public:
    static GeomCppExporter & Instance(){
        static GeomCppExporter instance;
        return instance;
    }


    void DumpGeometry( std::ostream & );
};

}} // end namespace


#endif /* CPPEXPORTER_H_ */
