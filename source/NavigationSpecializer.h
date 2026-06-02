/*
 * NavigationSpecializer.h
 *
 *  Created on: 11.09.2015
 *      Author: swenzel
 */

#ifndef VECGEOM_SERVICES_NAVIGATIONSPECIALIZER_H_
#define VECGEOM_SERVICES_NAVIGATIONSPECIALIZER_H_

#include "VecGeom/navigation/NavigationState.h"
#include <string>
#include <iosfwd>
#include <map>
#include <sstream>
#include <list>
#include <vector>
#include "VecGeom/base/Global.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
class LogicalVolume;

// A class providing convenient access to tabulated coordinate transformation data
// could make it a private subclass if we don't want to expose it
class TabulatedTransData {
public:
  TabulatedTransData(std::string name, bool soa = true)
      : fName(name), fTransCoefficients(3), fRotCoefficients(9), fSOA(soa), fTransVariableName(3), fRotVariableName(9),
        fVecTransVariableName(3), fVecRotVariableName(9)
  {
    for (size_t i = 0; i < 9; ++i)
      fRotCoefficients[i].resize(0);
    for (size_t i = 0; i < 3; ++i)
      fTransCoefficients[i].resize(0);
  }
  void Analyse();
  void SetRotCoef(size_t i, size_t index, double x) { fRotCoefficients[i][index] = x; }
  void SetTransCoef(size_t i, size_t index, double x) { fTransCoefficients[i][index] = x; }
  void ReserveRot(size_t index, size_t size)
  {
    fRotCoefficients[index].reserve(size);
    fRotCoefficients[index].resize(size, 0.);
  }
  void ReserveTrans(size_t index, size_t size)
  {
    fTransCoefficients[index].reserve(size);
    fTransCoefficients[index].resize(size, 0.);
  }
  void Print() const; // some debugging output

  void EmitTableDefinition(std::string /*classname*/, std::ostream &) const;
  void EmitTableDeclaration(std::ostream &);

  void EmitScalarGlobalTransformationCode(std::ostream &) const;
  void EmitScalarDeltaTransformationCode(std::ostream &) const;

  void EmitVectorGlobalTransformationCode(std::ostream &) const;

  //  void IsSOA() const {return fSOA;}

  void PrintStaticSOADefinition() const;
  void PrintStaticAOSDefinition(/*might need a name*/) const;

  bool RotIsZero(size_t index) const;     //
  bool RotIsOne(size_t index) const;      //
  bool RotIsMinusOne(size_t index) const; //
  bool TransIsZero(size_t index) const;   //
  bool RotIsConstant(size_t index) const;
  bool TransIsConstant(size_t index) const;

private:
  std::string fName; // a name addressing this transformation ( example: gGlobalTransf )
  std::vector<std::vector<double>> fTransCoefficients; // the raw numbers for transformations
  std::vector<std::vector<double>> fRotCoefficients;   // the raw number for rotations
  bool fSOA = false;                                   // emit SOA tables ( or AOS )

  // the following variables are initialized to true because its easier to convert them to false
  // during the analysis
  // they make only sense after a call to Analyse()
  bool fRotalwayszero[9]          = {true, true, true, true, true, true, true, true, true};
  bool fRotalwaysone[9]           = {true, true, true, true, true, true, true, true, true};
  bool fRotalwaysminusone[9]      = {true, true, true, true, true, true, true, true, true};
  bool fRotalwaysminusoneorone[9] = {true, true, true, true, true, true, true, true, true};
  bool fTransalwayszero[3]        = {true, true, true};
  bool fTransIsConstant[3]        = {true, true, true}; // indicates if this component is a constant for all entries
  bool fRotIsConstant[9] = {true, true, true, true, true,
                            true, true, true, true}; // indicates if this component is a constant for all entries
  std::vector<std::string> fTransVariableName;       // variable names which are set according to SOA/AOS choices etc
  std::vector<std::string> fRotVariableName;
  std::vector<std::string> fVecTransVariableName; // variable names which are set according to SOA/AOS choices etc
  std::vector<std::string> fVecRotVariableName;
};

/* A class which can produce (per logical volume) specialized C++ code
 * for navigation routines.
 *
 * A service which can transform generic navigation algorithms into specialized kernels
 * based on static analysis of the logical volume and the geometry hierarchy.
 * Example: It can convert a SimpleSafetyEstimator to FOOSafetyEstimator for a logical volume called
 *          FOO.
 * The output of the service are C++ files which can be (re- or just-in-time) compiled.
 *
 * Things which this service can (might) do are:
 *
 * a) avoid virtual functions in navigation methods
 * b) precompute all possible global transformations of a logical volume and cache them in the specialized kernels
 * c) provide a fast hard-coded lookup table to map NavigationStates objects to cached global transformations
 * d) vectorize loops over daughter volumes of the same type
 * e) loop splitting of daughter lists
 * f) realizing that daughtervolumes use the same rotations and do this transformation only once
 * g) ....
 *
 * The present class is an R&D prototype. Nothing is fix here!!
 *
 * 09/2015: idea and first implementation ( sandro.wenzel@cern.ch )
 */
class NavigationSpecializer {

public:
  // is this class a singleton ??
  NavigationSpecializer(std::string instatefile, std::string outstatefile)
      : fLogicalVolumeName(), fClassName(), fInStateFileName(instatefile), fOutStateFileName(outstatefile),
        fLogicalVolume(nullptr), fGeometryDepth(0), fIndexMap(), fStaticArraysInitStream(), fStaticArraysDefinitions(),
        fTransformationCode(), fVectorTransformVariables(),
        fVectorTransformationCode(), // to collect the many-path/SIMD transformation statements
        fTransformVariables(),       // stores the list of relevant transformation variables
        fUnrollLoops(false),         // whether to manually unroll all loops
        fUseBaseNavigator(false),    // whether to use the DaughterDetection from another navigator ( makes sense when
                                     // combined with voxel techniques )
        fBaseNavigator(), fGlobalTransData("globalTrans", true) // init 12 vectors : 3 for translation, 9 for rotation
  {};

  // produce a specialized SafetyEstimator class for a given logical volume
  // currently this is only done using the SimpleEstimator base algorithm
  // TODO: we could template here on some base algorithm in general and we could
  // specialize voxel algorithms and the like
  void ProduceSpecializedNavigator(LogicalVolume const *, std::ostream &);

private:
  typedef std::map<size_t, std::map<NavigationState::Value_t, size_t>> PathLevelIndexMap_t;

  // analysis functions
  void AnalyseLogicalVolume();
  void AnalysePaths(std::list<NavigationState *> const & /* inpaths */);
  void AnalyseTargetPaths(std::vector<NavigationState> const &, std::vector<NavigationState> const &);
  // void GeneratePathClassifierCode(std::list<std::pair<int, std::set<NavigationState::Value_t>>> const
  // &pathclassification,
  //                                 PathLevelIndexMap_t &map);

  void AddToIndexMap(size_t, NavigationState::Value_t);
  size_t PathToIndex(NavigationState const *);
  void AnalyseIndexCorrelations(std::list<NavigationState *> const &);

  // writer functions
  void DumpDisclaimer(std::ostream &);
  void DumpIncludeFiles(std::ostream &);
  void DumpNamespaceOpening(std::ostream &);
  void DumpNamespaceClosing(std::ostream &);
  void DumpStaticConstExprData(std::ostream &);
  void DumpStaticConstExprVariableDefinitions(std::ostream &);
  void DumpStaticInstanceFunction(std::ostream &);
  void DumpConstructor(std::ostream &) const;
  void DumpClassOpening(std::ostream &);
  void DumpClassDefinitions(std::ostream &);
  void DumpClassDeclarations(std::ostream &);
  void DumpClassClosing(std::ostream &);
  void DumpPathToIndexFunction(std::ostream &);
  void DumpVectorTransformationFunction(std::ostream &);
  void DumpLocalSafetyFunction(std::ostream &);
  void DumpPrivateClassDefinitions(std::ostream &);
  void DumpPublicClassDefinitions(std::ostream &);
  void DumpLocalSafetyFunctionDeclaration(std::ostream &);

  void DumpRelocateMethod(std::ostream &) const;

  void DumpSafetyFunctionDeclaration(std::ostream &);
  void DumpTransformationAsserts(std::ostream &);

  void DumpLocalHitDetectionFunction(std::ostream &) const;

  // they produce static component functions which are plugged into the generic implementation of VNavigatorHelper
  void DumpFoo(std::ostream &) const;
  void DumpStaticTreatGlobalToLocalTransformationFunction(std::ostream &) const;

  void DumpStaticTreatDistanceToMotherFunction(std::ostream &) const;
  void DumpStaticPrepareOutstateFunction(std::ostream &) const;

public:
  void EnableLoopUnrolling() { fUnrollLoops = true; }
  void DisableLoopUnrolling() { fUnrollLoops = false; }

  void SetBaseNavigator(std::string const &nav)
  {
    fUseBaseNavigator = true;
    fBaseNavigator    = nav;
  }

  typedef std::pair<int, std::string> FinalDepthShapeType_t;

private:
  // private state
  std::string fLogicalVolumeName;
  std::string fClassName;
  std::string fInStateFileName;
  std::string fOutStateFileName;
  LogicalVolume const *fLogicalVolume;
  unsigned int fGeometryDepth; // the depth of instances of fLogicalVolumes in the geometry hierarchy ( must be unique )
  unsigned int fNumberOfPossiblePaths;
  PathLevelIndexMap_t fIndexMap;                // in memory structure; used to map a NavigationState object to an index
  std::stringstream fStaticArraysInitStream;    // stream to collect code for the static arrays
  std::stringstream fStaticArraysDefinitions;   // stream to collect code for constexpr static array definitions
  std::stringstream fTransformationCode;        // to collect the specialized transformation statements for points
  std::stringstream fTransformationCodeDir;     // to collect the specialized transformation statements for dirs
  std::stringstream fVectorTransformVariables;  // to collect relevant vector variables for transformation
  std::stringstream fVectorTransformationCode;  // to collect the many-path/SIMD transformation statements
  std::vector<std::string> fTransformVariables; // stores the list of relevant transformation variables
  bool fUnrollLoops;                            // whether to manually unroll all loops
  bool fUseBaseNavigator; // whether to use the DaughterDetection from another navigator ( makes sense when combined
                          // with voxel techniques )
  std::string fBaseNavigator;

  std::stringstream fDeltaTransformationCode;
  std::vector<std::string>
      fTransitionTransformVariables; // stores the vector of relevant variables for the relocation transformations

  //
  std::vector<std::string> fTransitionStrings;
  std::vector<FinalDepthShapeType_t> fTransitionTargetTypes; // the vector of possible relocation target types
  std::vector<size_t> fTransitionOrder; // the vector keeping the order of indices of relocation transitions (
                                        // as stored in fTransitionTargetsTypes )
  std::vector<unsigned int>
      fTargetVolIds; // the ids of the target volumes ( to quickly fetch a representative volume pointer )

  std::vector<std::vector<int>>
      fPathxTargetToMatrixTable; // an in - memory table to fetch the correct transition matrix index
  std::stringstream fPathxTargetToMatrixTableStringStream; // string representation of the above

  // caching the transformation numbers --> to build a SOA/AOS form
  TabulatedTransData fGlobalTransData;

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_SERVICES_NAVIGATIONSPECIALIZER_H_ */
