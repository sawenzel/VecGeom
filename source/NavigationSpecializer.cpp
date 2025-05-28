/*
 * NavigationSpecializer.cpp
 *
 *  Created on: 11.09.2015
 *      Author: swenzel
 */

#include "NavigationSpecializer.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/NavStatePool.h"
#include <iostream>
#include <list>
#include <set>
#include <map>
#include <iomanip>
#include <iterator>
#include <tuple>

namespace vecgeom {

// returns number of different values of a list of NavigationStates at a particular  level
template <typename SetContainer>
int PathsAreDifferentAtGivenLevel(std::list<NavigationState *> const &paths, size_t level, SetContainer &values)
{
  auto cmpvalue = paths.front()->ValueAt(level);
  values.insert(cmpvalue);
  for (auto path : paths) {
    if (path->ValueAt(level) != cmpvalue) {
      values.insert(path->ValueAt(level));
    }
  }
  return values.size() >= 2 ? values.size() : 0;
}

// generic function to check if a collection contains just one element
template <typename Container, typename T>
bool CollectionIsConstant(Container const &c, T oneelementinc)
{
  for (auto element : c) {
    if (element != oneelementinc) return false;
  }
  return true;
}

// print all elements in vectors into a static array initializer list
auto printlambda = [](std::string name, std::vector<double> const &v, std::ostream &outstream) {
  outstream << "static constexpr double " << name << "[" << v.size() << "] = {";
  size_t numberofelements = v.size();
  size_t counter          = 0;
  for (auto e : v) {
    outstream << std::setprecision(35) << e;
    if (counter < numberofelements - 1) outstream << ", ";
    counter++;
  }
  outstream << "};\n";
};

// print one element static variables
auto printlambdasingle = [](std::string name, double e, std::ostream &outstream) {
  outstream << "static constexpr double " << name << "= ";
  outstream << std::setprecision(35) << e;
  outstream << ";\n";
};

void TabulatedTransData::Analyse()
{
  for (int i = 0; i < 9; ++i) {
    // rotation
    auto const &coeffs = fRotCoefficients[i];
    // if there is anything to analyse
    if (coeffs.size() > 0) {
      if (!CollectionIsConstant(coeffs, coeffs[0])) {
        fRotIsConstant[i] = false;
      }
      for (auto c : coeffs) {
        if (std::abs(c) > 1E-9) {
          fRotalwayszero[i] = false;
        }
        if (!(std::abs(c - 1.) < 1E-9)) {
          fRotalwaysone[i] = false;
        }
        if (!(std::abs(c + 1) < 1E-9)) {
          fRotalwaysminusone[i] = false;
        }
        if (!((std::abs(c + 1) < 1E-9) || (std::abs(c - 1) < 1E-9))) {
          fRotalwaysminusoneorone[i] = false;
        }
      }
    }
  }

  // translation
  for (int i = 0; i < 3; ++i) {
    auto const &coeffs = fTransCoefficients[i];
    if (coeffs.size() > 0) {
      if (!CollectionIsConstant(coeffs, coeffs[0])) {
        fTransIsConstant[i] = false;
      }
      for (auto c : coeffs) {
        if (std::abs(c) > 1E-9) {
          fTransalwayszero[i] = false;
        }
      }
    }
  } // end translation analysis
}

void TabulatedTransData::Print() const
{
  std::cerr << " --- output of analysis ---- \n";
  for (size_t i = 0; i < 9; ++i) {
    std::cerr << "rot[" << i << "] exists: " << (fRotCoefficients[i].size() > 0) << "\n";
    std::cerr << "rot[" << i << "] is constant " << fRotIsConstant[i] << "\n";
    std::cerr << "rot[" << i << "] is zero " << fRotalwayszero[i] << "\n";
    std::cerr << "rot[" << i << "] is one " << fRotalwaysone[i] << "\n";
    std::cerr << "rot[" << i << "] is minus one " << fRotalwaysminusone[i] << "\n";
    std::cerr << "rot[" << i << "] is either one or minus one " << fRotalwaysminusoneorone[i] << "\n";
  }
  for (size_t i = 0; i < 3; ++i) {
    std::cerr << "trans[" << i << "] exists: " << (fTransCoefficients[i].size() > 0) << "\n";
    std::cerr << "trans[" << i << "] is constant " << fTransIsConstant[i] << "\n";
    std::cerr << "trans[" << i << "] is zero " << fTransIsConstant[i] << "\n";
  }
}

void TabulatedTransData::EmitTableDeclaration(std::ostream &outstream)
{
  // emit in SOA form
  outstream << "// ------- generated tables ------\n";
  if (fSOA) {
    for (size_t i = 0; i < 3; ++i) {
      if (fTransCoefficients[i].size() > 0) {
        std::stringstream ss;
        ss << fName << "trans" << i;
        std::stringstream ss2;
        ss2 << ss.str();
        if (!fTransIsConstant[i]) {
          ss2 << "[index]";
          printlambda(ss.str(), fTransCoefficients[i], outstream);
        } else {
          printlambdasingle(ss.str(), fTransCoefficients[i][0], outstream);
        }
        fTransVariableName[i] = ss2.str();
        ss << "_v";
        fVecTransVariableName[i] = ss.str();
      }
    }
    for (size_t i = 0; i < 9; ++i) {
      if (fRotCoefficients[i].size() > 0) {
        std::stringstream ss;
        ss << fName << "rot" << i;
        std::stringstream ssv;
        ssv << ss.str();
        std::stringstream ss2;
        ss2 << ss.str();
        if (!fRotIsConstant[i]) {
          ss2 << "[index]";
          printlambda(ss.str(), fRotCoefficients[i], outstream);
        } else {
          printlambdasingle(ss.str(), fRotCoefficients[i][0], outstream);
        }
        fRotVariableName[i] = ss2.str();
        ss << "_v";
        fVecRotVariableName[i] = ss.str();
      }
    }
  } else { // emit in AOS form
    // convenience data view to emit AOS data
    std::vector<std::vector<double> const *> data;

    // we need to emit a struct for the non-const variabels
    outstream << "struct " << fName << "Struct{\n";
    for (size_t i = 0; i < 3; ++i) {
      if (fTransCoefficients[i].size() > 0 && !fTransIsConstant[i]) {
        outstream << "double trans" << i << ";\n";
        std::stringstream stringbuilder;
        std::stringstream vecstringbuilder;
        stringbuilder << fName << "[index]."
                      << "trans" << i;
        vecstringbuilder << "trans" << i << "_v";
        fTransVariableName[i] = stringbuilder.str();
        data.push_back(&fTransCoefficients[i]);
        fVecTransVariableName[i] = vecstringbuilder.str();
      }
    }
    for (size_t i = 0; i < 9; ++i) {
      if (fRotCoefficients[i].size() > 0 && !fRotIsConstant[i]) {
        outstream << "double rot" << i << ";\n";
        std::stringstream stringbuilder;
        std::stringstream vecstringbuilder;
        stringbuilder << fName << "[index]."
                      << "rot" << i;
        vecstringbuilder << "rot" << i << "_v";
        fRotVariableName[i] = stringbuilder.str();
        data.push_back(&fRotCoefficients[i]);
        fVecRotVariableName[i] = vecstringbuilder.str();
      }
    }
    outstream << "};\n";

    // still need to do the const variables externally
    for (size_t i = 0; i < 3; ++i) {
      if (fTransCoefficients[i].size() > 0 && fTransIsConstant[i]) {
        std::stringstream ss;
        ss << fName << "trans" << i;
        fTransVariableName[i] = ss.str();
        printlambdasingle(ss.str(), fTransCoefficients[i][0], outstream);
      }
    }
    for (size_t i = 0; i < 9; ++i) {
      if (fRotCoefficients[i].size() > 0 && fRotIsConstant[i]) {
        std::stringstream ss;
        ss << fName << "rot" << i;
        fRotVariableName[i] = ss.str();
        printlambdasingle(ss.str(), fRotCoefficients[i][0], outstream);
      }
    }

    // now the actual data
    outstream << "static constexpr " << fName << "Struct " << fName << "[] = {\n";
    for (size_t i = 0; i < data[0]->size(); ++i) {
      outstream << "{";
      for (size_t j = 0; j < data.size(); ++j) {
        outstream << (*data[j])[i];
        if (j < data.size() - 1) outstream << ",";
      }
      outstream << "}";
      if (i < data[0]->size() - 1) outstream << ",";
    }
    outstream << "};\n";
  }
}

void TabulatedTransData::EmitTableDefinition(std::string classname, std::ostream &outstream) const
{
  if (!fSOA) {
    outstream << "constexpr " << classname << "::" << fName << "Struct " << classname << "::" << fName << "[];\n";
  } else {
    for (size_t i = 0; i < 3; ++i) // translations
    {
      if (fTransCoefficients[i].size() > 0) {
        std::stringstream ss;
        ss << fName << "trans" << i;
        if (!fTransIsConstant[i]) {
          ss << "[]";
        }
        outstream << "constexpr double " << classname << "::" << ss.str() << ";\n";
      }
    }
    for (size_t i = 0; i < 9; ++i) {
      if (fRotCoefficients[i].size() > 0) {
        std::stringstream ss;
        ss << fName << "rot" << i;
        if (!fRotIsConstant[i]) {
          ss << "[]";
        }
        outstream << "constexpr double " << classname << "::" << ss.str() << ";\n";
      }
    }
  }
}

void TabulatedTransData::EmitScalarGlobalTransformationCode(std::ostream &outstream) const
{
  std::stringstream pointtransf;
  std::stringstream dirtrans;
  pointtransf << "Vector3D<Precision> tmp( globalpoint[0]";
  if (!fTransalwayszero[0]) {
    pointtransf << "- " << fTransVariableName[0] << "\n";
  }
  pointtransf << ", globalpoint[1]";
  if (!fTransalwayszero[1]) {
    pointtransf << "- " << fTransVariableName[1] << "\n";
  }
  pointtransf << ", globalpoint[2]";
  if (!fTransalwayszero[2]) {
    pointtransf << "- " << fTransVariableName[2] << "\n";
  }
  pointtransf << ");\n";

  int rotindex      = 0;
  bool indexseen[3] = {false, false, false};
  // tmp loop
  for (int tmpindex = 0; tmpindex < 3; ++tmpindex) {
    // local loop
    for (int localindex = 0; localindex < 3; ++localindex) {
      std::string op = indexseen[localindex] ? "+=" : "=";
      if (!fRotalwayszero[rotindex]) {
        indexseen[localindex] = true;
        // new line
        pointtransf << "localpoint[" << localindex << "]" << op;
        dirtrans << "localdir[" << localindex << "]" << op;
        //}
        if (fRotalwaysone[rotindex]) {
          pointtransf << "tmp[" << tmpindex << "];\n";
          dirtrans << "globaldir[" << tmpindex << "];\n";
        } else if (fRotalwaysminusone[rotindex]) {
          pointtransf << "-tmp[" << tmpindex << "];\n";
          dirtrans << "-globaldir[" << tmpindex << "];\n";
          //   else if check for plusorminus one --> could just copy sign instead of doing a multiplication

        } else { // generic version
                 // pointtransf << "tmp[" << tmpindex << "] * gRot" << rotindex << "[index];\n";
                 // dirtrans << "globaldir[" << tmpindex << "] * gRot" << rotindex << "[index];\n";
          pointtransf << "tmp[" << tmpindex << "] * " << fRotVariableName[rotindex] << ";\n";
          dirtrans << "globaldir[" << tmpindex << "] * " << fRotVariableName[rotindex] << ";\n";
        }
      }
      rotindex++;
    }
  }
  outstream << pointtransf.str();
  outstream << dirtrans.str();
}

void TabulatedTransData::EmitVectorGlobalTransformationCode(std::ostream &outstream) const
{
  std::stringstream pointtransf;
  std::stringstream dirtrans;

  // declare local vector variable names
  for (size_t i = 0; i < 3; ++i) {
    if (fTransCoefficients[i].size() > 0) {
      outstream << "T " << fVecTransVariableName[i] << ";\n";
    }
  }
  for (size_t i = 0; i < 9; ++i) {
    if (fRotCoefficients[i].size() > 0) {
      outstream << "T " << fVecRotVariableName[i] << ";\n";
    }
  }
  // fill them vector
  outstream << "// filling the vectors from the tabulated data \n";
  outstream << "// TODO: index independent data should come first (outside the loop)\n";
  outstream << "for(size_t i=0;i<ChunkSize;++i){\n";
  outstream << "auto trackindex = from_index + i;\n";
  outstream << "auto index = PathToIndex( in_states[trackindex] );\n";
  outstream << "// caching this index in internal navigationstate for later reuse\n";
  outstream << "// we know that is safe to do this because of static analysis (never do this in user code)\n";
  outstream << "internal[trackindex]->SetCacheValue(index);\n";
  for (size_t i = 0; i < 3; ++i) {
    if (fTransCoefficients[i].size() > 0 && !(fTransIsConstant[i])) {
      outstream << fVecTransVariableName[i] << "[i] = " << fTransVariableName[i] << ";\n";
    }
  }
  for (size_t i = 0; i < 9; ++i) {
    if (fRotCoefficients[i].size() > 0 && !(fRotIsConstant[i])) {
      outstream << fVecRotVariableName[i] << "[i] = " << fRotVariableName[i] << ";\n";
    }
  }
  outstream << "}\n";

  // create gpoint_v and gdir_v
  outstream << "Vector3D<T> "
               "gpoint_v(T(globalpoints.x()+from_index),T(globalpoints.y()+from_index),T(globalpoints.z()+from_index));"
               "\n";
  outstream << "Vector3D<T> "
               "gdir_v(T(globaldirs.x()+from_index),T(globaldirs.y()+from_index),T(globaldirs.z()+from_index));\n";

  // emit vector code

  pointtransf << "Vector3D<T> tmp_v( gpoint_v.x()";
  if (!fTransalwayszero[0]) {
    pointtransf << "- " << fVecTransVariableName[0] << "\n";
  }
  pointtransf << ", gpoint_v.y()";
  if (!fTransalwayszero[1]) {
    pointtransf << "- " << fVecTransVariableName[1] << "\n";
  }
  pointtransf << ", gpoint_v.z()";
  if (!fTransalwayszero[2]) {
    pointtransf << "- " << fVecTransVariableName[2] << "\n";
  }
  pointtransf << ");\n";

  int rotindex      = 0;
  bool indexseen[3] = {false, false, false};
  // tmp loop
  for (int tmpindex = 0; tmpindex < 3; ++tmpindex) {
    // local loop
    for (int localindex = 0; localindex < 3; ++localindex) {
      std::string op = indexseen[localindex] ? "+=" : "=";
      if (!fRotalwayszero[rotindex]) {
        indexseen[localindex] = true;
        // new line
        pointtransf << "localpoint[" << localindex << "]" << op;
        dirtrans << "localdir[" << localindex << "]" << op;
        //}
        if (fRotalwaysone[rotindex]) {
          pointtransf << "tmp_v[" << tmpindex << "];\n";
          dirtrans << "gdir_v[" << tmpindex << "];\n";
        } else if (fRotalwaysminusone[rotindex]) {
          pointtransf << "-tmp_v[" << tmpindex << "];\n";
          dirtrans << "-gdir_v[" << tmpindex << "];\n";
          //   else if check for plusorminus one --> could just copy sign instead of doing a multiplication

        } else { // generic version
                 // pointtransf << "tmp[" << tmpindex << "] * gRot" << rotindex << "[index];\n";
                 // dirtrans << "globaldir[" << tmpindex << "] * gRot" << rotindex << "[index];\n";
          pointtransf << "tmp_v[" << tmpindex << "] * " << fVecRotVariableName[rotindex] << ";\n";
          dirtrans << "gdir_v[" << tmpindex << "] * " << fVecRotVariableName[rotindex] << ";\n";
        }
      }
      rotindex++;
    }
  }
  outstream << pointtransf.str();
  outstream << dirtrans.str();
}

// function that generates a classification table in form of nested switch statements
// the table will be used to map a geometry path object to a unique integer which indexes
// into an array to fetch a cashed global matrix
// void NavigationSpecializer::GeneratePathClassifierCode(std::list<std::pair<int, std::set<NavigationState::Value_t>>>
// const &pathclassification,
//                                PathLevelIndexMap_t &map)
//{
//   // declare return variable
//   std::cerr << "size_t finalindex=0;\n";
//   int sizeaccum = 1;
//   for( auto levelsetpair : pathclassification ){
//      WriteSwitchStatement(levelsetpair, sizeaccum, std::cerr, map );
//   }
//   // return finalindex;
//   std::cerr << "return finalindex;\n";
//}

void NavigationSpecializer::DumpDisclaimer(std::ostream &outstream)
{
  outstream << "// The following code is an autogenerated and specialized Navigator \n";
  outstream << "// obtained from the NavigationSpecializerService ( which is part of VecGeom )\n";
  outstream << "// ADD INFORMATION ABOUT INPUT FILES, DATE, MD5 HASH OF INPUTFILES\n";
  outstream << "// ADD INFORMATION ABOUT WHICH PARTS HAVE BEEN SPECIALIZED \n";
  outstream << "// DO NOT MODIFY THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING \n";
  outstream << "\n";
}

void NavigationSpecializer::DumpClassOpening(std::ostream &outstream)
{
  outstream << "class " << fClassName << " : public VNavigatorHelper<" << fClassName << ","
            << fLogicalVolume->GetUnplacedVolume()->IsConvex() << "> {\n";
}

void NavigationSpecializer::DumpConstructor(std::ostream &outstream) const
{
  // need a constructor only in case we rely on a basic navigator
  if (fUseBaseNavigator) {
    // declaration of base navigator
    std::string basenavtype(fBaseNavigator + "<>");
    outstream << basenavtype << " const & fBaseNavigator;\n";
    outstream << fClassName << "() : fBaseNavigator((" << basenavtype << " const &)*" << basenavtype
              << "::Instance()) {}\n";
  }
}

void NavigationSpecializer::DumpClassClosing(std::ostream &outstream)
{
  outstream << "}; // end class\n";
}

void NavigationSpecializer::DumpIncludeFiles(std::ostream &outstream)
{
  outstream << "#include \"navigation/VNavigator.h\"\n";
  outstream << "#include \"navigation/NavigationState.h\"\n";
  outstream << "#include \"base/Transformation3D.h\"\n";
  outstream << "#include \"management/GeoManager.h\"\n";
  // outstream << "#include <Vc/Vc>\n";
  outstream << "// more relevant includes to be figures out ... \n";
  // for the moment I am putting some hard coded lists
  // we should rather figure out how to generate this dynamically
  outstream << "#include \"volumes/Box.h\"\n";
  outstream << "#include \"volumes/Trapezoid.h\"\n";
  outstream << "#include \"volumes/Tube.h\"\n";
  outstream << "#include \"volumes/Polycone.h\"\n";
  outstream << "#include \"volumes/Polyhedron.h\"\n";
  outstream << "#include \"volumes/Trd.h\"\n";
  outstream << "#include \"volumes/Cone.h\"\n";
  outstream << "#include \"volumes/BooleanVolume.h\"\n";

  outstream << "#include \"navigation/SimpleSafetyEstimator.h\"\n";
}

void NavigationSpecializer::DumpNamespaceOpening(std::ostream &outstream)
{
  outstream << "\n\n";
  outstream << "namespace vecgeom {\n";
  outstream << "inline namespace VECGEOM_IMPL_NAMESPACE {\n";
}

void NavigationSpecializer::DumpNamespaceClosing(std::ostream &outstream)
{
  outstream << "}} // end namespace\n";
}

void NavigationSpecializer::DumpPrivateClassDefinitions(std::ostream &outstream)
{
  outstream << "private:\n";
  DumpPathToIndexFunction(outstream);
  DumpStaticConstExprData(outstream);
  DumpConstructor(outstream);
}

void NavigationSpecializer::DumpStaticInstanceFunction(std::ostream &outstream)
{
  outstream << "\n";
  outstream << "static VNavigator *Instance(){\n"
            << "static " << fClassName << " instance;\n"
            << "return &instance;}\n";
  outstream << "\n";
}

typedef std::map<size_t, std::map<NavigationState::Value_t, size_t>> PathLevelIndexMap_t;
template <typename Stream>
void WriteSwitchStatement(std::pair<int, std::set<NavigationState::Value_t>> const &onelevelclassification,
                          int &sizeaccum, Stream &stream, PathLevelIndexMap_t &map)
{
  stream << "{\n"; // anonymous scope;
  stream << "size_t levelindex;\n";
  stream << "switch (path->At( " << onelevelclassification.first << " )){\n";
  int counter = 0;

  // fill (in-memory map) at the same time
  map.insert(
      PathLevelIndexMap_t::value_type(onelevelclassification.first, std::map<NavigationState::Value_t, size_t>()));

  // iterate over possible values
  for (auto value : onelevelclassification.second) {
    stream << " case " << value << " : { levelindex = " << counter << "; break; }\n";

    // fill (in-memory map) at the same time
    map[onelevelclassification.first][value] = counter;

    counter++;
  }
  stream << "}\n";
  // include this part into the final index
  stream << "finalindex += levelindex * " << sizeaccum << ";\n";
  stream << "}\n"; // close anonymous scope
  sizeaccum *= onelevelclassification.second.size();
}

void NavigationSpecializer::DumpPathToIndexFunction(std::ostream &outstream)
{
  outstream << "\n";
  outstream << "// automatically generated function to transform a path for " << fLogicalVolumeName
            << " into an array index\n";
  outstream << "static size_t PathToIndex( NavigationState const *path ){\n";
  outstream << "size_t finalindex=0;\n";
  int sizeaccum = 1;
  for (auto &mapelement : fIndexMap) {
    outstream << "{\n"; // anonymous scope;
    outstream << "// it might be better to init to -1 ( to detect possible inconsitencies )\n";
    outstream << "size_t levelindex(0);\n";
    outstream << "switch (path->ValueAt( " << mapelement.first << " )){\n";
    // iterate over possible values
    for (auto &valueindexpair : mapelement.second) {
      outstream << " case " << valueindexpair.first << " : { levelindex = " << valueindexpair.second << "; break; }\n";
    }
    outstream << "}\n";
    // include this part into the final index
    outstream << "finalindex += levelindex * " << sizeaccum << ";\n";
    outstream << "}\n"; // close anonymous scope
    sizeaccum *= mapelement.second.size();
  }
  // return finalindex;
  outstream << "return finalindex;\n";
  outstream << "}\n";
  outstream << "\n";
}

void NavigationSpecializer::DumpPublicClassDefinitions(std::ostream &outstream)
{
  outstream << "public:\n";
  outstream << "static constexpr const char *gClassNameString=\"" << fClassName << "\";\n";
  outstream << "typedef SimpleSafetyEstimator SafetyEstimator_t;\n"; // static constexpr const char
                                                                     // *gClassNameString=\"" << fClassName << "\";\n";
  DumpStaticInstanceFunction(outstream);
  DumpStaticTreatGlobalToLocalTransformationFunction(outstream);
  DumpStaticTreatDistanceToMotherFunction(outstream);
  DumpStaticPrepareOutstateFunction(outstream);
  DumpLocalHitDetectionFunction(outstream);
  DumpRelocateMethod(outstream);

  //  DumpSafetyFunctionDeclaration(outstream);
  //  DumpLocalSafetyFunctionDeclaration(outstream);
}

void NavigationSpecializer::DumpClassDefinitions(std::ostream &outstream)
{
  DumpPrivateClassDefinitions(outstream);
  DumpPublicClassDefinitions(outstream);
}

void NavigationSpecializer::DumpStaticConstExprData(std::ostream &outstream)
{
  outstream << "\n";
  outstream << fStaticArraysInitStream.str();
  outstream << "\n";
}

void NavigationSpecializer::DumpStaticConstExprVariableDefinitions(std::ostream &outstream)
{
  outstream << "\n";
  outstream << fStaticArraysDefinitions.str();
  outstream << "\n";
}

void NavigationSpecializer::ProduceSpecializedNavigator(LogicalVolume const *lvol, std::ostream &outstream)
{
  // start with the analysis
  fLogicalVolume = lvol;

  // dump C++ code
  fLogicalVolumeName = lvol->GetLabel();
  fClassName         = fLogicalVolumeName + "Navigator";
  // fClassName = "GeneratedNavigator";

  AnalyseLogicalVolume();

  DumpDisclaimer(outstream);
  DumpIncludeFiles(outstream);
  DumpNamespaceOpening(outstream);
  DumpClassOpening(outstream);
  DumpClassDefinitions(outstream);
  DumpClassClosing(outstream);
  DumpStaticConstExprVariableDefinitions(outstream);
  DumpNamespaceClosing(outstream);
}

void NavigationSpecializer::AnalyseLogicalVolume()
{
  // generate all possible geometry paths for this volume
  std::list<NavigationState *> allpaths;
  GeoManager::Instance().getAllPathForLogicalVolume(fLogicalVolume, allpaths);
  fNumberOfPossiblePaths = allpaths.size();

  AnalysePaths(allpaths);

  // analyse path transitions
  // try to read from generated outpaths ( add error handling these files do not exist )
  int npointsin, ndepthin;
  int npointsout, ndepthout;
  NavStatePool::ReadDepthAndCapacityFromFile(fInStateFileName, npointsin, ndepthin);
  NavStatePool::ReadDepthAndCapacityFromFile(fOutStateFileName, npointsout, ndepthout);

  if (npointsin != npointsout || ndepthin != ndepthout || ndepthin != GeoManager::Instance().getMaxDepth()) {
    VECGEOM_LOG(critical) << "Failed to read state files";
    std::exit(1);
  }
  NavStatePool inpool(npointsin, GeoManager::Instance().getMaxDepth());
  NavStatePool outpool(npointsout, GeoManager::Instance().getMaxDepth());
  auto s1 = inpool.FromFile(fInStateFileName);
  auto s2 = outpool.FromFile(fOutStateFileName);
  if (s1 != npointsin || s2 != npointsin) {
    VECGEOM_LOG(critical) << "Failed to read state files";
    std::exit(1);
  }
  std::cerr << "Read " << npointsin << " states to analyse\n";
  AnalyseTargetPaths(inpool, outpool);
}

void NavigationSpecializer::AddToIndexMap(size_t level, NavigationState::Value_t keyvalue)
{
  if (fIndexMap.find(level) == fIndexMap.end()) {
    // level does not exist; so create it
    fIndexMap.insert(std::pair<size_t, std::map<NavigationState::Value_t, size_t>>(
        level, std::map<NavigationState::Value_t, size_t>()));
  }
  // keyvalue should not be in level already
  if (fIndexMap[level].find(keyvalue) != fIndexMap[level].end()) {
    std::cerr << "trying to insert value which already exists\n";
  } else {
    size_t index = fIndexMap[level].size();
    fIndexMap[level].insert(std::pair<NavigationState::Value_t, size_t>(keyvalue, index));
  }
}

void NavigationSpecializer::AnalyseIndexCorrelations(std::list<NavigationState *> const &paths)
{
  std::list<size_t> removalcandidates;
  // count real combinations versus assumed combinations

  // analyse any two pairs of distinctive levels in the keymap and see if
  // these levels are actually independent in the paths
  for (auto &level1 : fIndexMap)
    for (auto &level2 : fIndexMap) {
      if (level1 < level2) {
        // get combinationcount as assumed from IndexMap
        size_t combinationcount = fIndexMap[level1.first].size() * fIndexMap[level2.first].size();

        // check real number of combinations in the paths
        std::set<std::pair<NavigationState::Value_t, NavigationState::Value_t>> realcombinations;
        for (auto &path : paths) {
          realcombinations.insert(std::pair<NavigationState::Value_t, NavigationState::Value_t>(
              path->ValueAt(level1.first), path->ValueAt(level2.first)));
        }
        std::cerr << level1.first << ";" << level2.first << " : COMB " << combinationcount << " vs REAL COMB "
                  << realcombinations.size() << "\n";

        if (combinationcount != realcombinations.size()) {
          // look if the redundancy is trivially decomposable ( which should be the case when the number of real
          // combinations is divisible
          // by the size of level1 ( and hence level 2)
          if (combinationcount / fIndexMap[level2.first].size() == realcombinations.size()) {
            std::cerr << " level1 and level2 are trivially correlated --> mark as removal candidate\n";
            removalcandidates.push_back(level2.first);
          } else {
            std::cerr << " level " << level1.first << " and level " << level2.first
                      << " are not trivially correlated --> need to implement a more sophisticated indexmap\n";
          }
        }
      }
    }

  // remove trivial redundancies
  for (auto &r : removalcandidates) {
    fIndexMap.erase(r);
  }
}

// calculates an index from a states using a prefilled PathLevelIndexMap
size_t NavigationSpecializer::PathToIndex(NavigationState const *state)
{
  std::list<std::pair<size_t, size_t>> indexlist;
  auto cl = state->GetCurrentLevel();
  for (auto level = decltype(cl){0}; level < cl; ++level) {
    if (fIndexMap.find(level) != fIndexMap.end()) { // this level is a distinctive feature
      auto &m           = fIndexMap[level];
      size_t levelindex = m[state->ValueAt(level)];
      indexlist.push_back(std::pair<size_t, size_t>(levelindex, fIndexMap[level].size()));
    }
  }
  int finalindex = 0;
  int size       = 1;
  for (auto &pair : indexlist) {
    finalindex += pair.first * size;
    size *= pair.second;
  }
  return finalindex;
}

void NavigationSpecializer::AnalysePaths(std::list<NavigationState *> const &paths)
{

  // analyse level
  auto level     = paths.front()->GetCurrentLevel();
  fGeometryDepth = level;
  bool samelevel(true);
  for (auto path : paths) {
    if (path->GetCurrentLevel() != level) {
      samelevel = false;
    }
  }
  //
  if (samelevel)
    std::cerr << "all " << paths.size() << " paths have the same level " << level << "\n";
  else {
    std::cerr << "paths have different levels --> NOT SUPPORTED AT THE MOMENT \n";
    return;
  }

  // in-memory - table lookup
  PathLevelIndexMap_t indexMap;

  // try to find a distinctive feature/classifier for paths
  // we can then try to write a function that maps the product space of distinctive levels
  // onto a single index which can be used to fetch a cached global matrix
  if (samelevel) {
    for (decltype(level) l(0); l < level; ++l) {
      std::set<NavigationState::Value_t> values;
      if (PathsAreDifferentAtGivenLevel(paths, l, values) > 0) {
        for (auto &v : values)
          AddToIndexMap(l, v);
      }
    }
  }

  // iteratively check if we reach a situation where all dependencies have been resolved
  AnalyseIndexCorrelations(paths);
  AnalyseIndexCorrelations(paths);
  AnalyseIndexCorrelations(paths);

  // cross check if space of indices map matches size of paths
  // if not there are likely some correlations in the indices which have to be reduced (to be supported)
  bool redundency = false;
  size_t maxindex = 0;
  size_t minindex = 100000000;
  for (auto &path : paths) {
    size_t index = PathToIndex(path);
    if (index >= paths.size()) {
      redundency = true;
    }
    minindex = std::min(index, minindex);
    maxindex = std::max(index, maxindex);
  }
  if (redundency) {
    std::cerr << "maxindex " << maxindex << "  vs " << paths.size() << "\n";
    std::cerr << "redundency of factor " << (maxindex + 1) / (paths.size())
              << " detected --> ask for an implementation !!\n";
    return;
  }
  if (minindex != 0) {
    std::cerr << "minindex not zero\n";
    return;
  }
  if (maxindex != paths.size() - 1) {
    std::cerr << "maxindex not size - 1\n";
    return;
  }

  // analyse global matrix and generate static data with right index
  bool rotalwayszero[9]          = {true, true, true, true, true, true, true, true, true};
  bool rotalwaysone[9]           = {true, true, true, true, true, true, true, true, true};
  bool rotalwaysminusone[9]      = {true, true, true, true, true, true, true, true, true};
  bool rotalwaysminusoneorone[9] = {true, true, true, true, true, true, true, true, true};
  bool transalwayszero[3]        = {true, true, true};
  for (auto path : paths) {
    Transformation3D m;
    path->TopMatrix(m);
    for (int i = 0; i < 9; ++i) {
      if (std::abs(m.Rotation(i)) > 1E-9) {
        rotalwayszero[i] = false;
      }
      if (!(std::abs(m.Rotation(i) - 1.) < 1E-9)) {
        rotalwaysone[i] = false;
      }
      if (!(std::abs(m.Rotation(i) + 1) < 1E-9)) {
        rotalwaysminusone[i] = false;
      }
      if (!((std::abs(m.Rotation(i) + 1) < 1E-9) || (std::abs(m.Rotation(i) - 1) < 1E-9))) {
        rotalwaysminusoneorone[i] = false;
      }
    }
    for (int i = 0; i < 3; ++i) {
      if (std::abs(m.Translation(i)) > 1E-9) {
        transalwayszero[i] = false;
      }
    }
  }

  // print out result
  for (int i = 0; i < 3; ++i) {
    if (transalwayszero[i]) {
      std::cerr << " trans[" << i << "] is zero\n";
    }
  }

  // prepare specialized transformation routine
  fTransformationCode << "Vector3D<Precision> tmp( globalpoint[0]";
  if (!transalwayszero[0]) fTransformationCode << "- gTrans0[index]";
  fTransformationCode << ", globalpoint[1]";
  if (!transalwayszero[1]) fTransformationCode << "- gTrans1[index]";
  fTransformationCode << ", globalpoint[2]";
  if (!transalwayszero[2]) fTransformationCode << "- gTrans2[index]";
  fTransformationCode << ");\n";

  // for vectorized version this is a bit different ( we need a SIMD global point first of all )
  fVectorTransformationCode << "Vector3D<Vc::double_v> gpoint_v("
                            << "Vc::double_v(globalpoints.x()+i),"
                            << "Vc::double_v(globalpoints.y()+i),"
                            << "Vc::double_v(globalpoints.z()+i));\n";
  fVectorTransformationCode << "Vector3D<Vc::double_v> tmp( gpoint_v.x()";
  if (!transalwayszero[0]) fVectorTransformationCode << "- gTrans0_v";
  fVectorTransformationCode << ", gpoint_v.y()";
  if (!transalwayszero[1]) fVectorTransformationCode << "- gTrans1_v";
  fVectorTransformationCode << ", gpoint_v.z()";
  if (!transalwayszero[2]) fVectorTransformationCode << "- gTrans2_v";
  fVectorTransformationCode << ");\n";

  int rotindex      = 0;
  bool indexseen[3] = {false, false, false};
  // tmp loop
  for (int tmpindex = 0; tmpindex < 3; ++tmpindex) {
    // local loop
    for (int localindex = 0; localindex < 3; ++localindex) {
      std::string op = indexseen[localindex] ? "+=" : "=";
      if (!rotalwayszero[rotindex]) {
        indexseen[localindex] = true;
        // new line
        fTransformationCode << "localpoint[" << localindex << "]" << op;
        fTransformationCodeDir << "localdir[" << localindex << "]" << op;
        fVectorTransformationCode << "localpoint[" << localindex << "]" << op;
        //}
        if (rotalwayszero[rotindex]) {
          fTransformationCode << "0;\n";
          fTransformationCodeDir << "0;\n";
          fVectorTransformationCode << "0;\n";
        }
        if (rotalwaysone[rotindex]) {
          fTransformationCode << "tmp[" << tmpindex << "];\n";
          fTransformationCodeDir << "globaldir[" << tmpindex << "];\n";
          fVectorTransformationCode << "tmp[" << tmpindex << "];\n";
        } else if (rotalwaysminusone[rotindex]) {
          fTransformationCode << "-tmp[" << tmpindex << "];\n";
          fTransformationCodeDir << "-globaldir[" << tmpindex << "];\n";
          fVectorTransformationCode << "-tmp[" << tmpindex << "];\n";
          //   else if check for plusorminus one --> could just copy sign instead of doing a multiplication

        } else { // generic version
          fTransformationCode << "tmp[" << tmpindex << "] * gRot" << rotindex << "[index];\n";
          fTransformationCodeDir << "globaldir[" << tmpindex << "] * gRot" << rotindex << "[index];\n";
          fVectorTransformationCode << "tmp[" << tmpindex << "] * gRot" << rotindex << "_v;\n";
        }
      }
      rotindex++;
    }
  }

  for (int i = 0; i < 9; ++i) {
    if (rotalwayszero[i]) {
      std::cerr << " rot[" << i << "] is zero\n";
    }
    if (rotalwaysone[i]) {
      std::cerr << " rot[" << i << "] is one\n";
    }
    if (rotalwaysminusone[i]) {
      std::cerr << " rot[" << i << "] is minus one\n";
    }
    if (rotalwaysminusoneorone[i]) {
      std::cerr << " rot[" << i << "] is minus one or one\n";
    }
  }

  // generate the static variable data one by one
  // first the translation data
  for (int i = 0; i < 3; ++i) {
    if (!transalwayszero[i]) {
      std::string variable("gTrans" + std::to_string(i));
      // export the following into a lambda
      std::vector<double> values(paths.size());
      fGlobalTransData.ReserveTrans(i, paths.size());
      for (auto path : paths) {
        Transformation3D m;
        path->TopMatrix(m);
        size_t index = PathToIndex(path);
        if (index >= paths.size()) std::cerr << "SCHEISSE " << index << " \n";
        VECGEOM_ASSERT(index < paths.size());
        values[index] = m.Translation(i);
        fGlobalTransData.SetTransCoef(i, index, m.Translation(i));
      }
      //
      if (CollectionIsConstant(values, values[0])) {
        fStaticArraysInitStream << "// **** HEY: THESE VALUES ARE ALL THE SAME ---> can safe memory\n";
      }
      printlambda(variable, values, fStaticArraysInitStream);
      //  fStaticArraysDefinitions << "constexpr double " << fClassName << "::" << variable << "[" << values.size()
      //                          << "];\n";
      fTransformVariables.push_back(variable);
    }
  }

  // then rotation data
  for (int i = 0; i < 9; ++i) {
    if (!rotalwayszero[i]) {
      std::string variable("gRot" + std::to_string(i));
      // export the following into a lambda
      std::vector<double> values(paths.size());
      fGlobalTransData.ReserveRot(i, paths.size());
      for (auto &path : paths) {
        Transformation3D m;
        path->TopMatrix(m);
        auto index    = PathToIndex(path);
        values[index] = m.Rotation(i);
        fGlobalTransData.SetRotCoef(i, index, m.Rotation(i));
      }
      //
      if (CollectionIsConstant(values, values[0])) {
        fStaticArraysInitStream << "// **** HEY: THESE VALUES ARE ALL THE SAME ---> can safe memory\n";
      }
      printlambda(variable, values, fStaticArraysInitStream);
      //   fStaticArraysDefinitions << "constexpr double " << fClassName << "::" << variable << "[" << values.size()
      //                           << "];\n";
      fTransformVariables.push_back(variable);
    }
  }

  fGlobalTransData.Analyse();
  fGlobalTransData.Print();
  fGlobalTransData.EmitTableDeclaration(fStaticArraysInitStream);
  fGlobalTransData.EmitTableDefinition(fClassName, fStaticArraysDefinitions);
  std::stringstream ss;
  fGlobalTransData.EmitScalarGlobalTransformationCode(ss);
  std::cerr << ss.str() << "\n";
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v)
{

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i)
    idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1].first > v[i2].first; });

  return idx;
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v, const std::vector<size_t> &v2)
{

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i)
    idx[i] = i;

  // sort indexes based on comparing values in v then in v2
  std::sort(idx.begin(), idx.end(), [&v, &v2](size_t i1, size_t i2) {
    if (v[i1].first == v[i2].first) {
      return v2[i1] > v2[i2];
    } else {
      return v[i1].first > v[i2].first;
    }
  });
  return idx;
}

void NavigationSpecializer::AnalyseTargetPaths(NavStatePool const &inpool, NavStatePool const &outpool)
{
  // the purpose of this function is to generate a list of possible target states
  // including their corresponding matrix transformations
  // the information produced here shall accelerate the relocation step of navigation
  std::cerr << " --- ANALYSIS OF PATH TRANSITIONS ---- \n";

  // analyse global matrix and generate static data with right index
  bool rotalwayszero[9]          = {true, true, true, true, true, true, true, true, true};
  bool rotalwaysone[9]           = {true, true, true, true, true, true, true, true, true};
  bool rotalwaysminusone[9]      = {true, true, true, true, true, true, true, true, true};
  bool rotalwaysminusoneorone[9] = {true, true, true, true, true, true, true, true, true};
  bool transalwayszero[3]        = {true, true, true};

  std::set<VPlacedVolume const *> pset;
  std::set<LogicalVolume const *> lset;
  std::set<std::string> pathset;
  std::set<std::string> crossset;

  std::vector<std::string> matrixstrings;
  std::vector<Transformation3D> matrixcache;
  // mapping of navigation state index x transitionindex -> delta matrices
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> mapping;

  std::vector<size_t> transitioncounter; // counts the number of transitions of each type
  // could be used as a sorting criterion to optimize early returns from relocation

  for (auto j = decltype(outpool.capacity()){0}; j < outpool.capacity(); ++j) {
    std::stringstream pathstringstream2;
    auto *navstate = outpool[j];
    navstate->printValueSequence(pathstringstream2);
    pset.insert(navstate->Top());
    if (navstate->Top() != nullptr) lset.insert(navstate->Top()->GetLogicalVolume());
    pathset.insert(pathstringstream2.str());

    std::stringstream pathstringstream1;
    auto *instate = inpool[j];
    instate->printValueSequence(pathstringstream1);
    crossset.insert(pathstringstream1.str() + " -- " + pathstringstream2.str());

    // the string characterising the relative path difference between instate and outstate
    std::string deltapathstring = instate->RelativePath(*navstate);
    unsigned int transitionindex;
    auto found = std::find(fTransitionStrings.begin(), fTransitionStrings.end(), deltapathstring);
    if (found == fTransitionStrings.end()) {
      // we found a new transition path
      fTransitionStrings.push_back(deltapathstring);
      // register information about the new state

      // shape-type
      std::stringstream type;
      navstate->Top()->PrintType(type);
      transitionindex = fTransitionTargetTypes.size();
      fTransitionTargetTypes.push_back(FinalDepthShapeType_t(navstate->GetCurrentLevel(), type.str()));

      fTargetVolIds.push_back(navstate->Top()->id());
      transitioncounter.push_back(0);
    } else {
      transitionindex = std::distance(fTransitionStrings.begin(), found);
    }
    transitioncounter[transitionindex]++;

    Transformation3D deltamatrix;
    instate->DeltaTransformation(*navstate, deltamatrix);

    // Transformation3D const & deltam = deltamatrix;
    // deltam.Print();
    // analyse matrices on the fly ( this code is duplicated --> fix this )
    for (int i = 0; i < 9; ++i) {
      if (std::abs(deltamatrix.Rotation(i)) > 1E-9) {
        rotalwayszero[i] = false;
      }
      if (!(std::abs(deltamatrix.Rotation(i) - 1.) < 1E-9)) {
        rotalwaysone[i] = false;
      }
      if (!(std::abs(deltamatrix.Rotation(i) + 1) < 1E-9)) {
        rotalwaysminusone[i] = false;
      }
      if (!((std::abs(deltamatrix.Rotation(i) + 1) < 1E-9) || (std::abs(deltamatrix.Rotation(i) - 1) < 1E-9))) {
        rotalwaysminusoneorone[i] = false;
      }
    }
    for (int i = 0; i < 3; ++i) {
      if (std::abs(deltamatrix.Translation(i)) > 1E-9) {
        transalwayszero[i] = false;
      }
    }

    std::stringstream matrixstream;
    deltamatrix.Print(matrixstream);
    auto found2 = std::find(matrixstrings.begin(), matrixstrings.end(), matrixstream.str());
    unsigned int matrixindex;
    if (found2 == matrixstrings.end()) {
      matrixindex = matrixcache.size();
      matrixstrings.push_back(matrixstream.str());
      matrixcache.push_back(deltamatrix);
    } else {
      // this matrix exists already
      matrixindex = std::distance(matrixstrings.begin(), found2);
    }
    // update the map of instances x transitions -> global matrices
    // std::stringstream mappingstream;
    auto transition        = std::make_tuple(PathToIndex(instate), transitionindex, matrixindex);
    auto transition_exists = std::find(mapping.begin(), mapping.end(), transition);
    if (transition_exists == mapping.end()) {
      mapping.push_back(transition);
    }
  }

  // make up lookup table to fetch
  // the transition matrix given an instate and a transition
  // ( we have to see whether we really need constant access or a std::map or hash_map would be enough )
  std::vector<std::vector<short>> deltamatrixmapping(fNumberOfPossiblePaths);
  for (auto i = decltype(fNumberOfPossiblePaths){0}; i < fNumberOfPossiblePaths; ++i) {
    deltamatrixmapping[i].resize(fTransitionStrings.size(), -1); // -1 means uninitialized
  }

  // CHECK THIS (verify this stupid mapping ) !!!!!
  // very brute force: scan the complete index space and see if transition was recored in "mapping" as a string
  // I know this can be done more efficient
  for (auto i = decltype(fNumberOfPossiblePaths){0}; i < fNumberOfPossiblePaths; ++i) {
    for (auto j = decltype(fTransitionStrings.size()){0}; j < fTransitionStrings.size(); ++j) {
      for (auto &t : mapping) {
        if (i == std::get<0>(t) && j == std::get<1>(t)) deltamatrixmapping[i][j] = std::get<2>(t);
      }
    }
  }
  // generate static map in C++ ( combine with previous loop )?
  fStaticArraysInitStream << "static constexpr short deltamatrixmapping[" << fNumberOfPossiblePaths << "]["
                          << fTransitionStrings.size() << "] = {";
  for (auto i = decltype(fNumberOfPossiblePaths){0}; i < fNumberOfPossiblePaths; ++i) {
    fStaticArraysInitStream << "{";
    for (auto j = decltype(fTransitionStrings.size()){0}; j < fTransitionStrings.size(); ++j) {
      fStaticArraysInitStream << deltamatrixmapping[i][j];
      if (j < fTransitionStrings.size() - 1) fStaticArraysInitStream << ",";
    }
    fStaticArraysInitStream << "}";
    if (i < fNumberOfPossiblePaths - 1) fStaticArraysInitStream << ",";
  }
  fStaticArraysInitStream << "};\n";
  fStaticArraysDefinitions << "constexpr short " << fClassName << "::deltamatrixmapping[" << fNumberOfPossiblePaths
                           << "][" << fTransitionStrings.size() << "];\n";
  //

  std::cerr << "analysis for delta matrices\n";
  for (int i = 0; i < 9; ++i) {
    if (rotalwayszero[i]) {
      std::cerr << " rot[" << i << "] is zero\n";
    }
    if (rotalwaysone[i]) {
      std::cerr << " rot[" << i << "] is one\n";
    }
    if (rotalwaysminusone[i]) {
      std::cerr << " rot[" << i << "] is minus one\n";
    }
    if (rotalwaysminusoneorone[i]) {
      std::cerr << " rot[" << i << "] is minus one or one\n";
    }
  }

  std::cerr << " size of diffset " << fTransitionStrings.size() << "\n";
  std::cerr << " size of matrixset " << matrixstrings.size() << "\n";
  std::cerr << " size of target pset " << pset.size() << "\n";
  std::cerr << " size of target lset " << lset.size() << "\n";
  std::cerr << " size of target state set " << pathset.size() << "\n";
  std::cerr << " total combinations " << crossset.size() << "\n";
  std::cerr << " normalized per input state " << crossset.size() / (1. * pathset.size()) << "\n";

  for (auto &s : crossset) {
    std::cerr << s << "\n";
  }

  size_t index = 0;
  for (auto &s : fTransitionStrings) {
    std::cerr << s << "\t" << transitioncounter[index] << "\n";
    index++;
  }
  for (auto &s : matrixstrings) {
    std::cerr << s << "\n";
  }
  //  for (auto &s : mapping) {
  //    std::cerr << s << "\n";
  //  }

  // sort the possible transitions from most specific ( deepest depth ) to least specific
  // we are not sorting the vector itself but create an "index" vector

  // fTransitionOrder = sort_indexes(fTransitionTargetTypes);
  fTransitionOrder = sort_indexes(fTransitionTargetTypes, transitioncounter);

  // std::sort(fTransitionTargetTypes.begin(), fTransitionTargetTypes.end());
  std::cerr << "transition order\n";
  for (auto &tp : fTransitionTargetTypes) {
    std::cerr << tp.first << " " << tp.second << "\n";
  }
  for (auto &tp : fTransitionOrder) {
    std::cerr << tp << "\n";
  }

  // generate the static variable data for the transition matrices one by one
  // first the translation data
  for (int i = 0; i < 3; ++i) {
    if (!transalwayszero[i]) {
      std::string variable("gDeltaTrans" + std::to_string(i));
      // export the following into a lambda
      std::vector<double> values(matrixcache.size());
      size_t index = 0;
      for (auto &m : matrixcache) {
        values[index++] = m.Translation(i);
      }
      //
      if (CollectionIsConstant(values, values[0])) {
        fStaticArraysInitStream << "// **** HEY: THESE VALUES ARE ALL THE SAME ---> can safe memory\n";
      }
      printlambda(variable, values, fStaticArraysInitStream);
      fStaticArraysDefinitions << "constexpr double " << fClassName << "::" << variable << "[" << values.size()
                               << "];\n";
      fTransformVariables.push_back(variable);
    }
  }

  // then rotation data
  for (int i = 0; i < 9; ++i) {
    if (!rotalwayszero[i]) {
      std::string variable("gDeltaRot" + std::to_string(i));
      // export the following into a lambda
      std::vector<double> values(matrixcache.size());
      size_t index = 0;
      for (auto &m : matrixcache) {
        values[index++] = m.Rotation(i);
      }
      //
      if (CollectionIsConstant(values, values[0])) {
        fStaticArraysInitStream << "// **** HEY: THESE VALUES ARE ALL THE SAME ---> can safe memory\n";
      }
      printlambda(variable, values, fStaticArraysInitStream);
      fStaticArraysDefinitions << "constexpr double " << fClassName << "::" << variable << "[" << values.size()
                               << "];\n";
      fTransformVariables.push_back(variable);
    }
  }

  // generate the transformation code
  // prepare specialized transformation routine
  fDeltaTransformationCode << "Vector3D<Precision> tmp( pointafterboundary[0]";
  if (!transalwayszero[0]) fDeltaTransformationCode << "- gDeltaTrans0[index]";
  fDeltaTransformationCode << ", pointafterboundary[1]";
  if (!transalwayszero[1]) fDeltaTransformationCode << "- gDeltaTrans1[index]";
  fDeltaTransformationCode << ", pointafterboundary[2]";
  if (!transalwayszero[2]) fDeltaTransformationCode << "- gDeltaTrans2[index]";
  fDeltaTransformationCode << ");\n";
  fDeltaTransformationCode << "Vector3D<Precision> localpoint;\n";

  // for vectorized version this is a bit different ( we need a SIMD global point first of all )
  //   fVectorTransformationCode << "Vector3D<Vc::double_v> gpoint_v("
  //                             << "Vc::double_v(globalpoints.x()+i),"
  //                             << "Vc::double_v(globalpoints.y()+i),"
  //                             << "Vc::double_v(globalpoints.z()+i));\n";
  //   fVectorTransformationCode << "Vector3D<Vc::double_v> tmp( gpoint_v.x()";
  //   if (!transalwayszero[0])
  //     fVectorTransformationCode << "- gTrans0_v";
  //   fVectorTransformationCode << ", gpoint_v.y()";
  //   if (!transalwayszero[1])
  //     fVectorTransformationCode << "- gTrans1_v";
  //   fVectorTransformationCode << ", gpoint_v.z()";
  //   if (!transalwayszero[2])
  //     fVectorTransformationCode << "- gTrans2_v";
  //   fVectorTransformationCode << ");\n";
  //   fVectorTransformationCode << "Vector3D<Vc::double_v> local(0.);\n";

  int rotindex      = 0;
  bool indexseen[3] = {false, false, false};
  // tmp loop
  for (int tmpindex = 0; tmpindex < 3; ++tmpindex) {
    // local loop
    for (int localindex = 0; localindex < 3; ++localindex) {
      std::string op = indexseen[localindex] ? "+=" : "=";
      if (!rotalwayszero[rotindex]) {
        indexseen[localindex] = true;
        // new line
        fDeltaTransformationCode << "localpoint[" << localindex << "]" << op;
        //            fVectorTransformationCode << "localpoint[" << localindex << "]" << op;
        //}
        if (rotalwayszero[rotindex]) {
          fTransformationCode << "0;\n";
          //              fVectorTransformationCode << "0;\n";
        }
        if (rotalwaysone[rotindex]) {
          fDeltaTransformationCode << "tmp[" << tmpindex << "];\n";
          //              fVectorTransformationCode << "tmp[" << tmpindex << "];\n";
        } else if (rotalwaysminusone[rotindex]) {
          fDeltaTransformationCode << "-tmp[" << tmpindex << "];\n";
          //              fVectorTransformationCode << "-tmp[" << tmpindex << "];\n";
          //   else if check for plusorminus one --> could just copy sign instead of doing a multiplication

        } else { // generic version
          fDeltaTransformationCode << "tmp[" << tmpindex << "] * gDeltaRot" << rotindex << "[index];\n";
          //              fVectorTransformationCode << "tmp[" << tmpindex << "] * gRot" << rotindex << "_v;\n";
        }
      }
      rotindex++;
    }
  }
}

void NavigationSpecializer::DumpSafetyFunctionDeclaration(std::ostream &outstream)
{
  outstream << "\n";
  outstream << "virtual\n"
            << "Precision\n"
            << "ComputeSafety(Vector3D<Precision> const& globalpoint, NavigationState const &state) const override {\n";

  outstream << "size_t index = PathToIndex(&state);\n";

  // put specialized global to local transformation code
  outstream << fTransformationCode.str();

  DumpTransformationAsserts(outstream);

  // call local function
  outstream << "return " << fClassName << "::ComputeSafetyForLocalPoint(local, state.Top());\n";

  outstream << "}\n"; // function closing
  outstream << "\n";
}

void NavigationSpecializer::DumpLocalSafetyFunctionDeclaration(std::ostream &outstream)
{
  outstream << "VECGEOM_FORCE_INLINE\n"
            << "virtual\n"
            << "Precision\n"
            << "ComputeSafetyForLocalPoint(Vector3D<Precision> const& localpoint, VPlacedVolume const *pvol) const "
               "override {\n";

  // get the stream --> very cumbersome
  VPlacedVolume *pvol = fLogicalVolume->Place();
  std::stringstream shapetypestream;
  pvol->PrintType(shapetypestream);
  delete pvol;
  std::string shapetype = shapetypestream.str();

  outstream << "double s = ((" << shapetype << "*)pvol)->" << shapetype << "::SafetyToOut(localpoint);\n";

  // analyse daughters
  // this is very crude --> we trust the shapetypes given back from the shape
  // the only optimization here is that we get rid of the virtual function and that we unroll the loop

  // but we could also analyse it deeply ourselves here and specialize ourselves
  // future possible optimizations include the following:
  // a) vectorization of similar daughter shapes
  // b) constant extraction and propagation ( detect of shapes use same rotation and pull them out ... )
  // c) put actual numeric values of shapes which may simplify the algorithms ... ( extreme shape specialization )
  auto daughters  = fLogicalVolume->GetDaughtersp();
  auto ndaughters = daughters->size();
  if (ndaughters > 0) {
    outstream << "auto daughters = pvol->GetLogicalVolume()->GetDaughtersp();\n";

    // first pass to separate list of polymorphic shapes into separated loops over the same kind
    std::string currenttype;
    std::list<std::pair<std::string, size_t>> looplist; // list for loop splitting
    shapetypestream.str("");
    (*daughters)[0]->PrintType(shapetypestream);
    currenttype  = shapetypestream.str();
    size_t count = 1;
    for (auto i = decltype(ndaughters){1}; i < ndaughters; ++i) {
      auto daughter = (*daughters)[i];
      shapetypestream.str("");
      daughter->PrintType(shapetypestream);
      std::string thistype = shapetypestream.str();
      if (currenttype.compare(thistype) == 0)
        count++;
      else {
        looplist.push_back(std::pair<std::string, size_t>(currenttype, count));
        count       = 1;
        currenttype = thistype; /*somehow next type*/
      }
    }
    looplist.push_back(std::pair<std::string, size_t>(currenttype, count));

    // print some info about the loop list
    for (auto &pair : looplist) {
      std::cerr << "## " << pair.first << " " << pair.second << "\n";
    }

    if (fUnrollLoops) {
      for (auto i = decltype(ndaughters){0}; i < ndaughters; ++i) {
        shapetypestream.str(""); // this clears the stream
        auto daughter = (*daughters)[i];
        daughter->PrintType(shapetypestream);
        shapetype = shapetypestream.str();
        outstream << "s = Min(s, ((" << shapetype << "*) (*daughters)[" << i << "])->" << shapetype
                  << "::SafetyToIn(localpoint));\n";
      }
    } else {
      // emit smaller loops
      int offset = 0;
      for (auto &pair : looplist) {
        if (pair.second > 1) {
          outstream << "for(int i = " << offset << ";i<" << offset + pair.second << ";++i){";
          outstream << "s = Min(s, ((" << pair.first << "*) (*daughters)[ i ])->" << pair.first
                    << "::SafetyToIn(localpoint));\n";
          outstream << "} // end loop\n";
        } else {
          outstream << "s = Min(s, ((" << pair.first << "*) (*daughters)[" << offset << "])->" << pair.first
                    << "::SafetyToIn(localpoint));\n";
        }
        offset += pair.second;
      }
    }
  }
  outstream << "return s;\n";
  outstream << "}\n";
}

void NavigationSpecializer::DumpStaticTreatGlobalToLocalTransformationFunction(std::ostream &outstream) const
{

  outstream
      << "template <typename T>\n VECGEOM_FORCE_INLINE\n static void DoGlobalToLocalTransformation(NavigationState "
         "const &in_state,"
      << "Vector3D<T> const &globalpoint, Vector3D<T> const &globaldir,"
      << "Vector3D<T> &localpoint, Vector3D<T> &localdir, NavigationState * internal)  {\n";

  outstream << "auto index = PathToIndex( &in_state );\n";
  // TODO: check if we have to do anything at all ( check for unity )
  outstream << "// caching this index in internal navigationstate for later reuse\n";
  outstream << "// we know that is safe to do this because of static analysis (never do this in user code)\n";
  outstream << "internal->SetCacheValue(index);\n";
  fGlobalTransData.EmitScalarGlobalTransformationCode(outstream);
  outstream << "}\n";
}

void NavigationSpecializer::DumpStaticTreatDistanceToMotherFunction(std::ostream &outstream) const
{
  // this is stupid but no other way at the moment
  VPlacedVolume *pvol = fLogicalVolume->Place();
  std::stringstream shapetypestream;
  pvol->PrintType(shapetypestream);
  delete pvol;
  std::string shapetype = shapetypestream.str();

  outstream << "template <typename T> VECGEOM_FORCE_INLINE static T TreatDistanceToMother(VPlacedVolume const *pvol, "
               "Vector3D<T> const "
               "&localpoint, Vector3D<T> const &localdir, T step_limit) {\n";
  outstream << "T step;\n";
  outstream << "VECGEOM_ASSERT(pvol != nullptr && \"currentvolume is null in navigation\");\n";
  outstream << "step = ((" << shapetype << "*)pvol)->" << shapetype
            << "::DistanceToOut(localpoint, localdir, step_limit);\n";
  outstream << "vecCore::MaskedAssign(step, step < T(0.0), InfinityLength<T>());\n";
  outstream << "return step;\n";
  outstream << "}\n";
}

void NavigationSpecializer::DumpStaticPrepareOutstateFunction(std::ostream &outstream) const
{

  outstream
      << "VECGEOM_FORCE_INLINE static Precision PrepareOutState(NavigationState const &in_state, NavigationState "
         "&out_state, Precision geom_step, Precision step_limit, VPlacedVolume const *hitcandidate, bool &done){\n";
  // now we have the candidates and we prepare the out_state

  outstream
      << "// this is the special part ( fast navigation state copying since we know the depth at compile time )\n";
  outstream << "in_state.CopyToFixedSize<NavigationState::SizeOf(" << fGeometryDepth << ")>(&out_state);\n";

  outstream << "// this is just the first try -- we should factor out the following part which is probably not \n";
  outstream << "// special code\n";
  outstream << "done = false;\n";

  outstream << "if (geom_step == kInfLength && step_limit > 0.) {"
               "geom_step = vecgeom::kTolerance;"
               "out_state.SetBoundaryState(true);"
               "out_state.Pop();"
               "done=true;"
               "return geom_step;"
               "}\n";

  outstream << "// is geometry further away than physics step?\n"
               "// this is a physics step\n";
  outstream << "if (geom_step > step_limit) {"
               "// don't need to do anything\n"
               "geom_step = step_limit;"
               "out_state.SetBoundaryState(false);"
               "return geom_step;"
               "}\n";

  // otherwise it is a geometry step
  outstream << "out_state.SetBoundaryState(true);\n";

  if (fLogicalVolume->GetDaughtersp()->size() > 0) {
    outstream << "if (hitcandidate)"
                 "out_state.Push(hitcandidate);\n";
  }
  outstream << "if (geom_step < 0.) {"
               "// InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );\n"
               "geom_step = 0.;"
               "}"
               "return geom_step;"
               "}\n";
}

void NavigationSpecializer::DumpTransformationAsserts(std::ostream &outstream)
{
  outstream << "\n";
  outstream << "// piece of code which can be activated to check correctness of table lookup plus transformation\n";
  outstream << "#ifdef " << fClassName << "_CHECK_TRANSFORMATION\n";
  outstream << "Transformation3D checkmatrix;\n";
  outstream << "state.TopMatrix(checkmatrix);\n";
  outstream << "Vector3D<Precision> crosschecklocalpoint = checkmatrix.Transform(globalpoint);\n";
  outstream << "VECGEOM_ASSERT( std::abs(crosschecklocalpoint[0] - local[0]) < 1E-9 && \"error in transformation\");\n";
  outstream << "VECGEOM_ASSERT( std::abs(crosschecklocalpoint[1] - local[1]) < 1E-9 && \"error in transformation\");\n";
  outstream << "VECGEOM_ASSERT( std::abs(crosschecklocalpoint[2] - local[2]) < 1E-9 && \"error in transformation\");\n";
  outstream << "#endif\n";
  outstream << "\n";
}

void NavigationSpecializer::DumpFoo(std::ostream &outstream) const
{

  // get the stream --> very cumbersome
  VPlacedVolume *pvol = fLogicalVolume->Place();
  std::stringstream shapetypestream;
  pvol->PrintType(shapetypestream);
  delete pvol;
  std::string shapetype = shapetypestream.str();

  // analyse daughters
  // this is very crude --> we trust the shapetypes given back from the shape
  // the only optimization here is that we get rid of the virtual function and that we unroll the loop

  // but we could also analyse it deeply ourselves here and specialize ourselves
  // future possible optimizations include the following:
  // a) vectorization of similar daughter shapes
  // b) constant extraction and propagation ( detect of shapes use same rotation and pull them out ... )
  // c) put actual numeric values of shapes which may simplify the algorithms ... ( extreme shape specialization )
  auto daughters  = fLogicalVolume->GetDaughtersp();
  auto ndaughters = daughters->size();
  if (ndaughters > 0) {
    outstream << "auto daughters = lvol->GetDaughtersp();\n";

    // first pass to separate list of polymorphic shapes into separated loops over the same kind
    std::string currenttype;
    std::list<std::pair<std::string, size_t>> looplist; // list for loop splitting
    shapetypestream.str("");
    (*daughters)[0]->PrintType(shapetypestream);
    currenttype  = shapetypestream.str();
    size_t count = 1;
    for (auto i = decltype(ndaughters){1}; i < ndaughters; ++i) {
      auto daughter = (*daughters)[i];
      shapetypestream.str("");
      daughter->PrintType(shapetypestream);
      std::string thistype = shapetypestream.str();
      if (currenttype.compare(thistype) == 0)
        count++;
      else {
        looplist.push_back(std::pair<std::string, size_t>(currenttype, count));
        count       = 1;
        currenttype = thistype; /*somehow next type*/
      }
    }
    looplist.push_back(std::pair<std::string, size_t>(currenttype, count));

    // print some info about the loop list
    for (auto &pair : looplist) {
      std::cerr << "## " << pair.first << " " << pair.second << "\n";
    }

    if (fUnrollLoops) {
      for (auto i = decltype(ndaughters){0}; i < ndaughters; ++i) {
        shapetypestream.str(""); // this clears the stream
        auto daughter = (*daughters)[i];
        daughter->PrintType(shapetypestream);
        shapetype = shapetypestream.str();
        outstream << "{\n";
        outstream << "auto daughter = (*daughters)[" << i << "];\n";
        outstream << "auto ddistance = ((" << shapetype << "*) daughter)->" << shapetype
                  << "::DistanceToIn(localpoint, localdir, step);\n";

        outstream << "bool valid = (ddistance < step && !IsInf(ddistance));\n";
        outstream << "hitcandidate = valid ? daughter : hitcandidate;\n";
        outstream << "step = valid ? ddistance : step;}\n";
      }
    } else {
      outstream << "// reach currently unimplemented point in code generation\n";
      //      // emit smaller loops
      //      int offset = 0;
      //      for (auto &pair : looplist) {
      //        if (pair.second > 1) {
      //          outstream << "for(int i = " << offset << ";i<" << offset + pair.second << ";++i){";
      //          outstream << "s = Min(s, ((" << pair.first << "*) (*daughters)[ i ])->" << pair.first
      //                    << "::DistanceToIn(localpoint, localdir, step));\n";
      //          outstream << "} // end loop\n";
      //        } else {
      //          outstream << "s = Min(s, ((" << pair.first << "*) (*daughters)[" << offset << "])->" << pair.first
      //                    << "::DistanceToIn(localpoint, localdir, step));\n";
      //        }
      //        offset += pair.second;
      //      }
    }
  }
  outstream << "return false;\n";
}

void NavigationSpecializer::DumpRelocateMethod(std::ostream &outstream) const
{
  // function header
  outstream << "VECGEOM_FORCE_INLINE\n";
  outstream << "virtual void Relocate(Vector3D<Precision> const &pointafterboundary, NavigationState const "
               "&__restrict__ in_state,"
               "NavigationState &__restrict__ out_state) const override {\n";
  outstream << "// this means that we are leaving the mother\n";
  outstream << "// alternatively we could use nextvolumeindex like before\n";

  if (fLogicalVolume->GetDaughtersp()->size() > 0) {
    outstream << "if( out_state.Top() == in_state.Top() ){\n";
  }
  outstream << "// this was probably calculated before \n";
  outstream << "auto pathindex = out_state.GetCacheValue();\n";
  outstream << "if(pathindex < 0){ pathindex = PathToIndex(&in_state);\n }";

  for (size_t i = 0; i < fTransitionOrder.size(); ++i) {
    size_t transitionid    = fTransitionOrder[i];
    auto &transitionstring = fTransitionStrings[transitionid];
    // tokenize the string with getline

    std::istringstream ss(transitionstring);
    std::string token;
    bool horizstate = false;
    bool downstate  = false;

    // count number of ops first of all
    unsigned int downcount  = 0;
    unsigned int upcount    = 0;
    unsigned int horizcount = 0;
    while (std::getline(ss, token, '/')) {
      if (token.compare("down") == 0) downcount++;
      if (token.compare("up") == 0) upcount++;
      if (token.compare("horiz") == 0) horizcount++;
    }
    if ((downcount > 0) && !(upcount || horizcount)) {
      // filter out pure down states
      std::cerr << "not doing" << transitionstring << "\n";
      continue;
    } else {
      std::cerr << transitionstring << " has " << downcount << ";" << upcount << ";" << horizcount << "\n";
    }

    outstream << "{\n";
    outstream << "// considering transition " << transitionstring << "\n";
    outstream << "short index = deltamatrixmapping[pathindex][" << transitionid << "];\n";
    outstream << "if(index!=-1){\n";
    outstream << fDeltaTransformationCode.str();
    outstream << "VPlacedVolume const * pvol = &GeoManager::gCompactPlacedVolBuffer[" << fTargetVolIds[transitionid]
              << "];\n";
    outstream << "bool intarget = "
              << "((" << fTransitionTargetTypes[transitionid].second << "const *) pvol)->"
              << fTransitionTargetTypes[transitionid].second << "::UnplacedContains(localpoint);\n";
    outstream << "if(intarget){\n";
    // now parse the transition string an calculate the outstate from this

    std::stringstream ss2(transitionstring);

    while (std::getline(ss2, token, '/')) {
      // try parse a number first of all
      if (horizstate || downstate) {
        int number = std::stoi(token);

        if (horizstate) {
          outstream << "auto oldvalue = out_state.ValueAt( out_state.GetCurrentLevel()-1 );\n";
          outstream << "out_state.Pop();\n";
          outstream << "out_state.PushIndexType( oldvalue  + " << number << " );\n";
          horizstate = false;
        }
        if (downstate) {
          outstream << "out_state.PushIndexType(" << number << ");\n";
          downstate = false;
        }
      }

      // analyse token
      if (token.compare("up") == 0) {
        outstream << "out_state.Pop();\n";
      }
      if (token.compare("horiz") == 0) {

        // expect number as next token
        horizstate = true;
      }
      if (token.compare("down") == 0) {
        downstate = true;
      }
    }
    outstream << "return;}\n";
    outstream << "}\n";
    outstream << "}\n";
  }
  if (fLogicalVolume->GetDaughtersp()->size() > 0) {
    outstream << "}\n";
    outstream << "else {\n";

    // for the moment we find out whether all possible daughters are entered directly
    // in which case we don't do any further action
    // TODO: put into place a more generic but optimized treatment like for leaving the mother
    bool alldowntransitionsaretrivial = true;
    for (size_t i = 0; i < fTransitionOrder.size(); ++i) {
      size_t transitionid    = fTransitionOrder[i];
      auto &transitionstring = fTransitionStrings[transitionid];
      // tokenize the string with getline

      std::istringstream ss(transitionstring);
      std::string token;

      // count number of ops first of all
      unsigned int downcount  = 0;
      unsigned int upcount    = 0;
      unsigned int horizcount = 0;
      while (std::getline(ss, token, '/')) {
        if (token.compare("down") == 0) downcount++;
        if (token.compare("up") == 0) upcount++;
        if (token.compare("horiz") == 0) horizcount++;
      }
      if (downcount && !(upcount || horizcount)) {
        // this is a down transition
        if (downcount > 1) alldowntransitionsaretrivial = false;
      }
    }
    if (alldowntransitionsaretrivial) {
      outstream << "// we don't do anything; the outstate should already be correct in any case\n";
      outstream << "return;\n";
      outstream << "}\n";
    } else {
      outstream << " // fallback to generic treatment first of all\n";
      outstream << "// continue directly further down ( next volume should have been stored in out_state already )\n";
      outstream << "VPlacedVolume const *nextvol = out_state.Top();\n";
      outstream << "out_state.Pop();\n";
      outstream << "GlobalLocator::LocateGlobalPoint(nextvol, "
                   "nextvol->GetTransformation()->Transform(pointafterboundary), out_state, false);\n";

      outstream << "VECGEOM_ASSERT(in_state.Distance(out_state) != 0 && \" error relocating when entering \")\n";
      outstream << "}\n";
    }
  }
  outstream << "}\n";
}

void NavigationSpecializer::DumpLocalHitDetectionFunction(std::ostream &outstream) const
{
  // if user specified base navigator we are using this here

  bool needdaughtertreatment = fLogicalVolume->GetDaughtersp()->size() > 0;

  /*
  // function 1 opening
  outstream << "virtual Precision ComputeStepAndHittingBoundaryForLocalPoint(Vector3D<Precision> const & localpoint, "
               "Vector3D<Precision> const & localdir, Precision pstep, NavigationState const & in_state, "
               "NavigationState & out_state) const override {\n";

  if (fUseBaseNavigator) {
    outstream << "return fBaseNavigator." << fBaseNavigator
              << "<>::ComputeStepAndHittingBoundaryForLocalPoint(localpoint, localdir, pstep, "
                 "in_state, out_state);\n";
  } else {
    // put specialized local hit detection ( probably only useful for small number of daughters )
    outstream << " VECGEOM_ASSERT(false && \"reached unimplemented point\");\n";
    outstream << " return -1;\n";
  }

  // function closing
  outstream << "}\n";
  */

  // function 2
  // the bool return type indicates if out_state was already modified; this may happen in assemblies;
  // in this case we don't need to copy the in_state to the out state later on
  outstream << "virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const & "
               "localpoint, Vector3D<Precision> const & localdir,                                           "
               "NavigationState const * in_state, NavigationState * out_state, Precision &step, VPlacedVolume const *& "
               "hitcandidate) const override {\n";
  if (!needdaughtertreatment) {
    // empty function; do nothing
    outstream << "return false;\n";
  } else {
    outstream << "// we need daughter treatment\n";
    if (fUseBaseNavigator) {
      outstream << "// we fall back daughter treatment of existing navigator \n";
      outstream << "return fBaseNavigator." << fBaseNavigator
                << "<>::CheckDaughterIntersections(lvol, localpoint, localdir, in_state, out_state, "
                   "step, hitcandidate);\n";
    } else {
      outstream << "// we emit specialized daughter treatment\n";
      // put specialized local hit detection ( probably only useful for small number of daughters )
      DumpFoo(outstream);
    }
  }

  outstream << "}\n";
}

} // namespace vecgeom
