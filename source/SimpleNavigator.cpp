/// \file SimpleNavigator.cpp
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 16.04.2014

#include "navigation/SimpleNavigator.h"

#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "navigation/NavigationState.h"
#include "volumes/PlacedVolume.h"

#ifdef VECGEOM_ROOT
#include "TGeoManager.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#endif

#ifdef VECGEOM_GEANT4
#include "management/G4GeoManager.h"
#include "G4Navigator.hh"
#include "G4VPhysicalVolume.hh"
#endif

#undef NDEBUG

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_ROOT

void SimpleNavigator::InspectEnvironmentForPointAndDirection
   (   Vector3D<Precision> const & globalpoint,
      Vector3D<Precision> const & globaldir,
      NavigationState const & state,
      std::ostream & outstream
   ) const
{
   Transformation3D m;
   state.TopMatrix(m);
   Vector3D<Precision> localpoint = m.Transform( globalpoint );
   Vector3D<Precision> localdir = m.TransformDirection( globaldir );

   // check that everything is consistent
   {
      NavigationState * tmpstate = NavigationState::MakeInstance( state.GetMaxLevel() );
      tmpstate->Clear();
      LocatePoint( GeoManager::Instance().GetWorld(),
              globalpoint, *tmpstate, true );
      bool locationcorrect = tmpstate->Top() == state.Top();
      if( !locationcorrect ){
          outstream << "WARNING: LOCATION NOT CORRECT\n";
          outstream << "presumed "; state.printVolumePath(outstream);
          outstream << "actual"; tmpstate->printVolumePath(outstream);
          //outstream << tmpstate->Top()->GetLabel() << " versus presumed " << state.Top()->GetLabel() << "\n";
          //tmpstate->Print();
      }
      NavigationState::ReleaseInstance( tmpstate );
   }

   // now check mother and daughters
   VPlacedVolume const * currentvolume = state.Top();
   outstream << "############################################ " << "\n";
   outstream << "Navigating in placed volume : " << RootGeoManager::Instance().GetName( currentvolume ) << "\n";

   int nexthitvolume = -1; // means mother
   int tmp = outstream.precision();
   outstream << std::setprecision(20);
   outstream << "localpoint " << localpoint << "\n";
   outstream << "localdir " << localdir << "\n";
   outstream << std::setprecision(tmp);
   outstream << "check containment in mother " << currentvolume->UnplacedContains( localpoint ) << "\n";
   double step = currentvolume->DistanceToOut( localpoint, localdir );

   outstream << "DistanceToOutMother : " << step << "\n";

   // iterate over all the daughters
   Vector<Daughter> const * daughters = currentvolume->GetLogicalVolume()->GetDaughtersp();

   outstream << "ITERATING OVER " << daughters->size() << " DAUGHTER VOLUMES " << "\n";
   for(int d = 0; d<daughters->size(); ++d)
   {
      VPlacedVolume const * daughter = daughters->operator [](d);
      //    previous distance becomes step estimate, distance to daughter returned in workspace
      Precision ddistance = daughter->DistanceToIn( localpoint, localdir, step );

      outstream << "DistanceToDaughter : " << RootGeoManager::Instance().GetName( daughter ) << " "
                << ddistance << " CONTAINED " << daughter->Contains(localpoint) << "\n";

      nexthitvolume = (ddistance < step) ? d : nexthitvolume;
      step      = (ddistance < step) ? ddistance  : step;
   }
   outstream << "DECIDED FOR NEXTVOLUME " << nexthitvolume << "\n";

   // same information from ROOT
   TGeoNode const * currentRootNode = RootGeoManager::Instance().tgeonode( currentvolume );
   double lp[3]={localpoint[0],localpoint[1],localpoint[2]};
   double ld[3]={localdir[0],localdir[1],localdir[2]};
   double rootstep =  currentRootNode->GetVolume()->GetShape()->DistFromInside( lp, ld, 3, 1E30, 0 );
   outstream << "---------------- CMP WITH ROOT ---------------------" << "\n";
   outstream << "DistanceToOutMother ROOT : " << rootstep << "\n";
   outstream << "Current shape type ROOT : " << currentRootNode->GetVolume()->GetShape()->ClassName() << "\n";
   outstream << "Containment in mother " << currentRootNode->GetVolume()->GetShape()->Contains(lp) << "\n";
   outstream << "ITERATING OVER " << currentRootNode->GetNdaughters() << " DAUGHTER VOLUMES " << "\n";
   for( int d=0; d<currentRootNode->GetNdaughters();++d )
   {
      TGeoNode const * daughter=currentRootNode->GetDaughter(d);
      TGeoMatrix const * m = daughter->GetMatrix();
      double llp[3], lld[3];
      m->MasterToLocal(lp, llp);
      m->MasterToLocalVect(ld, lld);
      Precision ddistance = daughter->GetVolume()->GetShape()->DistFromOutside(llp,lld,3,1E30,0);

      outstream << "DistanceToDaughter ROOT : " << daughter->GetName() << "(" << daughter->GetVolume()->GetShape()->ClassName() << ")" << " " << ddistance << " "
        << " CONTAINED " << daughter->GetVolume()->GetShape()->Contains(llp) << "\n";
   }

#ifdef VECGEOM_GEANT4
      // same information from G4
      // have to find volume first
      G4Navigator * g4nav = G4GeoManager::Instance().GetNavigator();

      G4ThreeVector g4gp(10*globalpoint.x(),10*globalpoint.y(),10*globalpoint.z());
      G4ThreeVector g4gd(globaldir.x(),globaldir.y(),globaldir.z());

      G4VPhysicalVolume *g4physvol = g4nav->LocateGlobalPointAndSetup( g4gp, &g4gd, false );
      G4LogicalVolume *  g4lvol = g4physvol->GetLogicalVolume();

      double safety;
      double g4step = g4nav->ComputeStep( g4gp, g4gd, kInfinity, safety );

      // we could add a check on the local points; g4nav can give the transformation
      outstream << "Vecgeom local" << localpoint << "\n";
      outstream << "G4localpoint " << g4nav->GetGlobalToLocalTransform().TransformPoint( g4gp ) << "\n";
      //g4nav->GetGlobalToLocalTransform().TransformAxis( g4gd );

      G4ThreeVector g4lp(10*localpoint[0],10*localpoint[1],10*localpoint[2]);
      G4ThreeVector g4ld(localdir[0],localdir[1],localdir[2]);

      outstream << "---------------- CMP WITH G4 ---------------------" << "\n";
      outstream << "Point is in Volume : " << g4lvol->GetSolid()->Inside(g4lp) << "\n";
      outstream << "Point is in Volume crosscheck : " << g4lvol->GetSolid()->Inside( g4nav->GetGlobalToLocalTransform().TransformPoint(g4gp)) << "\n";
      outstream << "DistanceToOutMother G4 : " << g4lvol->GetSolid()->DistanceToOut( g4lp, g4ld, false )/10 << "\n";
      outstream << "DistanceToOutMother G4 crosscheck : " << g4lvol->GetSolid()->DistanceToOut( g4nav->GetGlobalToLocalTransform().TransformPoint(g4gp),
              g4nav->GetGlobalToLocalTransform().TransformAxis(g4gd), false )/10 << "\n";
      outstream << "ITERATING OVER " << g4lvol->GetNoDaughters() << " DAUGHTER VOLUMES " << "\n";
      for( int d=0; d<g4lvol->GetNoDaughters();++d ) {
         G4VPhysicalVolume * daughterv = g4lvol->GetDaughter(d);

         G4AffineTransform trans( daughterv->GetRotation(), daughterv->GetTranslation() );
         G4ThreeVector llp = trans.TransformPoint( g4lp );
         G4ThreeVector lld = trans.TransformAxis( g4ld );
         Precision ddistance = daughterv->GetLogicalVolume()->GetSolid()->DistanceToIn(llp,llp);
         outstream << "DistanceToDaughter G4 : " << daughterv->GetName() << " " << ddistance/10. << "\n";
      }
      outstream << "STEP PROPOSED " << g4step/10. << "\n";
   #endif
}


void SimpleNavigator::InspectSafetyForPoint
   (   Vector3D<Precision> const & globalpoint,
      NavigationState const & state ) const
{
   Transformation3D m;
   state.TopMatrix(m);
   Vector3D<Precision> localpoint = m.Transform( globalpoint );

   // check that everything is consistent
   {
      NavigationState * tmpstate = NavigationState::MakeCopy( state );
      tmpstate->Clear();
      assert( LocatePoint( GeoManager::Instance().GetWorld(),
              globalpoint, *tmpstate, true ) == state.Top() );
   }

   std::cout << "############################################ " << "\n";
   // safety to mother
   VPlacedVolume const * currentvol = state.Top();
   double safety = currentvol->SafetyToOut( localpoint );
   std::cout << "Safety in placed volume : " << RootGeoManager::Instance().GetName( currentvol ) << "\n";
   std::cout << "Safety to Mother : " << safety << "\n";

   //assert( safety > 0 );

   // safety to daughters
   Vector<Daughter> const * daughters = currentvol->GetLogicalVolume()->GetDaughtersp();
   int numberdaughters = daughters->size();
   for(int d = 0; d<numberdaughters; ++d)
   {
       VPlacedVolume const * daughter = daughters->operator [](d);
       double tmp = daughter->SafetyToIn( localpoint );
       std::cout << "Safety to Daughter " << tmp << "\n";
       safety = Min(safety, tmp);
   }
   std::cout << "Would return" << safety << "\n";

   // same information from ROOT
   TGeoNode const * currentRootNode = RootGeoManager::Instance().tgeonode( currentvol );
   double lp[3]={localpoint[0],localpoint[1],localpoint[2]};
   double rootsafe =  currentRootNode->GetVolume()->GetShape()->Safety( lp, kTRUE );
   std::cout << "---------------- CMP WITH ROOT ---------------------" << "\n";
   std::cout << "SafetyToOutMother ROOT : " << rootsafe << "\n";
   std::cout << "ITERATING OVER " << currentRootNode->GetNdaughters() << " DAUGHTER VOLUMES " << "\n";
   for( int d=0; d<currentRootNode->GetNdaughters();++d )
   {
      TGeoNode const * daughter=currentRootNode->GetDaughter(d);
      TGeoMatrix const * m = daughter->GetMatrix();
      double llp[3];
      m->MasterToLocal(lp, llp);
      Precision ddistance = daughter->GetVolume()->GetShape()->Safety(llp, kFALSE);
      std::cout << "Safety ToDaughter ROOT : " << daughter->GetName() << " " << ddistance << "\n";
   }
}


void SimpleNavigator::CreateDebugDump(
        Vector3D<Precision> const & globalpoint,
        Vector3D<Precision> const & globaldir,
        NavigationState const & currentstate,
        double const pstep ) const {

    static int debugcounter=0;
    if( debugcounter < 20 ){
    // write out geometry first off all
    std::stringstream geomfilename;
    //geomfilename << "DebugGeom"<<debugcounter << ".root";
    geomfilename << "DebugGeom" << ".root";
    //gGeoManager->Export(geomfilename.str().c_str());

    std::stringstream header;
    // write header
    header << "#include \"base/Global.h\"\n";
    header << "#include \"volumes/PlacedVolume.h\"\n";
    header << "#include \"base/SOA3D.h\"\n";
    header << "#include \"base/Vector3D.h\"\n";
    header << "#include \"management/GeoManager.h\"\n";
    header << "#include \"navigation/NavigationState.h\"\n";
    header << "#include \"management/RootGeoManager.h\"\n";
    header << "#include \"TGeoNavigator.h\"\n";
    header << "#include \"TGeoNode.h\"\n";
    header << "#include \"TGeoMatrix.h\"\n";
    header << "#include \"navigation/SimpleNavigator.h\"\n";
    header << "#include \"TGeoManager.h\"\n";
    header << "#include \"TGeoBranchArray.h\"\n";
    header << "#include \"TPolyMarker3D.h\"\n";
    header << "#include \"TPolyLine3D.h\"\n";
    header << "#include \"TApplication.h\"\n";
    header << "using namespace vecgeom; \n";
    header << "typedef Vector3D<Precision> Vec3D;\n";
    // could be placed into macro or function
    header << "Vec3D const gpoint(\n";
    header << std::setprecision(64) << globalpoint.x() << ",\n";
    header << std::setprecision(64) << globalpoint.y() << ",\n";
    header << std::setprecision(64) << globalpoint.z() << ");\n";
    header << "Vec3D const gdir(\n";
    header << std::setprecision(64) << globaldir.x() << ",\n";
    header << std::setprecision(64) << globaldir.y() << ",\n";
    header << std::setprecision(64) << globaldir.z() << ");\n";

    header << "uint level = " << currentstate.GetLevel() << ";\n";
    std::list<uint> indices;
    currentstate.GetPathAsListOfIndices( indices );
    header << "std::list<uint> pathlist{";
    uint counter=0;
    for(auto x : indices )
    {
        counter++;
    if(counter==indices.size()) header << x;
    else header << x << ",";
    }
    header << "};\n";

    header << "std::string currentvolumename(\"" << currentstate.Top()->GetLabel() << "\");\n";

    header << "int main() {\n";
    header << " TGeoManager::Import(\"" << geomfilename.str() << "\");\n";
    header << " RootGeoManager::Instance().LoadRootGeometry();\n";
    header << " NavigationState * serializedstate = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );\n";
    header << " TGeoNavigator * nav = gGeoManager->GetCurrentNavigator();\n";
    header << " serializedstate->ResetPathFromListOfIndices( GeoManager::Instance().GetWorld(), pathlist );\n";
    header << " TGeoBranchArray * branch = serializedstate->ToTGeoBranchArray();\n";
    header << " branch->UpdateNavigator( nav );\n";

    std::stringstream visprogram;
    visprogram << " TGeoVolume * startvol = branch->GetCurrentNode()->GetVolume();\n";
    visprogram << "TApplication app(\"VecGeom Visualizer\", NULL, NULL);\n";
    //visprogram << "TGeoVolume * currentvol = gGeoManager->GetTopVolume();\n";
    visprogram << "TGeoVolume * currentvol = startvol;\n";
    visprogram << "currentvol->SetTransparency(30);\n";
    visprogram << "gGeoManager->SetVisLevel(3);\n";
    visprogram << "gGeoManager->SetTopVisible( true );\n";
    visprogram << "currentvol->Draw();\n";
    visprogram << "TPolyMarker3D marker;\n";
    visprogram << "marker.SetNextPoint( gpoint.x(), gpoint.y(), gpoint.z() );\n";
    visprogram << "marker.Draw();\n";
    visprogram << "TPolyLine3D line(2);\n";
    visprogram << "line.SetLineColor(kRed);\n";
    visprogram << "line.SetNextPoint( gpoint.x(), gpoint.y(), gpoint.z() );\n";
    visprogram << "line.SetNextPoint( gpoint.x() + nav->GetStep()*gdir.x(), gpoint.y() + nav->GetStep()*gdir.y(), gpoint.z() + nav->GetStep()*gdir.z() );\n";
    visprogram << "line.Draw();\n";
    visprogram << "app.Run();\n";

    // could put this into a vecgeom function which is just called
    std::stringstream rootprogram;
    //rootprogram << " // TGeoNode const * orignode = nav->FindNode( gpoint.x(), gpoint.y(), gpoint.z() );\n";
    //rootprogram << " // std::cout << \"##-- ROOT orignode --##\" << orignode->GetName() << std::endl;\n";
    rootprogram << " std::cout << \"##-- ROOT serialized top node --##\" << nav->GetCurrentNode()->GetName() << std::endl;\n";
    rootprogram << " std::cout << \"##-- ROOT path --## \" << nav->GetPath() << std::endl;\n";
    rootprogram << " nav->SetCurrentPoint( gpoint.x(), gpoint.y(), gpoint.z() );\n";
    rootprogram << " nav->SetCurrentDirection( gdir.x(), gdir.y(), gdir.z() );\n";
    rootprogram << " TGeoNode const * nextnode = nav->FindNextBoundaryAndStep( vecgeom::kInfinity );\n";
    rootprogram << " std::cout << \"##-- ROOT step --##\" << nav->GetStep() << std::endl;\n";
    rootprogram << " std::cout << \"##-- ROOT next node --##\" << nav->GetCurrentNode()->GetName() << std::endl;\n";

    std::stringstream vecgeomprogram;
    //vecgeomprogram << " branch->UpdateNavigator( nav );\n";
    //vecgeomprogram << " nav->SetCurrentPoint( gpoint.x(), gpoint.y(), gpoint.z() );\n";
    vecgeomprogram << " NavigationState * curnavstate = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );\n";
    vecgeomprogram << " NavigationState * newnavstate = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth() );\n";

    vecgeomprogram << " std::cout << \"##-- VecGeom serializednode --##\" << serializedstate->Top()->GetLabel() << std::endl;\n";
    vecgeomprogram << " std::cout << \"##-- VecGeom path --## \"; serializedstate->printVolumePath(std::cout); std::cout << std::endl;\n";
    vecgeomprogram << " SimpleNavigator vnav;\n";
    vecgeomprogram << " vnav.LocatePoint( GeoManager::Instance().GetWorld(), gpoint, *curnavstate, true );\n";
    vecgeomprogram << " double step;\n";
    vecgeomprogram << " std::cout << \"##-- VecGeom real node --##\" << curnavstate->Top()->GetLabel() << std::endl;\n";
    vecgeomprogram << " vnav.FindNextBoundaryAndStep( gpoint, gdir,*serializedstate, *newnavstate, vecgeom::kInfinity, step );\n";
    vecgeomprogram << " std::cout << \"##-- VecGeom step --##\" << step << std::endl;\n";
    vecgeomprogram << " std::cout << \"##-- VecGeom next node --##\" << newnavstate->Top()->GetLabel() << std::endl;\n";

    vecgeomprogram << " // call function showing some close inspection details\n";
    vecgeomprogram << " vnav.InspectEnvironmentForPointAndDirection( gpoint, gdir, *serializedstate );\n";


    std::stringstream tail;
    tail << " return 1;}\n";

    // write out global points and direction

    // write the program
    std::ofstream outFile, outFileVis;
    std::stringstream filename; filename << "DebugProgram" << debugcounter << ".cpp";

    outFile.open( filename.str() );
    outFile << header.str();
    outFile << rootprogram.str();
    outFile << vecgeomprogram.str();
    outFile << tail.str();
    outFile.close();

    std::stringstream filenamevis; filenamevis << "DebugProgram" << debugcounter << "_vis.cpp";
    outFileVis.open( filenamevis.str() );
    outFileVis << header.str();
    outFileVis << rootprogram.str();
    outFileVis << visprogram.str();
    outFileVis << tail.str();
    outFileVis.close();
    // write out state:
    // level
    // currentvolumename
    }
    debugcounter++;
}
#endif // VECGEOM_ROOT

} } // End global namespace
