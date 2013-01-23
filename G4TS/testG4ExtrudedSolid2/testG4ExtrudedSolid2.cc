
#include "G4TwoVector.hh"
#include "G4ExtrudedSolid.hh"
#include "G4Timer.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4ios.hh"

G4VSolid* solid1 = 0;
G4VSolid* solid2 = 0;
G4VSolid* solid3 = 0;


//_____________________________________________________________________________
void createSolids()
{

/*
VGM info:      Extruded  "tfHEXA_4"
Polygon,  6 vertices:
  (-68.7616, 0)  (-32.9973, 50.1581)  (32.9973, 50.1581)  (68.7616, 0)  (35.7678, -54.3684)  (-35.7678, -54.3684) mm
Planes:
  z = -110 mm    x0 = 0 mm    y0 = 0 mm    scale= 0.54176
  z = 110 mm    x0 = 0 mm    y0 = 0 mm    scale= 1
*/

  std::vector<G4TwoVector> polygon2;
  polygon2.push_back(G4TwoVector(-68.7616, 0));  
  polygon2.push_back(G4TwoVector(-32.9973, 50.1581));  
  polygon2.push_back(G4TwoVector(32.9973, 50.1581));  
  polygon2.push_back(G4TwoVector(68.7616, 0));  
  polygon2.push_back(G4TwoVector(35.7678, -54.3684));  
  polygon2.push_back(G4TwoVector(-35.7678, -54.3684));
  solid2 = new G4ExtrudedSolid(
               "tfHEXA_4", polygon2, 110.*mm, 
               G4TwoVector(), 0.54176, G4TwoVector(), 1.0);

  return;

/*
VGM info:      Extruded  "mlHEXA_4"
Polygon,  6 vertices:
  (-68.7816, 0)  (-32.9973, 50.1781)  (32.9973, 50.1781)  (68.7816, 0)  (35.7678, -54.3884)  (-35.7678, -54.3884) mm
Planes:
  z = -110 mm    x0 = 0 mm    y0 = 0 mm    scale= 0.54205
  z = 110 mm    x0 = 0 mm    y0 = 0 mm    scale= 1
*/

  std::vector<G4TwoVector> polygon;
  polygon.push_back(G4TwoVector(-68.7816, 0));  
  polygon.push_back(G4TwoVector(-32.9973, 50.1781));  
  polygon.push_back(G4TwoVector(32.9973, 50.1781));  
  polygon.push_back(G4TwoVector(68.7816, 0));  
  polygon.push_back(G4TwoVector(35.7678, -54.3884)); 
  polygon.push_back(G4TwoVector(-35.7678, -54.3884));
  
  solid1 = new G4ExtrudedSolid(
               "mlHEXA_4", polygon, 110.*mm, 
                G4TwoVector(), 0.54205, G4TwoVector(), 1.0);

  
/*
VGM info:      Extruded  "HEXA_4"
Polygon,  6 vertices:
  (-68.6416, 0)  (-32.9973, 50.0381)  (32.9973, 50.0381)  (68.6416, 0)  (35.7678, -54.2484)  (-35.7678, -54.2484) mm
Planes:
  z = -110 mm    x0 = 0 mm    y0 = 0 mm    scale= 0.54096
  z = 110 mm    x0 = 0 mm    y0 = 0 mm    scale= 1
*/
  std::vector<G4TwoVector> polygon3;
  polygon3.push_back(G4TwoVector(-68.6416, 0));  
  polygon3.push_back(G4TwoVector(-32.9973, 50.0381));  
  polygon3.push_back(G4TwoVector(32.9973, 50.0381));  
  polygon3.push_back(G4TwoVector(68.6416, 0));  
  polygon3.push_back(G4TwoVector(35.7678, -54.2484));  
  polygon3.push_back(G4TwoVector(-35.7678, -54.2484));
  solid3 = new G4ExtrudedSolid(
               "HEXA_4", polygon3, 110.*mm, 
                G4TwoVector(), 0.54096, G4TwoVector(), 1.0);
}                             

//_____________________________________________________________________________
int main()
{
  G4Timer timer;
  timer.Start();

  createSolids();
  timer.Stop();
  G4cout << timer << G4endl;
  
//  delete solid1;
  delete solid2;
  //delete solid3;
    
  return 0;
}
