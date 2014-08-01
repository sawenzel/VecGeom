#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UBox.hh"

//#include "base/Vector3D.h"
//#include "volumes/Box.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif

int main() {

  UBox* box=new UBox("test_box",5.,5.,5.);
  
  ShapeTester tester;

#ifdef VECGEOM_ROOT
    TApplication theApp("App",0,0);
#endif

  tester.Run(box);

#ifdef VECGEOM_ROOT
  theApp.Run();
#endif

  return 0;
}



