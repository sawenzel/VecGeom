#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UBox.hh"
#include "UOrb.hh"

#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Orb.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

//typedef UBox Box_t;
typedef vecgeom::SimpleOrb Orb_t;

int main(  int argc,char *argv[]) {

  VUSolid* orb=new Orb_t("test_orb",35);
   // VUSolid* orb=new UOrb("test_UOrb",3.);
  ShapeTester tester;

  if(argc>1)
  {
    if(strcmp(argv[1],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     tester.Run(orb);
     theApp.Run();
     #endif
    }
  }
  else
  {
    tester.Run(orb);

   }

  return 0;
}



