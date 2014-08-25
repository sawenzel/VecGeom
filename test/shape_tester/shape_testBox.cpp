#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UBox.hh"

#include "base/Vector3D.h"
#include "volumes/Box.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

//typedef UBox Box_t;
typedef vecgeom::SimpleBox Box_t;

int main(  int argc,char *argv[]) {

  VUSolid* box=new Box_t("test_box",5.,5.,5.);
  ShapeTester tester;

  if(argc>1)
  {
    if(strcmp(argv[1],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     tester.Run(box);
     theApp.Run();
     #endif
    }
  }
  else
  {
    tester.Run(box);

   }

  return 0;
}



