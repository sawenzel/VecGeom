#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UCons.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Cone.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

//typedef UCons Cone_t;
typedef vecgeom::SimpleCone Cone_t;

int main(int argc,char *argv[]) {

  VUSolid* cone=new Cone_t("test_cone",5.,6.,5.5,7.,2,0,vecgeom::kTwoPi);
  ShapeTester tester;

  if(argc>1)
  {
    if(strcmp(argv[1],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     tester.Run(cone);
     theApp.Run();
     #endif
    }
  }
  else
  {
    tester.Run(cone);
   }
  return 0;
}



