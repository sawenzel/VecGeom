#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTrap.hh"

#include "base/Vector3D.h"
#include "volumes/Trapezoid.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

//using Trap_t = UTrap;
using Trap_t = vecgeom::SimpleTrapezoid;

int main(int argc, char *argv[]) {

  VUSolid* trap = new Trap_t("test_trap",5.,0.,0.,5.,5.,5.,0.,5.,5.,5.,0.);
  // VUSolid* trap = new Trap_t("test_trap",5.,5.,5.);
  ShapeTester tester;

  if(argc>1) {
    if(strcmp(argv[1],"vis")==0) {
      #ifdef VECGEOM_ROOT
        TApplication theApp("App",0,0);
        tester.Run(trap);
        theApp.Run();
      #endif
    }
  }
  else  {
    tester.Run(trap);
  }

  return 0;
}
