#
# GEANT4 SBT Script to test G4Sphere
# I.Hrivnacova, IPN Orsay 23/01/2008 
#
#
# --- sphere.a1.log
# No inner radius,  no phi segment, no theta segment
#
  
/performance/maxPoints 10000
/performance/repeat 1000

#
# --- sphere.e1.log
# No inner radius,  no phi segment, with theta segment
#

/solid/G4Sphere 0 1 0 360 0 45
/performance/errorFileName log/sphere-test5-10k/spherep.a1.log
/control/execute usolids/values.sbt

#exit

/solid/G4Sphere 0 1 0 360 0 180
/performance/errorFileName log/sphere-test1-10k/spherep.a1.log
/control/execute usolids/values.sbt

#
# --- sphere.b1.log
# With inner radius,  no phi segment, no theta segment
#
/solid/G4Sphere 0.5 1 0 360 0 180
/performance/errorFileName log/sphere-test2-10k/spherep.a1.log
/control/execute usolids/values.sbt

#
# --- sphere.c1.log
# No inner radius,  with phi segment, no theta segment
#
/solid/G4Sphere 0 1 0 90 0 180
/performance/errorFileName log/sphere-test3-10k/spherep.a1.log
/control/execute usolids/values.sbt

#
# --- sphere.d1.log
# With inner radius,  with phi segment, no theta segment
#
/solid/G4Sphere 0.5 1 0 90 0 180
/performance/errorFileName log/sphere-test4-10k/spherep.a1.log
/control/execute usolids/values.sbt


#
# --- sphere.f1.log
# With inner radius,  no phi segment, with theta segment
#
/solid/G4Sphere 0.5 1 0 360 0 90
/performance/errorFileName log/sphere-test6-10k/spherep.a1.log
/control/execute usolids/values.sbt

#
# --- sphere.g1.log
# No inner radius,  with phi segment, with theta segment
#
/solid/G4Sphere 0 1 0 90 0 90
/performance/errorFileName log/sphere-test7-10k/spherep.a1.log
/control/execute usolids/values.sbt

#
# --- sphere.h1.log
# With inner radius,  with phi segment, with theta segment
#
/solid/G4Sphere 0.5 1 0 90 0 90
/performance/errorFileName log/sphere-test8-10k/spherep.a1.log
/control/execute usolids/values.sbt

#
# --- sphere.i1.log
# With inner radius,  with phi segment, with theta segment
# Very thin in radius
#

/solid/G4Sphere 0.00999 0.01001 10 260 30 70
/performance/errorFileName log/sphere-test9-10k/spherep.a1.log
/control/execute usolids/values.sbt

exit
