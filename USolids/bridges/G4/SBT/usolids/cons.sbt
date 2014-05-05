#
# GEANT4 SBT Script to test G4Cons
# DCW 19/3/99 First try
#
# Increment the number below when errors become a bit more rare
#
#
# --- cons.a1.log
# Here is a tube with no inner radius and no phi segmentation
#

/solid/G4Cons 1 2 1 3 1 0 360
/performance/errorFileName log/cons-test10-p10k/sbt.log
/control/execute usolids/performance.sbt

exit


### this test case does not work, error!
/solid/G4Cons 0 0 0 1 1 0 180
/performance/errorFileName log/cons-test9-p10k/sbt.log
/control/execute usolids/performance.sbt
exit

/performance/maxPoints 10000
/performance/repeat 100

# rMin1, rMax1, rMin2, rMax2, dz, sPhi, dPhi




/solid/G4Cons 0 1 0 1 1 20 181
/performance/errorFileName log/cons-test5-p10k/sbt.log
/control/execute usolids/performance.sbt

/solid/G4Cons 0 1 0 1 1 0 360
/performance/errorFileName log/cons-test-p10k/sbt.log
/control/execute usolids/performance.sbt

# does not work, only 5 parameters there
#/solid/G4Cons 0.5 1 0.5 1.2 0 360
#/performance/errorFileName log/cons-test2-p10k/sbt.log
#/control/execute usolids/performance.sbt

/solid/G4Cons 0 1 0.5 1 1 0 360
/performance/errorFileName log/cons-test3-p10k/sbt.log
/control/execute usolids/performance.sbt

/solid/G4Cons 0 1 0 1 1 0 90
/performance/errorFileName log/cons-test4-p10k/sbt.log
/control/execute usolids/performance.sbt

/solid/G4Cons 0 1 0 1 1 20 181
/performance/errorFileName log/cons-test5-p10k/sbt.log
/control/execute usolids/performance.sbt

/solid/G4Cons 0.5 1.0 0.7 1.2 1 20 350
/performance/errorFileName log/cons-test6-p10k/sbt.log
/control/execute usolids/performance.sbt

/solid/G4Cons 0.0 0.2 0.8 1.0 0.0001 10 90
/performance/errorFileName log/cons-test7-p10k/sbt.log
/control/execute usolids/performance.sbt

/solid/G4Cons 5 10 20 25 40 0 120
/performance/errorFileName log/cons-test8-p10k/sbt.log
/control/execute usolids/performance.sbt


exit

#
/voxel/errorFileName log/consv.a1.log
/voxel/run
#
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/cons.a2.log
/test/run
#
# --- cons.b1.log
# Now add an inner radius
#
/test/gridSizes 0 0 0 m
/solid/G4Cons 0.5 1 0.5 1.2 0 360
/test/errorFileName  log/cons.b1.log
/test/run
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/cons.b2.log
/test/run
#
# --- cons.c1.log
# Do something particularly cruel: close the cone off on one end
#
/test/gridSizes 0 0 0 m
/solid/G4Cons 0 1 0.5 1 1 0 360
/test/errorFileName  log/cons.c1.log
/test/run
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/cons.c2.log
/test/run
#
# --- cons.d1.log
# Now add a phi segment to the simple case
#
/test/gridSizes 0 0 0 m
/solid/G4Cons 0 1 0 1 1 0 90
/test/errorFileName  log/cons.d1.log
/test/run
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/cons.d2.log
/test/run
#
# --- cons.e1.log
# Do the same but with a slightly different phi slice
#
/test/gridSizes 0 0 0 m
/solid/G4Cons 0 1 0 1 1 20 181
/test/errorFileName  log/cons.e1.log
/test/run
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/cons.e2.log
/test/run
#
# --- cons.f1.log
# Now add a phi slice to a cone with inner radii
#
/test/gridSizes 0 0 0 m
/solid/G4Cons 0.5 1.0 0.7 1.2 1 20 350
/test/errorFileName  log/cons.f1.log
/test/run
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/cons.f2.log
/test/run
#
# --- cons.g1.log
# A real nightmare
#
/test/gridSizes 0 0 0 m
/solid/G4Cons 0.0 0.2 0.8 1.0 0.0001 10 90
/test/errorFileName  log/cons.g1.log
/test/widths 1 1 0.0002 m
/test/run
/test/gridSizes 0.02 0.02 0.00001 m
/test/errorFileName  log/cons.g2.log
/test/run
#
exit

