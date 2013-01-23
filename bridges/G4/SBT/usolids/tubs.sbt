#
# GEANT4 SBT Script to test G4Tubs
# DCW 19/3/99 First try
#
# Increment the number below to **really** waste CPU time
#
#
# --- tubs.a1.log
# Here is a tube with no inner radius and no phi segmentation
#

/performance/maxPoints 10000
/performance/repeat 100

/solid/G4Tubs 0 1 1 0 360
/performance/errorFileName log/tubs-test-p10k/sbt.log
/control/execute usolids/values.sbt

/solid/G4Tubs 0.5 1 1 0 360
/performance/errorFileName log/tubs-test2-p10k/sbt.log
/control/execute usolids/values.sbt

/solid/G4Tubs 0 1 1 0 90
/performance/errorFileName log/tubs-test3-p10k/sbt.log
/control/execute usolids/values.sbt


/solid/G4Tubs 0.8 1 1 0 90
/performance/errorFileName log/tubs-test4-p10k/sbt.log
/control/execute usolids/values.sbt

/solid/G4Tubs 0.00999 0.01001 1 10 260
/performance/errorFileName log/tubs-test5-p10k/sbt.log
/control/execute usolids/values.sbt

/solid/G4Tubs 0.00999 0.01001 1 350 10
/performance/errorFileName log/tubs-test6-p10k/sbt.log
/control/execute usolids/values.sbt

exit

#
# --- tubs.b1.log
# Add an inner radius
#
/test/gridSizes 0 0 0 m
/solid/G4Tubs 0.5 1 1 0 360
/test/errorFileName  log/tubs.b1.log
/test/run
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/tubs.b2.log
/test/run
#
# --- tubs.c1.log
# Add a phi segment to a tube with no inner radius
#
/test/gridSizes 0 0 0 m
/solid/G4Tubs 0 1 1 0 90
/test/errorFileName  log/tubs.c1.log
/test/run
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/tubs.c2.log
/test/run
#
# --- tubs.d1.log
# Add a phi segment to a tube with an inner radius
#
/test/gridSizes 0 0 0 m
/solid/G4Tubs 0.8 1 1 0 90
/test/errorFileName  log/tubs.d1.log
/test/run
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/tubs.d2.log
/test/run
#
# --- tubs.e1.log
# Make a nice and long and thin tube
#
/test/gridSizes 0 0 0 m
/solid/G4Tubs 0.00999 0.01001 1 10 260
/test/widths 0.02 0.02 1 m
/test/gridSizes 0.005 0.005 0.1 m
/test/errorFileName  log/tubs.e1.log
/test/run
#
# --- tubs.f1.log
# Make a nice and long and thin tube with a different phi slice
#
/test/gridSizes 0 0 0 m
/solid/G4Tubs 0.00999 0.01001 1 350 10
/test/errorFileName  log/tubs.f1.log
/test/run
#
exit
