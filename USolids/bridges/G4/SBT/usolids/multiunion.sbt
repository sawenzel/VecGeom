#
# GEANT4 SBT Script to test G4TRD
# DCW 17/9/99 
#
#
# --- trd.a1.log
# Start with a nice and easy case
#

# performance scalability test

/performance/maxPoints 10000
/performance/repeat 10

/solid/G4MultiUnion 10
/performance/errorFileName log/multiunion-test-p10k/sbt.log

/control/execute usolids/test.sbt

exit
 
/performance/maxInsidePercent 80
# maxOutsidePercent must be 0, otherwise errors are produced in Geant4
/performance/maxOutsidePercent 0
/performance/method DistanceToOut
/performance/run 

exit

/control/execute usolids/values.sbt

exit

/solid/MultiUnion 2
/performance/maxPoints 10000
/performance/repeat 1
/performance/errorFileName log/multiunion-2-t10k/sbt.log
/control/execute usolids/test.sbt


/test/errorFileName log/trd.a1.log
/test/run

#
# --- trd.a2.log
# Up the ante ans generate points on a grid
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/trd.a2.log
/test/run
#
# --- trd.b1.log
# Adjust just x
#
#/test/gridSizes 0 0 0 m
/solid/G4trd 0.5 1.5 1 1 1
/test/errorFileName  log/trd.b1.log
/test/run
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/trd.b2.log
/test/run
#
# --- trd.c1.log
# Adjust x and y
#
/test/gridSizes 0 0 0 m
/solid/G4trd 0.5 1.5 0.25 1 1
/test/errorFileName  log/trd.c1.log
/test/run
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/trd.c2.log
/test/run
#
#
# --- trd.d1.log
# extreme case
#
/test/widths 1 0.00002 1 m
/test/gridSizes 0 0 0 m
/solid/G4trd 0.000001 1 0.00001 0.00002 1
/test/errorFileName  log/trd.d1.log
/test/run
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/trd.d2.log
/test/run
#
exit 