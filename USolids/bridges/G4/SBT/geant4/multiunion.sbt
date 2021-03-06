#
# GEANT4 SBT Script to test G4TRD
# DCW 17/9/99 
#
#
# --- trd.a1.log
# Start with a nice and easy case
#

/solid/MultiUnion 1 1 1

/performance/errorFileName log/multiunion-5-t10k/multiunion.a1.log
/performance/maxPoints 1000
/performance/repeat 1
/control/execute usolids/test.sbt

exit

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