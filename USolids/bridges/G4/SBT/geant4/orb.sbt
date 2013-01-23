#
# GEANT4 SBT Script to test G4Orb
# I.Hrivnacova, IPN Orsay 23/01/2008 
#
#
# --- orb.a1.log
#
/solid/G4Orb 1

/test/maxPoints 10000
/test/errorFileName log/orb.a1.log
/test/run

/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/orb.a2.log
/test/run
#
exit
