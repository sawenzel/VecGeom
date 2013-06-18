#
# GEANT4 SBT Script to test G4Tet
# I.Hrivnacova, IPN Orsay 28/01/2008 

#
/test/maxPoints 1000
#
# --- extru.a1.log
# Regular
#
#/solid/G4ExtrudedSolid 4 (-0.3,-0.3,0.3,0.3) (-0.3,0.3,0.3,-0.3) 2 (-0.3,0.3) (0.0,0.0) (0.0,0.0) (1.0,1.0)
# --- extrudedSolid.d1.log
# Another extruded solid, where polygon decomposition was failing
# in Geant4 9.1
#
#solid a3 from Documentation
/solid/G4ExtrudedSolid 8 (-30,-30,30,30,15,15,-15,-15) (-30,30,30,-30,-30,15,15,-30) 4 (-60,-15,10,60) (0,0,0,0) (30,-30,0,30) (0.8,1.0,0.6,1.2)

#solid a2 reported problem in old Geant4 version
#/solid/G4ExtrudedSolid 8 (-0.2,-0.2,0.1,0.1,0.2,0.2,-0.1,-0.1) (0.1,0.25,0.25,-0.1,-0.1,-0.25,-0.25,0.1) 2 (-0.2,0.2) (0.0,0.0) (0.0,0.0) (1.0,1.0)
#solid a4 solid from ATLAS
/solid/G4ExtrudedSolid 8 (-320.35,-858.219082449407,-1183.5,-417,417,1183.5,858.219082449407,320.35) (0,-222.792298222938,562.505304394976,880,880,562.505304394976,-222.792298222938,0) 2 (-275,275) (0,0) (0,0) (1,1)

/performance/errorFileName log/extr-test4-v10k2/sbt.log
/performance/maxPoints 10000
/performance/repeat 10

#/performance/errorFileName log/extr-test4-v10k6/sbt.log
#/performance/maxPoints 1000000
#/performance/repeat 1


#/control/execute usolids/performance.sbt
/control/execute usolids/values.sbt

/test/gridSizes 0.1 0.1 0.1 m
/test/errorFileName  log/extr.a4.log
/test/run

exit

/test/gridSizes 0 0 0 m
/test/errorFileName  log/tet.a1.log
/test/run
/voxel/errorFileName log/tetv.a1.log
/voxel/run
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/tet.a2.log
/test/run
#
# --- tet.b1.log
# Assymetric
#
/solid/G4Tet (0.0,0.0,1.0) (-1.0,-1.0,-1.0) (+1.0,-1.0,-1.0) (0.0,1.0,-1.0)
/test/gridSizes 0 0 0 m
/test/errorFileName  log/tet.b1.log
/test/run
/voxel/errorFileName log/tetv.b1.log
/voxel/run
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/tet.b2.log
/test/run
#
# --- tet.c1.log
# Assymetric, more extreme
#
/solid/G4Tet (0.0,0.0,1.0) (-0.2,-1.0,-1.0) (0.2,-1.0,-1.0) (0.0,1.0,-1.0)
/test/gridSizes 0 0 0 m
/test/errorFileName  log/tet.c1.log
/test/run
/voxel/errorFileName log/tetv.c1.log
/voxel/run
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/tet.c2.log
/test/run
#
# --- tet.d1.log
# Assymetric, very extreme
#
/solid/G4Tet (0.0,0.0,1.0) (-0.001,-1.0,-1.0) (+0.001,-1.0,-1.0) (0.0,1.0,-1.0)
/test/gridSizes 0 0 0 m
/test/errorFileName  log/tet.d1.log
/test/run
/voxel/errorFileName log/tetv.d1.log
/voxel/run
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/tet.d2.log
/test/run
#
exit
