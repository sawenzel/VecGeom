
# performance question geant4 vs. usolid(slow)

#
# GEANT4 SBT Script to test G4Box
# DCW 19/3/99 
#
/test/maxPoints 10000

#
# --- box.a1.log
# Start with a nice and easy case
#
/solid/G4Box 1 1 1

/test/errorFileName log/box.a1.log
/test/run
#
/voxel/errorFileName log/boxv.a1.log
/voxel/run
#
# --- box.a2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/box.a2.log
/test/run
#
# --- box.b1.log
# Try an odd case
#
/solid/G4Box 0.0001 1 2
/test/errorFileName  log/box.b1.log
/test/run
#
/voxel/errorFileName log/boxv.b1.log
/voxel/run
#
exit
