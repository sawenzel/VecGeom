#
# GEANT4 SBT Script to test Ubox
# DCW 19/3/99 
#
/test/maxPoints 10000
#
# --- ubox.a1.log
# Start with a nice and easy case
#
/solid/UBox 1 1 1
/test/errorFileName log/ubox.a1.log
/test/run
#
/voxel/errorFileName log/uboxv.a1.log
/voxel/run
#
# --- ubox.a2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/ubox.a2.log
/test/run
#
# --- ubox.b1.log
# Try an odd case
#
/solid/UBox 0.0001 1 2
/test/errorFileName  log/ubox.b1.log
/test/run
#
/voxel/errorFileName log/uboxv.b1.log
/voxel/run
#
exit
