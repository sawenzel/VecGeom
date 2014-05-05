
/performance/differenceTolerance 0.02

/performance/maxInsidePercent 50
/performance/maxOutsidePercent 50
/performance/method Inside
/performance/run

/performance/maxInsidePercent 10 # john: 10
/performance/maxOutsidePercent 60 # john: 60
/performance/method DistanceToIn
/performance/run

/performance/maxInsidePercent 60 # john 60
# maxOutsidePercent must be 0, otherwise errors are produced in Geant4
/performance/maxOutsidePercent 0 # john 10
/performance/method DistanceToOut
/performance/run 


/performance/maxInsidePercent 0
/performance/maxOutsidePercent 0
/performance/method Normal
/performance/run

/performance/maxInsidePercent 80 # john: 80
# does not make sense to test safety from inside for outside points
/performance/maxOutsidePercent 0
/performance/method SafetyFromInside
/performance/run

# does not make sense to test safety from outside for inside points
/performance/maxInsidePercent 0
/performance/maxOutsidePercent 80 # john: 80
/performance/method SafetyFromOutside
/performance/run

