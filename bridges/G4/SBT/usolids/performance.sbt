 
/performance/differenceTolerance 0.02

/performance/maxInsidePercent 25
/performance/maxOutsidePercent 50
/performance/method Inside
/performance/run

/performance/maxInsidePercent 100
# maxOutsidePercent must be 0, otherwise errors are produced in Geant4
/performance/maxOutsidePercent 0
/performance/method DistanceToOut
/performance/run 

/performance/maxInsidePercent 0
/performance/maxOutsidePercent 100
/performance/method DistanceToIn
/performance/run

/performance/maxInsidePercent 0
/performance/maxOutsidePercent 0
/performance/method Normal
/performance/run

# does not make sense to test safety from outside for inside points
/performance/maxInsidePercent 0
/performance/maxOutsidePercent 75
/performance/method SafetyFromOutside
/performance/run

/performance/maxInsidePercent 75
# does not make sense to test safety from inside for outside points
/performance/maxOutsidePercent 0
/performance/method SafetyFromInside
/performance/run
