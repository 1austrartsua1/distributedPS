
maxIterSync=1000
maxIterSyncD=200
maxIterAsync=80000

scaleTime=.005

#srun python syncCentralPS.py --maxIter $maxIterSync
#srun python syncCentralPS.py --maxIter $maxIterSyncD --addDelay --scaleTime $scaleTime 

#srun python parallelGD.py --maxIter $maxIterSync
srun python parallelGD.py --maxIter $maxIterSyncD --addDelay --scaleTime $scaleTime 

#srun python centralizedPS.py --maxIter $maxIterAsync 
#srun python centralizedPS.py --maxIter $maxIterAsync --addDelay --scaleTime $scaleTime 
python readOut.py