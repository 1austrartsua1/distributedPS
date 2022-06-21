# distributedPS
Implementations for solving least-squares. These were run on the Cori cluster using an interactive session. Get an interactive session with 
```
salloc -C gpu -N 1 --ntasks 20 -t 180 --cpus-per-task 1 -A m3898
```
Note that there is no GPU. This is cpu-only code. 20 is the number of CPUs and tasks.

There are three files you may run: centralizedPS.py, syncCentralPS.py and parallelGD.py. In addition, you need to randomly generate an instance of data using makeRandomLS.py. To run, use 
```
srun python syncCentralPS.py 
```
or 
```
srun python centralizedPS.py
```
or 
```
srun python parallelGD.py 
```
An entire experiment can be run using ./runExperiment.sh. 

There are command line arguments to choose the random delay amounts. 


## List of Files 

* centralizedPS.py: Async Master-worker implementation of projective splitting.
* syncCentralPS.py: Synchronous master-worker projective splitting
* parallelGD.py: synchronous accelerated GD using master-worker topology.
* singleProcPS.py: single threaded implementation to compare results with. 