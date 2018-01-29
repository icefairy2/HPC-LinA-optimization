## Compile and run

To run the implementation, one must:  
- clone the repository on *CooLMUC3*  
- load the hdf5 module: `module load hdf5/mpi/1.8.15`  
- compile: `export ORDER=2;bash -x compilescript`  
- create a script to run the code with mpi (e.g. it should contain an
mpirun command as `mpirun -np 2 build/lina -s 0 -x 10 -y 10 -a 10 -b 5 -o output/test`)  
.
