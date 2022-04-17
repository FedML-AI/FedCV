#!/bin/sh

#SBATCH -o /apps/mpi/myjob.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
mpirun python /apps/mpi/helloworld.py



