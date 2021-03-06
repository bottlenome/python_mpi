#!/usr/bin/env python3
"""
Parallel Hello World
"""

from mpi4py import MPI
import sys

mpi_info = MPI.Info.Create()
mpi_info.Set("add-hostfile", "hosts")

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

sys.stdout.write(
    "Hello, World! I am process %d of %d on %s.\n"
    % (rank, size, name))
