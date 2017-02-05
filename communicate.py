#!/usr/bin/env python3
from mpi4py import MPI
import numpy
import time

#tags 
TAG_SCORE = 0
TAG_BEST_SCORE = 1
TAG_STATUS = 1
TAG_SEND_BEST = 2
TAG_UPDATE_BEST = 3

TAG_TO_ROOT = 0
TAG_TO_NODE = 1

#status
STATUS_NORMAL = 0
STATUS_SEND_BEST = 1
STATUS_UPDATE_BEST = 2

mpi_info = MPI.Info.Create()
mpi_info.Set("add-hostfile", "hosts")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

def task(best):
    score = numpy.random.random()
    best = numpy.random.randint(100)
    time.sleep(1)
    return score ,best

def send_score(comm, score):
    req = comm.isend(score, dest = 0, tag = TAG_SCORE)
    req.wait()

def get_status(comm):
    req = comm.irecv(source = 0, tag = TAG_STATUS)
    return req.wait()

score_best = 1000000000.0
best = numpy.random.randint(100)
i = 0
if rank == 0:
    data = [[score_best, best]] * size
    req_recv = [None] * size
    req_send = [None] * size
    while True:
        #recv score and best of node
        for i in range(1, size):
            req_recv[i] = comm.irecv(source = i, tag = TAG_TO_ROOT)
        for i in range(1, size):
            score, data = req_recv[i].wait()
            if score < score_best:
                print("update score at node %d:"%rank, score_best, "->", score)
                score_best = score
                best = data
        #send score and best to node
        for i in range(1, size):
            req_send[i] = comm.isend([score_best, best], dest = i, tag = TAG_TO_NODE)
            req_send[i].wait()
else:
    while True:
        i += 1
        print("rank %d, iter : %d, score:%f"%(rank, i, score_best))
        #do task
        score, data = task(best)
        if score < score_best:
            score_best = score
            best = data
        try:
            req_send.wait()
            score, data = req_recv.wait()
        except:
            pass

        #send score and best to node
        req_send = comm.isend([score_best, best], dest = 0, tag = TAG_TO_ROOT)

        #recv score and best of root
        req_recv = comm.irecv(source = 0, tag = TAG_TO_NODE)
        if score < score_best:
            print("update score at node %d:"%rank, score_best, "->", score)
            score_best = score
            best = data

