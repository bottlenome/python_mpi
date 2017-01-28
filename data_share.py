#!/usr/bin/env python3

from mpi4py import MPI
import threading
import time
import numpy

TAG_SCORE = 0
TAG_BEST_SCORE = 1
TAG_DATA = 2
TAG_STATUS = 3

STATUS_ALIVE = 0
STATUS_DEAD = 1

SCORE_MAX = 1000000000

INDEX_SIZE = 5
LOOP = 10

def child_process(comm, rank, size):
    def task(index, rank):
        #print("do task %d : "%rank, index)
        return numpy.random.random()

    best_score = [SCORE_MAX] * INDEX_SIZE
    best_data = [numpy.random.random()] * INDEX_SIZE

    for index in range(INDEX_SIZE):
        for i in range(LOOP):
            score = task(index, rank)
            time.sleep(1)
            if score < best_score[index]:
                best_score[index] = score

            # 1. send scores
            comm.isend([index , score], dest = 0, tag = TAG_SCORE)
            #print("send score (%d) %d -> 0 :"%(index, rank), score)

            # 2. get best datas
            while comm.iprobe(source = 0, tag = TAG_DATA):
                req = comm.irecv(source = 0, tag = TAG_DATA)
                index_tmp, score, data = req.wait()
                #print("get  data  (%d) %d -> 0 :"%(index, rank), score, data)
                if score < best_score[index_tmp]:
                    best_data[index_tmp] = data

            # 3. get best scores
            best_score_tmp = SCORE_MAX
            while comm.iprobe(source = 0, tag = TAG_BEST_SCORE):
                req = comm.irecv(source = 0, tag = TAG_BEST_SCORE)
                index, best_score_tmp = req.wait()
                #print("get  score (%d) %d -> 0 :"%(index, rank), score)
            if best_score[index] < best_score_tmp:
                # 4. send best datas
                comm.isend([index, best_score[index], best_data[index]], dest = 0, tag = TAG_DATA)
                #print("send data  (%d) %d -> 0: "%(index, rank), best_score[index], best_data[index])


    # 5. send all task is ends
    req = comm.isend(STATUS_DEAD, dest = 0, tag = TAG_STATUS)
    req.wait()
    print("end process : ", rank)

def parent_process(comm, size):
    best_score = [100000000] * INDEX_SIZE
    best_data = [numpy.random.random()] * INDEX_SIZE
    nodes = [i for i in range(1, size)]
    isUpdated = 0
    while True:
        # change node scores and update
        for i in nodes:
            # 1. get scores
            if comm.iprobe(source = i, tag = TAG_SCORE):
                req = comm.irecv(source = i, tag = TAG_SCORE)
                index, score = req.wait()
                #print("#get (%d) %d -> 0:"%(index, i), score)
                if best_score[index] < score:
                    # 2. send best datas
                    req = comm.isend([index, best_score[index], best_data[index]],
                                     dest = i, tag = TAG_DATA)
                    #print("#send (%d) 0 -> %d :"%(index, i), best_score[index], best_data[index])
        # 3. send best scores
        if isUpdated != 0:
            for i in nodes:
                if isUpdated != i:
                    comm.isend([index, best_score[index]], dest = i, tag = TAG_BEST_SCORE)
                    #print("#broadcast scores (%d) 0 -> %d"%(index, i), best_score[index])
            isUpdated = 0

        # 4. get scores and datas
        for i in nodes:
            while comm.iprobe(source = i, tag = TAG_DATA):
                req = comm.irecv(source = i, tag = TAG_DATA)
                index, score, data = req.wait()
                if score < best_score[index]:
                    best_score[index] = score
                    best_data[index] = data
                    isUpdated = i
                    print("#updated (%d)"%index, score, data)
        # 5. check node status
        remove_nodes = []
        for i in nodes:
            while comm.iprobe(source = i, tag = TAG_STATUS):
                req = comm.irecv(source = i, tag = TAG_STATUS)
                status = req.wait()
                remove_nodes.append(i)
        for i in remove_nodes:
            nodes.remove(i)
        if len(nodes) == 0:
            break

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    if rank == 0:
        parent_process(comm, size)
    else:
        child_process(comm, rank, size)

