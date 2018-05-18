#ifndef MATRIXCAL_H
#define MATRIXCAL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "algri.h"

#include "memorypool.h"
#include "memorypoolmanager.h"

#include <cublas_v2.h> //cuda×Ô´ø¿âº¯Êý
//#include <cublas.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>

cudaError_t matrixMul(Mat256x256i8& sourceMatrix, const Mat256x256i8* tmpMatrix, const AlgriMatList* matList_int8, uint8_t *sequence);
//__global__ void mulKernel(Mat256x256i8& sourceMatrix, Mat256x256i8* tmpMatrix, Mat256x256i8* seqMatrix);

void iter(const uint8_t *msg, uint32_t len, uint8_t result[32], uint32_t threadID);

double GetMillsec();

#endif
