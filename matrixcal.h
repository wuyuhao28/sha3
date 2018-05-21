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

extern cublasHandle_t g_handle[6];
extern int8_t* g_device_matList[6];

#define LOOP_COUNT		2
#define SEQUENCE_COUNT	32

#define BLOCK_SIZE		256
#define THREAD_SIZE		256

cudaError_t matrixMul(Mat256x256i8& sourceMatrix, const Mat256x256i8* tmpMatrix, const AlgriMatList* matList_int8, uint8_t *sequence, int8_t* threadID);

//__global__ void mulKernel(Mat256x256i8& sourceMatrix, Mat256x256i8* tmpMatrix, Mat256x256i8* seqMatrix);

//void iter(const uint8_t *msg, uint32_t len, uint8_t result[32], uint32_t threadID, Mat256x256i8 *res, Mat256x256i8 *mat, sha3_ctx *ctx);
void iter();

double GetMillsec();

#endif