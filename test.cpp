#include "algri.h"
#include "seed.h"
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <pthread.h>


#include "matrixcal.h"
#define DEVICEMEMORY		(1024*1024*1024)			 //1G

AlgriMatList* matList_int8;
int g_deviceNum;
cublasHandle_t g_handle[6];
int8_t* g_device_matList[6];

static uint8_t g_msg[32] = {
        0xd0, 0xda, 0xd7, 0x3f, 0xb2, 0xda, 0xbf, 0x33,
        0x53, 0xfd, 0xa1, 0x55, 0x71, 0xb4, 0xe5, 0xf6,
        0xac, 0x62, 0xff, 0x18, 0x7b, 0x35, 0x4f, 0xad,
        0xd4, 0x84, 0x0d, 0x9f, 0xf2, 0xf1, 0xaf, 0xdf,
};

static uint8_t g_seed[32] = {
        0x07, 0x37, 0x52, 0x07, 0x81, 0x34, 0x5b, 0x11,
        0xb7, 0xbd, 0x0f, 0x84, 0x3c, 0x1b, 0xdd, 0x9a,
        0xea, 0x81, 0xb6, 0xda, 0x94, 0xfd, 0x14, 0x1c,
        0xc9, 0xf2, 0xdf, 0x53, 0xac, 0x67, 0x44, 0xd2,    
};

// result
static uint8_t g_results[32] = {
        0xe3, 0x5d, 0xa5, 0x47, 0x95, 0xd8, 0x2f, 0x85,
        0x49, 0xc0, 0xe5, 0x80, 0xcb, 0xf2, 0xe3, 0x75,
        0x7a, 0xb5, 0xef, 0x8f, 0xed, 0x1b, 0xdb, 0xe4,
        0x39, 0x41, 0x6c, 0x7e, 0x6f, 0x8d, 0xf2, 0x27,  
};

typedef struct st_calculateThreadArg{
	uint8_t *msg;
	uint32_t len;
	uint32_t threadID;
	uint8_t result[32];
}stCalculateThreadArg, *pstCalculateThreadArg;

Words32 extSeedCreate(uint8_t *seed);

static void init_seed(Words32 &seed, uint32_t _seed[32])
{
    for (int i = 0; i < 16; i++)
        seed.lo.w[i] = _seed[i];
    for (int i = 0; i < 16; i++)
        seed.hi.w[i] = _seed[16 + i];
}


//add by wyh, create extseed from seed
Words32 extSeedCreate(uint8_t *seed)
{
	uint32_t exted[32];
	extend(exted, seed);
	Words32 extSeed;
	init_seed(extSeed, exted);

	return extSeed;
}

void* calculate_Thread(void *arg)
{
	pstCalculateThreadArg calculateThreadArg = (pstCalculateThreadArg)arg;

	//iter(calculateThreadArg->msg, calculateThreadArg->len, calculateThreadArg->result, calculateThreadArg->threadID);

	for (int j = 0; j < 32; j++) {
		if (calculateThreadArg->result[j] != g_results[j]) {
			printf("Results does not match j : %d \n", j);
			break;
		}
	}
}

void memoryManageInit()
{
	cudaGetDeviceCount(&g_deviceNum);
	printf("device num: %d\n", g_deviceNum);

	memory_pool->inital(g_deviceNum, DEVICEMEMORY);
}


int main(void)
{
	memoryManageInit();

	double start_t, end_t;
	//start_t = GetMillsec();

    uint8_t results[32] = { 0 };
	Words32 extSeed = extSeedCreate(g_seed);
	
	matList_int8 = new AlgriMatList;
	matList_int8->init(extSeed);

	for (int i = 0; i < g_deviceNum; i++)
	{
		cudaSetDevice(i);
		cublasCreate(&g_handle[i]);

		g_device_matList[i] = (int8_t *)memory_pool->CMalloc(i, sizeof(int8_t) * 256 * 256 * 256);
		cudaError_t cudaStatus = cudaMemcpy(g_device_matList[i], matList_int8->matVec, sizeof(int8_t) * 256 * 256 * 256, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	}
	
	Mat256x256i8 *res = new Mat256x256i8[4];
	Mat256x256i8 *mat = new Mat256x256i8;
	sha3_ctx *ctx = (sha3_ctx*)calloc(1, sizeof(*ctx)); 

	/////////////////////////////////////////////////////////////////////////////////////
	start_t = GetMillsec();

    for (int i = 0;i<1 ; i++) {	

		//iter(g_msg, 32, results, i, res, mat, ctx);
		//iter();
		end_t = GetMillsec();
		printf("iter out time: %lf\n", end_t - start_t);

        int j = 0;
        for (; j < 32; j++) {
            // printf("0x%02x, ",results[i][j]);
            if (results[j] != g_results[j]) {
				printf("Results does not match, i: %d , j : %d \n", i, j);
                break;
            }
        }
    }
	//delete matList_int8;

	end_t = GetMillsec();
	std::cout << "all time : "
		<< end_t - start_t << "ms"
		<< std::endl;

	delete mat;
	delete[] res;
	free(ctx);

	//printf("\n\n Multi process in.\n");
	///////////////////////////////////////////////////////////////////////////////////
	//start_t = GetMillsec();
	////while (1)
	////{
	//	pthread_t *calculateThread = (pthread_t *)malloc(sizeof(pthread_t) * g_deviceNum);
	//	int threadNum = g_deviceNum;
	//	pstCalculateThreadArg calculateThreadArg = new stCalculateThreadArg[threadNum]();
	//	for (int i = 0; i < threadNum; i++)
	//	{
	//		calculateThreadArg[i].threadID = i;
	//		calculateThreadArg[i].msg = g_msg;
	//		calculateThreadArg[i].len = 32;
	//		memset(calculateThreadArg[i].result, 0, sizeof(calculateThreadArg[i].result));

	//		if (pthread_create(&calculateThread[i], NULL, calculate_Thread, (void *)&calculateThreadArg[i]) != 0)
	//		{
	//			printf("ERROR: calculateThread create failed.\n");
	//			return -1;
	//		}
	//	}

	//	for (int i = 0; i < threadNum; i++)
	//	{
	//		if (pthread_join(calculateThread[i], NULL) != 0)
	//		{
	//			printf("ERROR: calculateThread join failed.\n");
	//			return -1;
	//		}
	//	}

	//	if (calculateThreadArg)
	//		delete[] calculateThreadArg;

	//	end_t = GetMillsec();
	//	std::cout << "all time : "
	//		<< end_t - start_t << "ms"
	//		<< std::endl;

	//	//usleep(10000);
	////}



	delete matList_int8;

	for (int i = 0; i < g_deviceNum; i++)
	{
		cudaSetDevice(i);
		cublasDestroy(g_handle[i]);
		memory_pool->CFree(i, g_device_matList[i]);
	}
	
    return 0;
}
