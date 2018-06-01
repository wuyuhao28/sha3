#include <iostream>
#include <cstdio>
#include <map>
#include "matrixcal.h"
#include "seed.h"
#include "interface.h"
using namespace std;
AlgriMatList* matList_int8;
int g_deviceNum;
cublasHandle_t g_handle[6];
int8_t* g_device_matList[6];
uint8_t g_seed[32]={0};
cTaskQueue g_tskQueue[DEVICENUM];
// template class threadsafe_queue<pTaskST> g_tskQueue[DEVICENUM];
//SafeQueue<tag_stTaskST*> g_tskQueue[DEVICENUM];
//SafeQueue<tag_stTaskST*> test;
// threadsafe_queue<tag_stTaskST*> test;
// typedef struct st_calculateThreadArg{
// 	uint8_t *msg;
// 	uint32_t len;
// 	uint32_t threadID;
// 	uint8_t result[32];
// 	uint8_t seed[32];
// 	//Mat256x256i8 *res;
// 	//Mat256x256i8 *mat;
// 	//sha3_ctx *ctx;
// }stCalculateThreadArg, *pstCalculateThreadArg;
int get_hash_init(){
    memory_pool->inital(DEVICENUM, DEVICEMEMORY);

    // uint8_t results[32] = { 0 };
	// Words32 extSeed = extSeedCreate(g_seed);
	matList_int8 = new AlgriMatList;
	// matList_int8->init(extSeed);
    for (int i = 0; i < DEVICENUM; i++)
	{
		cudaSetDevice(i);
		cublasCreate(&g_handle[i]);

		//g_device_matList[i] = (int8_t *)memory_pool->CMalloc(i, sizeof(int8_t) * 256 * 256 * 256);
		cudaMalloc((void **)&g_device_matList[i], sizeof(int8_t) * 256 * 256 * 256);
		// cudaError_t cudaStatus = cudaMemcpy(g_device_matList[i], matList_int8->matVec, sizeof(int8_t) * 256 * 256 * 256, cudaMemcpyHostToDevice);
		// if (cudaStatus != cudaSuccess)
		// 	printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	}
    // for (int i = 0; i < TASK_NUM; i++)
	// {
	// 	pTaskST taskNode = (pTaskST)malloc(sizeof(TaskST));
	// 	taskNode->msg = g_msg;
	// 	taskNode->len = 32;
	// 	//taskNode->threadID = i % DEVICENUM;
	// 	memset(taskNode->result, 0, sizeof(taskNode->result));
	// 	memcpy(taskNode->seed, g_seed, sizeof(g_seed));
	// 	taskNode->pNext = NULL;
	// 	g_tskQueue[i % DEVICENUM].InQueue(taskNode);
	// }  
    pthread_t *calculateThread = (pthread_t *)malloc(sizeof(pthread_t) * DEVICENUM);
    int threadNum = DEVICENUM;
	int threadID[DEVICENUM];
	for (int i = 0; i < DEVICENUM; i++)
	{
		g_tskQueue[i].pHeader = NULL;
		threadID[i] = i;
		if (pthread_create(&calculateThread[i], NULL, calculate_Thread, (void *)&(threadID[i])) != 0)
		{
			printf("ERROR: calculateThread create failed.\n");
			return -1;
		}
	}
}

// int call_hash(uint8_t msg[32], uint8_t seed[32],uint8_t ret[32],map<vector<uint8_t>,MatrixMatListGpu*> &seedCache,int deviceid)
// {
//   MatrixMatList *matList_int8=NULL;
//   MatrixMatListGpu *matListGpu_int8=NULL;
//   cudaSetDevice(deviceid);
//     vector<uint8_t> seedVec(seed, seed + 32);

//     if(seedCache.find(seedVec) != seedCache.end()) {
//         // printf("\t---%s---\n", "Seed already exists in the cache.");
//         matListGpu_int8 = seedCache[seedVec];
//     } else {
//         uint32_t exted[32];
//         extend(exted, seed); // extends seed to exted
//         Words32 extSeed;
//         init_seed(extSeed, exted);

//         matList_int8=new MatrixMatList;
//         initMatVec(matList_int8->matVec, extSeed);

//         matListGpu_int8=new MatrixMatListGpu;
//         initMatVecGpu(matListGpu_int8, matList_int8);

//         seedCache.insert(pair<vector<uint8_t>, MatrixMatListGpu*>(seedVec, matListGpu_int8));
//         delete matList_int8;
    
//     }
//     matList_int8=NULL;
//     int itercallret=itercall(msg,matListGpu_int8,ret,deviceid);
//     matListGpu_int8=NULL;

//     return itercallret;
// }
int get_hash(uint8_t msgs[32], uint8_t seeds[32],uint8_t ret[32],uint32_t *deviceids){
  int deviceid=int(*deviceids);
  //cudaSetDevice(deviceid);
  
//   if(deviceid<7){
//       return call_hash(msg,seed,ret,seedCache[deviceid],deviceid);
//   }
	printf("deviceid: %d\n", deviceid);
  	tag_stTaskST* taskNode = (tag_stTaskST*)malloc(sizeof(TaskST));
    // taskNode->msg = g_msg;
    // memcpy(taskNode->msg, &msgs, 8);
    taskNode->msg = msgs;
    taskNode->len = 32;
    //taskNode->threadID = i % DEVICENUM;
    memset(taskNode->result, 0, sizeof(taskNode->result));
    memcpy(taskNode->seed, seeds,sizeof(g_seed));
    taskNode->pNext = NULL;
	printf("before InQueue test\n");
    g_tskQueue[deviceid % DEVICENUM].InQueue(taskNode);
	//g_tskQueue[deviceid % DEVICENUM].Push(taskNode);
	//test.Push(taskNode);
  return 0;
}
