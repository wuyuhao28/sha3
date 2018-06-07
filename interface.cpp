#include <iostream>
#include <cstdio>
#include <map>
#include "matrixcal.h"
#include "seed.h"
#include "interface.h"
#include <atomic>
using namespace std;
// AlgriMatList* matList_int8;
int g_deviceNum=6;
cublasHandle_t g_handle[6];
int8_t* g_device_matList[6];
uint8_t g_seed[32]={0};
std::atomic<int> g_counter (0);
double start=0;
// extern uint8_t g_seed[32];
// cTaskQueue g_tskQueue[DEVICENUM];
// cTaskQueue g_retQueue;
int threadID[DEVICENUM];
// template class threadsafe_queue<pTaskST> g_tskQueue[DEVICENUM];
SafeQueue g_tskQueue[DEVICENUM];
SafeQueue g_tskQueue1;
SafeQueue g_tskQueue2;
SafeQueue g_tskQueue3;
SafeQueue g_tskQueue4;
SafeQueue g_tskQueue5;
SafeQueue g_tskQueue0;

SafeQueue g_retQueue;
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
	g_deviceNum=devicecount();
    memory_pool->inital(g_deviceNum, DEVICEMEMORY);

    // uint8_t results[32] = { 0 };
	// Words32 extSeed = extSeedCreate(g_seed);
	// matList_int8 = new AlgriMatList;
	// matList_int8->init(extSeed);
    for (int i = 0; i < g_deviceNum; i++)
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
    pthread_t *calculateThread = (pthread_t *)malloc(sizeof(pthread_t) * g_deviceNum);
    int threadNum = g_deviceNum;

	for (int i = 0; i < g_deviceNum; i++)
	{
		threadID[i] = i;
		if (pthread_create(&calculateThread[i], NULL, calculate_Thread, (void *)&(threadID[i])) != 0)
		{
			printf("ERROR: calculateThread create failed.\n");
			return -1;
		}
	}
	printf("in c:get_hash_init finished\n");
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
int get_hash(uint8_t msgs[32], uint8_t seeds[32],uint32_t *deviceids,uint8_t target[32],char* jobid,uint8_t* len,uint64_t* nonce){
	// printf("in c:%s,%d\n",jobid,*len);
  int deviceid=int(*deviceids);
//   cudaSetDevice(deviceid);
  
//   if(deviceid<7){
//       return call_hash(msg,seed,ret,seedCache[deviceid],deviceid);
//   }
  	tag_stTaskST* taskNode = (tag_stTaskST*)malloc(sizeof(TaskST));
    // taskNode->msg = g_msg;
    // memcpy(taskNode->msg, &msgs, 8);
    // taskNode->msg = msgs;
	memcpy(taskNode->msg, msgs,32);
    taskNode->len = 32;
    //taskNode->threadID = i % DEVICENUM;
    memset(taskNode->result, 0, sizeof(taskNode->result));
    memcpy(taskNode->seed, seeds,32);
	memcpy(taskNode->target, target,32);
	taskNode->jobid=(char*)malloc(*len*sizeof(char));
	memcpy(taskNode->jobid, jobid,*len);
	taskNode->lenOfJobid=*len;
	taskNode->nonce=*nonce;
	// printf("in c gethash nonce:%llu\n",taskNode->nonce);
    taskNode->pNext = NULL;
    // g_tskQueue[deviceid % DEVICENUM].InQueue(taskNode);
	// g_tskQueue[deviceid].Push(taskNode);
	if(deviceid==0){
		g_tskQueue0.Push(taskNode);
	}else if(deviceid==1){
		g_tskQueue1.Push(taskNode);
	}else if(deviceid==2){
		g_tskQueue2.Push(taskNode);
	}else if(deviceid==3){
		g_tskQueue3.Push(taskNode);
	}else if(deviceid==4){
		g_tskQueue4.Push(taskNode);
	}else if(deviceid==5){
		g_tskQueue5.Push(taskNode);
	}else{
		g_tskQueue0.Push(taskNode);
	}

	//test.Push(taskNode);
  return 0;
}
void get_rets(){
	if(g_counter.load(std::memory_order_relaxed)<=0){
		printf("should be only show once\n");
		start=GetMillsec();
	}
	pTaskST tmpTask;
	while(tmpTask=g_retQueue.Pop()){
		//  = g_retQueue.OutQueue();
		g_counter++;
		uint8_t *msg = tmpTask->msg; 
		uint8_t* result = tmpTask->result;
		// printf("%d msg:",0);
		// for(int i=0;i<3;i++){
		// 	printf("0x%02x ",msg[i]);
		// }
		// printf("\n%d ret:",0);
		// printf("%d ret:",0);
		// for(int i=0;i<1;i++){
			printf("0x%02x %s\n",result[0],tmpTask->jobid);
		// }
		// printf("\n");
		if(g_counter.load(std::memory_order_relaxed)>=60000){
			printf("should be only show once2\n");
			double alltime=(double)(GetMillsec() - start)/1000;
			printf("device all,Time : %f t/s\n",(double)(6*10000)/alltime);
		}
	}
	
}
int get_ret(uint8_t *msg,uint8_t* result,uint8_t* target,char* jobid,uint8_t* len,uint64_t* nonce){
	
	pTaskST tmpTask;
	// if(!g_retQueue.Empty())
	tmpTask=g_retQueue.Pop();
	if(tmpTask!=NULL)
	{		
		memcpy(msg, tmpTask->msg, 32);
		memcpy(result, tmpTask->result, 32);
		// memcpy(len, tmpTask->lenOfJobid, 1);
		*len=tmpTask->lenOfJobid;
		memcpy(jobid, tmpTask->jobid, *len);
		memcpy(target,tmpTask->target,32);
		*nonce=tmpTask->nonce;
		// printf("in c ret nonce:%llu %llu\n",*nonce,tmpTask->nonce);
		// msg = tmpTask->msg; 
		// result = tmpTask->result;
		// jobid=tmpTask->jobid;
		// len=&(tmpTask->lenOfJobid);
		// printf("%d\n",msg[0]);
		// printf("in c:0x%02x,0x%02x,0x%02x,0x%02x,%d,%lld\n",msg[0],result[0],jobid[0],target[0],*len,*nonce);
		// printf("in c rethash:");
		// for(int i=0;i<32;i++){
		// 	printf("%02x",result[i]);
		// }
		// printf("\n");
		// delete tmpTask;
		free(tmpTask->jobid);
		free(tmpTask);
		return 1;
	}
	else
	{
		return 0;
	}
}
// void clearOldJob(char* newjobid){
// 	g_tskQueue1.Clear();
// 	g_tskQueue1=new SafeQueue<tag_stTaskST*>();
// 	SafeQueue<tag_stTaskST*> g_tskQueue1;
// SafeQueue<tag_stTaskST*> g_tskQueue2;
// SafeQueue<tag_stTaskST*> g_tskQueue3;
// SafeQueue<tag_stTaskST*> g_tskQueue4;
// SafeQueue<tag_stTaskST*> g_tskQueue5;
// SafeQueue<tag_stTaskST*> g_tskQueue0;

// SafeQueue<tag_stTaskST*> g_retQueue;
// }
void clearOldJob(char* newjobid){
	for(int i=0;i<g_deviceNum;i++){
		// g_tskQueue[i].Clear();
		if(i==0){
		g_tskQueue0.Clear();
		}else if(i==1){
			g_tskQueue1.Clear();
		}else if(i==2){
			g_tskQueue2.Clear();
		}else if(i==3){
			g_tskQueue3.Clear();
		}else if(i==4){
			g_tskQueue4.Clear();
		}else if(i==5){
			g_tskQueue5.Clear();
		}
	}
	g_retQueue.Clear();
}
// void clearOldJob(char* newjobid){
// 	for(int i=0;i<g_deviceNum;i++){
// 		// if(g_tskQueue[i].Empty()){
// 		// 	continue;
// 		// }
// 		while(1){
// 			tag_stTaskST* taskNode=NULL;
// 			if(i==0){
// 				taskNode =g_tskQueue0.Front();
// 			}else if(i==1){
// 				taskNode =g_tskQueue1.Front();
// 			}else if(i==2){
// 				taskNode =g_tskQueue2.Front();
// 			}else if(i==3){
// 				taskNode =g_tskQueue3.Front();
// 			}else if(i==4){
// 				taskNode =g_tskQueue4.Front();
// 			}else if(i==5){
// 				taskNode =g_tskQueue5.Front();
// 			}
			
// 			if(taskNode==NULL){
// 				break;
// 			}
// 			if(memcmp(taskNode->jobid,newjobid,taskNode->lenOfJobid) != 0)
// 			{
// 				// g_tskQueue[i].Pop();
// 				if(i==0){
// 				g_tskQueue0.Pop();
// 				}else if(i==1){
// 					g_tskQueue1.Pop();
// 				}else if(i==2){
// 					g_tskQueue2.Pop();
// 				}else if(i==3){
// 					g_tskQueue3.Pop();
// 				}else if(i==4){
// 					g_tskQueue4.Pop();
// 				}else if(i==5){
// 					g_tskQueue5.Pop();
// 				}
// 			}
// 			else
// 			{
// 				break;
// 			}	
// 		}
// 	}
// 	while(1){
// 		tag_stTaskST* retNode=NULL;
// 		retNode =g_retQueue.Front();
// 		if(retNode==NULL){
// 			break;
// 		}
// 		if(memcmp(retNode->jobid,newjobid,retNode->lenOfJobid) != 0)
// 		{
// 			g_retQueue.Pop();
// 		}
// 		else
// 		{
// 			break;
// 		}	
// 	}
// }
// void ret_callback(pTaskST tmpTask){
// 	if(g_counter.load(std::memory_order_relaxed)<=0){
// 		start=GetMillsec();
// 	}
// 	g_counter++;
// 	// counter.store(x,std::memory_order_relaxed);
// 	if(g_counter.load(std::memory_order_relaxed)>=60000){
// 		double alltime=(double)(GetMillsec() - start)/CLOCKS_PER_SEC/1000;
// 		printf("device all,Time : %f t/s\n",(double)(6*10000)/alltime);
// 	}
// 	printf("ret_callback 0x%02x \n",tmpTask->result[0]);
// }
int devicecount(){
    int GPU_N=0;
    cudaError_t err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
        return 0;
    }
    return GPU_N;
}