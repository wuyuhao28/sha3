#include "matrixcal.h"

#define TILE_WIDTH 16
__global__ void matrixMultiplyShared(int8_t *A, int8_t *B_all, int8_t *C, uint8_t index)
{
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP

	__shared__ int8_t sharedM[TILE_WIDTH][TILE_WIDTH];
	__shared__ int8_t sharedN[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;
	int v = 0;

	int8_t *B = B_all + index * BLOCK_SIZE * BLOCK_SIZE;

	for (int i = 0; i < (int)(ceil((float)BLOCK_SIZE / TILE_WIDTH)); i++)
	{
		if (i*TILE_WIDTH + tx < BLOCK_SIZE && row < BLOCK_SIZE)
			sharedM[ty][tx] = A[row*BLOCK_SIZE + i*TILE_WIDTH + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*TILE_WIDTH + ty < BLOCK_SIZE && col < BLOCK_SIZE)
			sharedN[ty][tx] = B[(i*TILE_WIDTH + ty)*BLOCK_SIZE + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; j++)
			v += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}

	if (row < BLOCK_SIZE && col < BLOCK_SIZE)
	{
		//extra calculate
		v = ((v & 0xFF) + ((v >> 8) & 0xFF)) & 0xFF;

		C[row*BLOCK_SIZE + col] = v;
	}
}

__global__ void Matrix_Mul(int8_t *md, int8_t *nd, int8_t *pd, uint8_t index)
{
	int bx, by, tx, ty;
	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	int mulResult = 0;
	int8_t *B = nd + index * 1 * BLOCK_SIZE * BLOCK_SIZE;		//sizeof(uint8_t) = 1
	for (int i = 0; i < gridDim.x; ++i)
	{
		__shared__ int8_t d_m[TILE_WIDTH][TILE_WIDTH];
		__shared__ int8_t d_n[TILE_WIDTH][TILE_WIDTH];
		d_m[ty][tx] = *(md + (by * blockDim.y + ty) * BLOCK_SIZE + i * blockDim.x + tx);
		d_n[ty][tx] = *(B + (i * blockDim.y + ty) * BLOCK_SIZE + bx * blockDim.x + tx);
		__syncthreads();
		for (int j = 0; j < blockDim.x; ++j)
		{
			mulResult += d_m[ty][j] * d_n[j][tx];
		}
		__syncthreads();
	}
	pd[(by*blockDim.y + ty)*BLOCK_SIZE + bx*blockDim.x + tx] = ((mulResult & 0xFF) + ((mulResult >> 8) & 0xFF)) & 0xFF;
}

__global__ void matrixExtraCal(int *sourceMatrix, int8_t *tmpMatrix)
{
	int tid = threadIdx.x + (int)blockIdx.x * blockDim.x;
	int curRow = tid / BLOCK_SIZE;
	int curCol = tid % BLOCK_SIZE;

	if (tid < BLOCK_SIZE * THREAD_SIZE)
	{
		int tmp = sourceMatrix[curRow * BLOCK_SIZE + curCol];
		//int8_t tmp2 = tmpMatrix[curRow * BLOCK_SIZE + curCol];
		tmpMatrix[curCol * BLOCK_SIZE + curRow] = ((tmp & 0xFF) + ((tmp >> 8) & 0xFF)) & 0xFF;
		//sourceMatrix[curRow * BLOCK_SIZE + curCol] = tmp2;
	}
}


cudaError_t matrixMul(Mat256x256i8& sourceMatrix, const Mat256x256i8* tmpMatrix, int8_t* matList, 
	uint8_t *sequence, uint32_t threadID)
{
	cudaSetDevice(threadID);
	cudaError_t cudaStatus;

	int alpha = 1;
	int beta = 0;

	int matrixSize = sizeof(int8_t) * 256 * 256;
	int *source;
	int8_t *tmp;
	//  *tmpSource;
	source = (int *)memory_pool->CMalloc(threadID, sizeof(int) * 256 * 256);
	tmp = (int8_t *)memory_pool->CMalloc(threadID, matrixSize);

	cudaStatus = cudaMemcpy(tmp, tmpMatrix->d, matrixSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	cublasStatus_t cublasSatus;
	//////////////////////////////////single kernel loop////////////////////////////////
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		for (int j = 0; j < SEQUENCE_COUNT; j++)
		{

			cublasSatus = cublasGemmEx(g_handle[threadID], CUBLAS_OP_T, CUBLAS_OP_T, 256, 256, 256,
				(void *)&alpha, (void *)tmp, CUDA_R_8I, 256,
				(void *)(matList + sequence[j] * matrixSize), CUDA_R_8I, 256,
				(void *)&beta, (void *)source, CUDA_R_32I, 256,
				CUDA_R_32I, CUBLAS_GEMM_DFALT);
			
			if (cublasSatus != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasGemmEx error!, j: %d cublasError: %d\n", j, cublasSatus);
			}

			matrixExtraCal << <256, 256 >> >(source, tmp);
			cudaDeviceSynchronize();

			if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
			{
				printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
				return cudaStatus;
			}
		}
	}

	cudaStatus = cudaMemcpy(sourceMatrix.d, tmp, matrixSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	memory_pool->CFree(threadID, tmp);
	memory_pool->CFree(threadID, source);

	//t2 = GetMillsec();
	//printf("\t kernel total time: %lfms\n", (t2 - t1));
	return cudaStatus;
}

typedef struct st_matrixMulThreadArg{
	int threadID;
	//int k;
	uint8_t *msg;
	//uint32_t len;
	//sha3_ctx *ctx;
	Mat256x256i8 *res;
	int8_t* device_matList;
	uint8_t *sequence;
}stMatrixMulThreadArg, *pstMatrixMulThreadArg;

void* matrixMul_Thread(void *arg)
{
	// double start_t, end_t;
	// start_t = GetMillsec();
	pstMatrixMulThreadArg matrixMulThreadArg = (pstMatrixMulThreadArg)arg;
	Mat256x256i8 *res = matrixMulThreadArg->res;
	//sha3_ctx *ctx = matrixMulThreadArg->ctx;
	int8_t* device_matList = matrixMulThreadArg->device_matList;
	int threadID = matrixMulThreadArg->threadID; 

	uint8_t *sequence = matrixMulThreadArg->sequence;
	//uint8_t sequence[128];
	//rhash_sha3_256_init(ctx);
	//rhash_sha3_update(ctx, matrixMulThreadArg->msg + (matrixMulThreadArg->len * matrixMulThreadArg->k / 4), matrixMulThreadArg->len / 4);
	//rhash_sha3_final(ctx, sequence);
	Mat256x256i8 *tmp = new Mat256x256i8;
	tmp->toIdentityMatrix();

	int alpha = 1;
	int beta = 0;

	int matrixSize = sizeof(int8_t) * 256 * 256;
	int *source;
	int8_t *tmpMatrix;
	source = (int *)memory_pool->CMalloc(threadID, sizeof(int) * 256 * 256);
	tmpMatrix = (int8_t *)memory_pool->CMalloc(threadID, matrixSize);

	cudaError_t cudaStatus = cudaSetDevice(threadID);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	cudaStatus = cudaMemcpy(tmpMatrix, tmp->d, matrixSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	for (int i = 0; i < LOOP_COUNT; i++)
	{
		for (int j = 0; j < SEQUENCE_COUNT; j++)
		{
			cublasStatus_t cublasSatus = cublasGemmEx(g_handle[threadID], CUBLAS_OP_T, CUBLAS_OP_T, 256, 256, 256,
				(void *)&alpha, (void *)tmpMatrix, CUDA_R_8I, 256,
				(void *)(device_matList + sequence[j] * matrixSize), CUDA_R_8I, 256,
				(void *)&beta, (void *)source, CUDA_R_32I, 256,
				CUDA_R_32I, CUBLAS_GEMM_DFALT);

			if (cublasSatus != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasGemmEx error!, j: %d cublasError: %d\n", j, cublasSatus);
			}

			matrixExtraCal << <256, 256 >> >(source, tmpMatrix);
			/*cudaDeviceSynchronize();

			if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
			{
				printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
			}*/
		}
	}

	cudaStatus = cudaMemcpy(res->d, tmpMatrix, matrixSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	memory_pool->CFree(threadID, tmpMatrix);
	memory_pool->CFree(threadID, source);

	//res[k].copyFrom(*mat);
	delete tmp;
	return NULL;
}

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


double GetMillsec()//ret ms
{
	double t1;
	struct timeval starttime;
	gettimeofday(&starttime, NULL);
	//sprintf(temp," send msg time:%f ms\n",(starttime.tv_sec*1000+starttime.tv_usec*0.001));
	t1 = starttime.tv_sec * 1000 + starttime.tv_usec*0.001;
	return t1;
}


void* calculate_Thread(void *arg)
{
	int threadID = *(int*)arg;
	cudaSetDevice(threadID);
	cudaError_t cudaStatus;

	Mat256x256i8 *res = new Mat256x256i8[4];
	Mat256x256i8 *mat = new Mat256x256i8;
	// sha3_ctx *ctx = (sha3_ctx*)calloc(4, sizeof(*sha3_ctx));
	sha3_ctx ctx[4];

	uint8_t **sequence = (uint8_t **)malloc(sizeof(uint8_t *) * 4);
	for (int i = 0; i < 4; i++)
	{
		sequence[i] = (uint8_t *)malloc(sizeof(uint8_t) * 128);
	}

	while (1)
	{
		// pTaskST tmpTask = g_tskQueue[threadID].OutQueue();
		pTaskST tmpTask;
		int i=threadID;
		// g_tskQueue[threadID].wait_and_pop(tmpTask);
		// tmpTask=g_tskQueue[threadID].Pop();
		if(i==0){
			tmpTask=g_tskQueue0.Pop();
			}else if(i==1){
				tmpTask=g_tskQueue1.Pop();
			}else if(i==2){
				tmpTask=g_tskQueue2.Pop();
			}else if(i==3){
				tmpTask=g_tskQueue3.Pop();
			}else if(i==4){
				tmpTask=g_tskQueue4.Pop();
			}else if(i==5){
				tmpTask=g_tskQueue5.Pop();
		}
		if (tmpTask != NULL)
		{
			//iter(tmpTask->msg, tmpTask->seed, tmpTask->len, tmpTask->result, tmpTask->threadID);
			uint8_t* seed = tmpTask->seed;
			uint8_t *msg = tmpTask->msg; 
			uint32_t len = tmpTask->len;
			uint8_t* result = tmpTask->result;
			// for (int j = 0; j < 1; j++) {
			// 	printf("seed %d 0x%02x 0x%02x\n",threadID,msg[j],seed[j]);
			// }
			int8_t* device_matList;

			// double start_t, end_t;
			// start_t = GetMillsec();
			//add mutex?
			if (memcmp(seed, g_seed, 32) == 0)
			{
				// printf("seed alread exist.\n");
			}
			else
			{
				printf("seed change.\n");
				AlgriMatList* matList_int8=new AlgriMatList;
				Words32 extSeed = extSeedCreate(seed);
				// memset(matList_int8->matVec, 0, sizeof(Mat256x256i8) * 256);
				// for (int i = 0; i<256; i++) {
				// 	matList_int8->matVec[i].toIdentityMatrix();
				// }
				matList_int8->init(extSeed);
				memcpy(g_seed, seed, 32);
				cudaStatus = cudaMemcpy(g_device_matList[threadID], matList_int8->matVec, sizeof(int8_t) * 256 * 256 * 256, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess)
					printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
				delete matList_int8;
			}
			device_matList = g_device_matList[threadID];

			//multeThread process 
			//start_t = GetMillsec();
			/*Mat256x256i8 *res = new Mat256x256i8[4];
			Mat256x256i8 *mat = new Mat256x256i8;
			sha3_ctx *ctx = (sha3_ctx*)calloc(1, sizeof(*ctx));
			uint8_t **sequence = (uint8_t **)malloc(sizeof(uint8_t *) * 4);*/

			pthread_t matrixMulThread[4];
			pstMatrixMulThreadArg matrixMulThreadArg = new stMatrixMulThreadArg[4]();
			for (int i = 0; i < 4; i++)
			{
				//uint8_t sequence[128];
				//sequence[i] = (uint8_t *)malloc(sizeof(uint8_t) * 128);
				memset(sequence[i], 0, sizeof(uint8_t) * 128);
				rhash_sha3_256_init(&ctx[i]);
				rhash_sha3_update(&ctx[i], msg + (len * i / 4), len / 4);
				rhash_sha3_final(&ctx[i], sequence[i]);

				matrixMulThreadArg[i].threadID = threadID;
				//matrixMulThreadArg[i].k = i;
				matrixMulThreadArg[i].msg = msg;
				//matrixMulThreadArg[i].len = len;
				//matrixMulThreadArg[i].ctx = ctx;
				matrixMulThreadArg[i].res = &(res[i]);
				matrixMulThreadArg[i].device_matList = device_matList;
				matrixMulThreadArg[i].sequence = sequence[i];

				if (pthread_create(&matrixMulThread[i], NULL, matrixMul_Thread, (void *)&matrixMulThreadArg[i]) != 0)
				{
					printf("ERROR: calculateThread create failed.\n");
					return NULL;
				}
			}

			for (int i = 0; i < 4; i++)
			{
				if (pthread_join(matrixMulThread[i], NULL) != 0)
				{
					printf("ERROR: calculateThread join failed.\n");
					return NULL;
				}				
			}
			//end_t = GetMillsec();
			//printf("threadID :%d, multiprocess: %lf\n", threadID, end_t - start_t);

			mat->add(res[0], res[1]);
			mat->add(*mat, res[2]);
			mat->add(*mat, res[3]);;
			Arr256x64i32 arr(*mat);
			arr.reduceFNV();
			// uint32_t data[64]={0};
			// arr.fillWithD0(data);
			sha3_ctx sctx;
			rhash_sha3_256_init(&sctx);
			// uint8_t* tt=arr.d0RawPtr();
			// rhash_sha3_update(ctx, (uint8_t *)data, 256);
			// for (int j = 0; j < 1; j++) {
			// 	printf("before data %d 0x%02x 0x%02x\n",threadID,tt[0],g_seed[j]);
			// }
			rhash_sha3_update(&sctx, arr.d0RawPtr(), 256);
			
			// for (int j = 0; j < 1; j++) {
			// 	printf("data %d 0x%02x 0x%02x\n",threadID,tt[0],g_seed[j]);
			// }
			// for (int j = 0; j < 1; j++) {
			// 	printf("before %d 0x%02x\n",threadID,result[j]);
			// }
			rhash_sha3_final(&sctx, result);
			// printf("%d ",threadID);
			// for (int j = 0; j < 4; j++) {
			// 	printf("0x%02x ",tmpTask->result[j]);
			// }
			// printf("\n");
			//end_t = GetMillsec();
			//printf("threadID :%d, multiprocess2: %lf\n", threadID, end_t - start_t);
			/*delete mat;
			delete[] res;
			free(ctx);*/

			// for (int j = 0; j < 32; j++) {
			// 	if (tmpTask->result[j] != g_results[j]) {
			// 		printf("Results does not match j : %d \n", j);
			// 		break;
			// 	}
			// }
			// free(tmpTask);
			// g_retQueue.InQueue(tmpTask);
			g_retQueue.Push(tmpTask);
			// ret_callback(tmpTask);
		}
		//usleep(1);
	}
	delete mat;
	delete[] res;
	// free(ctx); 
	for (int i = 0; i < 4; i++)
	{
		free(sequence[i]);
	}
	free(sequence);
	return NULL;
}