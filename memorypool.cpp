#include <memory.h>  
#include "memorypool.h"  
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

/************************************************************************/  
/* 内存池起始地址对齐到ADDR_ALIGN字节 
/************************************************************************/ 
size_t check_align_addr(void*& pBuf)  
{  
    size_t align = 0;  
    size_t addr = (size_t)pBuf;  
    align = (ADDR_ALIGN - addr % ADDR_ALIGN) % ADDR_ALIGN;  
    pBuf = (char*)pBuf + align;  
    return align;  
}  

/************************************************************************/  
/* 内存block大小对齐到MINUNITSIZE字节 
/************************************************************************/  
size_t check_align_block(size_t size)  
{  
    size_t align = size % MINUNITSIZE;  
      
    return size - align;   
}

/************************************************************************/  
/* 分配内存大小对齐到SIZE_ALIGN字节 
/************************************************************************/  
size_t check_align_size(size_t size)  
{  
    size = (size + SIZE_ALIGN - 1) / SIZE_ALIGN * SIZE_ALIGN;  
    return size;  
}

/************************************************************************/  
/* 根据 MINUNITSIZE字节 和SIZE_ALIGN字节 重新校验内存池申请大小
/************************************************************************/ 
size_t check_memory_size(size_t size)
{
	size_t endsize = size;
	
	endsize += (size%SIZE_ALIGN);
	endsize += (endsize%SIZE_ALIGN);
	
	printf("check_memory_size:size[%d], end_size[%d].\n",size, endsize);
    return endsize;
}

/************************************************************************/  
/* 
内存映射表中的索引转化为内存起始地址                                           
                          
/************************************************************************/ 
void* index2addr(PMEMORYPOOL mem_pool, size_t index)  
{  
    char* p = (char*)(mem_pool->memory);  
    void* ret = (void*)(p + index *MINUNITSIZE);  
      
    return ret;  
}  

/************************************************************************/  
/* 
内存起始地址转化为内存映射表中的索引                                           
                          
/************************************************************************/ 
size_t addr2index(PMEMORYPOOL mem_pool, void* addr)  
{  
    char* start = (char*)(mem_pool->memory);  
    char* p = (char*)addr;  
    size_t index = (p - start) / MINUNITSIZE;  
    return index;  
}

/************************************************************************/  
/* 生成内存池 
 * sBufSize: 内存池可用大小
 * 返回生成的内存池指针 
/************************************************************************/ 
PMEMORYPOOL _CreateMemoryPool(int index, size_t sBufSize, int memoryType)
{
	
	printf("create memory pool \n");
    //申请内存值指针
	PMEMORYPOOL mem_pool = (PMEMORYPOOL)malloc(sizeof(MEMORYPOOL));
	if (NULL == mem_pool)
	{
		printf("[%s %d|Error] CreateMemoryPool failed. can not malloc mem_pool point. size[%d]\n",__FILE__,__LINE__,sizeof(MEMORYPOOL));
		return NULL;
	}

	//计算大小
	mem_pool->size = check_memory_size(sBufSize);
    mem_pool->memory_type = memoryType;
    printf("_CreateMemoryPool:size[%ld], end size[%ld].\n",sBufSize, mem_pool->size);
	mem_pool->mem_block_count = mem_pool->size / MINUNITSIZE;
	mem_pool->mem_map_pool_count = mem_pool->mem_block_count;
	//mem_pool->mem_map_unit_count = mem_pool->mem_block_count;

	//初始化pmem_map
	mem_pool->pmem_map = (memory_block*)malloc(sizeof(memory_block) * mem_pool->mem_block_count);   //数组的前指针
    if (NULL == mem_pool->pmem_map)
    {
        printf("[%s %d|Error] CreateMemoryPool failed. can not malloc mem_pool->pmem_map point. size[%d]\n",__FILE__,__LINE__, sizeof(memory_block) * mem_pool->mem_block_count);
        free(mem_pool);
		return NULL;
    }

    memset(mem_pool->pmem_map,0,sizeof(memory_block) * mem_pool->mem_block_count);
	mem_pool->pmem_map_end = &(mem_pool->pmem_map[mem_pool->mem_block_count-1]);     //block数组的尾指针
    mem_pool->pmem_map[0].count = mem_pool->mem_block_count;             //chunk中block的个数
    mem_pool->pmem_map[0].type = TYPE_IDLE;                                 //类型
    //mem_pool->pmem_map[0].pos = 0;
    mem_pool->pmem_map[mem_pool->mem_block_count-1].start = 0;    //chunk中最后一个block的开始位移

    mem_pool->mem_chunk_count = 1;                           //可用的chunk个数
	mem_pool->mem_used_size = 0;                                  //内存中已经使用的内存大小

    
    // mem_pool->memory = (char *)malloc(mem_pool->size);           //数据起始指针
    if (mem_pool->memory_type == TYPE_GPU)
    {
    	cudaSetDevice(index);
    	cudaError_t r = cudaMalloc((void**)&(mem_pool->memory), mem_pool->size);				//分配显存
    	if(r != cudaSuccess)
    	{
    	    free(mem_pool->pmem_map);
            free(mem_pool);
        	return NULL;
    	}
    }
    else
    {
        mem_pool->memory = (char *)malloc(mem_pool->size); 
		memset(mem_pool->memory, 0, mem_pool->size);
    }
	
    if (NULL == mem_pool->memory)
    {
        printf("[%s %d|Error] CreateMemoryPool failed. can not malloc mem_pool->memory point. size[%ld]\n",__FILE__,__LINE__, mem_pool->size);
        free(mem_pool->pmem_map);
        free(mem_pool);
		return NULL;
    }
	printf("create is over\n");
	
    return mem_pool;
}


void cleanMemoryPool(int num, PMEMORYPOOL mem_pool)
{
	memset(mem_pool->pmem_map, 0, sizeof(memory_block) * mem_pool->mem_block_count);
	mem_pool->pmem_map_end = &(mem_pool->pmem_map[mem_pool->mem_block_count - 1]);     //block数组的尾指针
	mem_pool->pmem_map[0].count = mem_pool->mem_block_count;             //chunk中block的个数
	mem_pool->pmem_map[0].type = TYPE_IDLE;                                 //类型
	//mem_pool->pmem_map[0].pos = 0;
	mem_pool->pmem_map[mem_pool->mem_block_count - 1].start = 0;    //chunk中最后一个block的开始位移

	mem_pool->mem_chunk_count = 1;                           //可用的chunk个数
	mem_pool->mem_used_size = 0;                                  //内存中已经使用的内存大小
    if (mem_pool->memory_type == TYPE_GPU)
    {
    	cudaError_t r = cudaMemset((void**)&(mem_pool->memory),0x00,  mem_pool->size);				//分配显存
    	if (r != cudaSuccess)
    	{
    		return;
    	}
    }
    else
    {
        memset(mem_pool->memory, 0x00, mem_pool->size);
    }
}

/************************************************************************/  
/* 暂时没用 
/************************************************************************/  
void _ReleaseMemoryPool(PMEMORYPOOL ppMem)   
{  
}  

/************************************************************************/  
/* 从内存池中分配指定大小的内存  
* pMem: 内存池 指针 
* sMemorySize: 要分配的内存大小 
* 成功时返回分配的内存起始地址，失败返回NULL 
/************************************************************************/  
void* _GetMemory(size_t sMemorySize, PMEMORYPOOL pMem, TYPE_MEMORY type)
{
    if (NULL == pMem)
    {
        return NULL;
    }
    
    //校准内存
    sMemorySize = check_align_size(sMemorySize);  
    
    size_t index = 0;
    //从左到右查找
    if (type == TYPE_COMMON)
    {
        memory_block* temp = pMem->pmem_map;
        for (index = 0; index < pMem->mem_chunk_count; index++)
        {
            if (temp->count * MINUNITSIZE >= sMemorySize && temp->type == TYPE_IDLE)  
            {             
                break;  
            }  

            //如果不是最后一个，移动至下一块内存块
            if (index < pMem->mem_chunk_count-1)
            {
                temp = (memory_block*)((char *)temp + (temp->count)*sizeof(memory_block));
            }
        }

        if (index == pMem->mem_chunk_count)  
        {  
            printf("GetMemory error. not enough memory. size[%ld]\n", sMemorySize);
            return NULL;  
        }

        //计算统计相关
        pMem->mem_used_size += sMemorySize;
        size_t needcount = sMemorySize/MINUNITSIZE;
        pMem->mem_map_pool_count -= needcount;

        //计算偏移量
        size_t current_index = temp-pMem->pmem_map;
        
        //需要切割数组
        if (temp->count > needcount)
        {
            //修改尾
            pMem->pmem_map[current_index+needcount-1].start = current_index;

            //修改下一个chunk
            pMem->pmem_map[current_index+needcount].count = temp->count-needcount;
            //pMem->pmem_map[current_index+needcount].type = TYPE_IDLE;   //最开始已经初始化，暂时屏蔽掉
            pMem->pmem_map[current_index+temp->count-1].start = current_index+needcount;

            //修改当前temp
            temp->count = needcount;

            //标记内存type
            temp->type = TYPE_COMMON;

            //修改chunk数量
            pMem->mem_chunk_count++;
            
        //不切割数组
        }
        else if (temp->count == needcount)
        {
            temp->type = TYPE_COMMON;
        }

        //返回数据可用真是地址
        return index2addr(pMem, current_index);
        
    }
    //从右向左
    else if (type == TYPE_PERMANENT)
    {
        memory_block* temp = pMem->pmem_map_end;
        for (index = 0; index < pMem->mem_chunk_count; index++)
        {
            //获取chunk头
            memory_block* start = &(pMem->pmem_map[temp->start]);
            if (start->count * MINUNITSIZE >= sMemorySize && start->type == TYPE_IDLE)  
            {             
                break;  
            }  

            //如果不是最前一个，移动至前一块内存块尾部
            if (index < pMem->mem_chunk_count-1)
            {
                temp = (memory_block*)((char *)temp - (start->count)*sizeof(memory_block));
            }
        }

        if (index == pMem->mem_chunk_count)  
        {  
            printf("GetMemory error. not enough memory. size[%ld]\n", sMemorySize);
            return NULL;  
        }

        //获取可分配chunk头
        memory_block* start = &(pMem->pmem_map[temp->start]);

        //计算统计相关
        pMem->mem_used_size += sMemorySize;
        size_t needcount = sMemorySize/MINUNITSIZE;
        pMem->mem_map_pool_count -= needcount;

        //计算偏移量
        size_t current_index = temp-pMem->pmem_map;
        
        //需要切割数组
        if (start->count > needcount)
        {
            //修改要分配的count
            pMem->pmem_map[current_index-needcount+1].count = needcount;
            pMem->pmem_map[current_index-needcount+1].type = TYPE_PERMANENT;
            temp->start = current_index-needcount+1;

            //修改前一个chunk的最后一个block
            pMem->pmem_map[current_index-needcount].start = current_index-start->count+1;

            //修改前一个
            start->count = start->count - needcount;
            //start->type = TYPE_IDLE;
            //修改chunk数量
            pMem->mem_chunk_count++;
        }
        //不切割数组
        else if (start->count == needcount)
        {
            start->type = TYPE_PERMANENT;
        }

        //返回数据可用真是地址
        return index2addr(pMem, current_index-needcount+1);
    }
    return NULL;
}

/************************************************************************/  
/* 从内存池中释放申请到的内存 
* pMem：内存池指针 
* ptrMemoryBlock：申请到的内存起始地址 
/************************************************************************/  
int _FreeMemory(void *ptrMemoryBlock, PMEMORYPOOL pMem)   
{
    if (NULL == ptrMemoryBlock || NULL == pMem)
        return -1;

    size_t current_index = addr2index(pMem, ptrMemoryBlock);  
    if (current_index < 0 || current_index > pMem->mem_block_count-1)
    {
        return -1;
    }
    
    size_t count = pMem->pmem_map[current_index].count;
    size_t size = count * MINUNITSIZE; 

    bool merge_front = false;
    bool merge_back = false;

    //判断向前能不能合并
    if (current_index > 0)
    {
        size_t front_index = pMem->pmem_map[current_index-1].start;
        memory_block* front_block = &(pMem->pmem_map[front_index]);
        if (front_block->type == TYPE_IDLE)
        {
            merge_front = true;
            //printf("debug...  _FreeMemory front.\n");
            
            pMem->pmem_map[current_index-1].start = 0;          
            front_block->count = front_block->count+count;
            
            pMem->pmem_map[current_index].type = TYPE_IDLE;
            pMem->pmem_map[current_index].count = 0;
            pMem->pmem_map[current_index+count-1].start = front_index;

            //讲current指针向前移动
            current_index = front_index;
            
            //修改chunk数量
            pMem->mem_chunk_count--;
            
        }
    }

    //判断向后能不能合并
    if (current_index < pMem->mem_block_count-1)
    {
        size_t back_index = pMem->pmem_map[current_index].count + current_index;
        if (back_index < pMem->mem_block_count-1)
        {
            memory_block* back_block = &(pMem->pmem_map[back_index]);
            if (back_block->type == TYPE_IDLE)
            {
                merge_back = true;

                //printf("debug...  _FreeMemory back.\n");
                
                //更新count 
                size_t  temp_count = pMem->pmem_map[current_index].count;

                //back_count
                size_t back_count = back_block->count;
                
                //修改
                pMem->pmem_map[back_index-1].start = 0;
                pMem->pmem_map[current_index].count = temp_count+back_count;
                pMem->pmem_map[current_index].type = TYPE_IDLE;

                pMem->pmem_map[back_index+back_count-1].start = current_index;
                
                //修改chunk数量
                pMem->mem_chunk_count--;
            }
        }
    }


    //如果前后都不可以合并
    if (!merge_back && !merge_front)
    {
        pMem->pmem_map[current_index].type = TYPE_IDLE;
    }

    //修改统计值
    pMem->mem_used_size -= size;
    pMem->mem_map_pool_count += count;    

    return 0;
}


