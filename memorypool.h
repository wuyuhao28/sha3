/**************************************************
description:内存管理器
author:zhoubin
time:2017-6-28
filename:MemoryPool.h
**************************************************/
#ifndef _MEMORYPOOL_H_  
#define _MEMORYPOOL_H_  
#include <stdlib.h>  

#ifdef __cplusplus
extern "C" {
#endif 

#define MINUNITSIZE 64  
#define ADDR_ALIGN 8  
#define SIZE_ALIGN MINUNITSIZE 

enum TYPE_MEMORY
{
    TYPE_IDLE = 0,
    TYPE_COMMON,
    TYPE_PERMANENT
};

enum MEMORYPOOL_TYPE
{
    TYPE_GPU = 1,
    TYPE_CPU = 2
};


typedef struct memory_block  
{  
    size_t count;  
    size_t start;
    short type;
}memory_block;   


// 内存池结构体   
typedef struct MEMORYPOOL  
{  
    void *memory;  
    size_t size; 
    int memory_type;                //memory_type
    memory_block* pmem_map;         //block数组头指针
    memory_block* pmem_map_end;     //block数组尾指针
    size_t mem_used_size; // 记录内存池中已经分配给用户的内存的大小  
    size_t mem_map_pool_count; // 记录链表单元缓冲池中剩余的单元的个数，个数为0时不能分配单元给pfree_mem_chunk  
    size_t mem_chunk_count; // 记录 pfree_mem_chunk链表中的单元个数  
    size_t mem_block_count; // 一个 mem_unit 大小为 MINUNITSIZE  
}MEMORYPOOL, *PMEMORYPOOL;

/************************************************************************/  
/* 生成内存池 
 * sBufSize: 内存池可用大小
 * 返回生成的内存池指针 
/************************************************************************/ 
PMEMORYPOOL _CreateMemoryPool(int index, size_t sBufSize, int memoryType);  


/************************************************************************/  
/* 暂时没用 
/************************************************************************/   
void _ReleaseMemoryPool(PMEMORYPOOL ppMem);   

/************************************************************************/  
/* 从内存池中分配指定大小的内存  
 * pMem: 内存池 指针 
 * sMemorySize: 要分配的内存大小 
 * 成功时返回分配的内存起始地址，失败返回NULL 
/************************************************************************/  
//void* GetMemory(size_t sMemorySize, PMEMORYPOOL pMem);  
void* _GetMemory(size_t sMemorySize, PMEMORYPOOL pMem, TYPE_MEMORY type = TYPE_COMMON);
/************************************************************************/  
/* 从内存池中释放申请到的内存 
 * pMem：内存池指针 
 * ptrMemoryBlock：申请到的内存起始地址 
/************************************************************************/  
int _FreeMemory(void *ptrMemoryBlock, PMEMORYPOOL pMem);  

/************************************************************************/  
/* 
内存映射表中的索引转化为内存起始地址                                           
                          
/************************************************************************/ 
void* index2addr(PMEMORYPOOL mem_pool, size_t index);

void cleanMemoryPool(int ,PMEMORYPOOL mem_pool);


#define    POOL_WALK(pool, iterator) \
    for(iterator = pool->pmem_map; (iterator != NULL && (iterator-pool->pmem_map)+(iterator->count) < pool->mem_block_count); (iterator = &(pool->pmem_map[((iterator-pool->pmem_map)+(iterator->count))])))


#ifdef __cplusplus
}
#endif 

#endif //_MEMORYPOOL_H 
