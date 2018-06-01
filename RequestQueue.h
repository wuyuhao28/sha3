/*�����������
//
//
*/

#ifndef _TASKQUEUE_H
#define _TASKQUEUE_H

#include <pthread.h>
#include "ustd.h"

#pragma pack(push,1)

typedef struct tag_stTaskST
{
	uint8_t *msg;
	uint32_t len;
	uint32_t threadID;
	uint8_t result[32];
	uint8_t seed[32];
	struct tag_stTaskST *pNext;
}TaskST, *pTaskST;

#pragma pack(pop)

class cTaskQueue
{
public:
	cTaskQueue(void);
	~cTaskQueue(void);
	
public:
	
	void InQueue(pTaskST pNode);
	pTaskST OutQueue();
	int getsize();
	
private:
	
	pTaskST m_pHeader;
	pTaskST m_pTail;
	int m_isize;
public:
	static pthread_mutex_t mutex;
};

#endif

