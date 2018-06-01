
#include "RequestQueue.h"
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>

pthread_mutex_t cTaskQueue::mutex = PTHREAD_MUTEX_INITIALIZER;

cTaskQueue::cTaskQueue(void)
{
	m_pHeader = NULL;
	m_pTail = NULL;
	m_isize =0;

	pthread_mutex_init(&mutex,NULL);
}

cTaskQueue::~cTaskQueue(void)
{
	pthread_mutex_destroy(&mutex);
}

void cTaskQueue::InQueue(pTaskST pNode)
{
	pTaskST pNewNode = pNode;
	pthread_mutex_lock(&mutex);
	if (!m_pHeader)
	{
		m_pHeader = pNewNode;
		m_pTail = pNewNode;
	}
	else
	{
		m_pTail->pNext = pNewNode;
		m_pTail = pNewNode;
	}
	m_isize++;
	pthread_mutex_unlock(&mutex);

}

pTaskST cTaskQueue::OutQueue()
{
	pTaskST pNewNode = NULL;

	pthread_mutex_lock(&mutex);

	if (m_pHeader)
	{
		pNewNode = m_pHeader;
		m_pHeader = m_pHeader->pNext;
		m_isize--;
	}

	pthread_mutex_unlock(&mutex);
	return pNewNode;

}

int cTaskQueue::getsize()
{
	int icount=0;
	pthread_mutex_lock(&mutex);
	icount = m_isize;
	pthread_mutex_unlock(&mutex);
	return icount;
}

