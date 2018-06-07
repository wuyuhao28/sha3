#ifndef THREADSAFE_QUEUE_H_
#define THREADSAFE_QUEUE_H_
#include "ustd.h"

#pragma pack(push,1)
#include <queue>  
#include <memory>  
#include <mutex>  
#include <condition_variable>   
typedef struct tag_stTaskST
{
	uint8_t msg[32];
	uint32_t len;
	uint32_t threadID;
	uint8_t result[32];
	uint8_t seed[32];
  uint8_t target[32];
  char * jobid;
  uint8_t lenOfJobid;
  uint64_t nonce;
	struct tag_stTaskST *pNext;
}TaskST, *pTaskST;

  class SafeQueue{
  public:
    SafeQueue() = default;
    virtual ~SafeQueue(){};
    void Push(const pTaskST t );
    void Clear();
    pTaskST Pop();
    pTaskST Front();
    pTaskST Back();
    //pTaskST Size();
    bool Empty();
  private:
    std::queue<pTaskST> queue;
    mutable std::mutex q_mutex;
    mutable std::condition_variable q_cond;
  }; 

  void SafeQueue::Push( const pTaskST t ){
    std::lock_guard<std::mutex> lock(q_mutex);
    queue.push(t);
    q_cond.notify_one();
  }

  void SafeQueue::Clear(){
    std::lock_guard<std::mutex> lock(q_mutex);
    // queue.push(t);
    // q_cond.notify_one();
    while(!queue.empty()){
      pTaskST tmp=queue.front();
      if(tmp!=NULL)
      {
          free(tmp->jobid);
          free(tmp);
      }
        
        queue.pop();
    }
  }

  pTaskST SafeQueue::Pop(){
    std::unique_lock<std::mutex> lock(q_mutex);
    while(queue.empty()){
      q_cond.wait(lock);
    }
    pTaskST value = queue.front();
    queue.pop();
    return value;
  }

  pTaskST SafeQueue::Front(){
    return queue.front();
  }

  pTaskST SafeQueue::Back(){
    return queue.back();
  }

  // pTaskST SafeQueue::Size(){
  //   return queue.size();
  // }

  bool SafeQueue::Empty(){
    return queue.empty() ? true : false;
  }
#endif
