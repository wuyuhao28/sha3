#include "RequestQueue.h"

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