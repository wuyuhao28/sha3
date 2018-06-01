#include <stdio.h>
#include <iostream>
#include <time.h>
#include <malloc.h>
#include <pthread.h>
#include "interface.h"

static uint8_t g_msg[ 32] = {
      0xd0, 0xda, 0xd7, 0x3f, 0xb2, 0xda, 0xbf, 0x33,
      0x53, 0xfd, 0xa1, 0x55, 0x71, 0xb4, 0xe5, 0xf6,
      0xac, 0x62, 0xff, 0x18, 0x7b, 0x35, 0x4f, 0xad,
      0xd4, 0x84, 0x0d, 0x9f, 0xf2, 0xf1, 0xaf, 0xdf,
};

static uint8_t g_seed[ 32] = {
      0x07, 0x37, 0x52, 0x07, 0x81, 0x34, 0x5b, 0x11,
      0xb7, 0xbd, 0x0f, 0x84, 0x3c, 0x1b, 0xdd, 0x9a,
      0xea, 0x81, 0xb6, 0xda, 0x94, 0xfd, 0x14, 0x1c,
      0xc9, 0xf2, 0xdf, 0x53, 0xac, 0x67, 0x44, 0xd2,
};

static uint8_t g_results[ 32] = {
      0xe3, 0x5d, 0xa5, 0x47, 0x95, 0xd8, 0x2f, 0x85,
      0x49, 0xc0, 0xe5, 0x80, 0xcb, 0xf2, 0xe3, 0x75,
      0x7a, 0xb5, 0xef, 0x8f, 0xed, 0x1b, 0xdb, 0xe4,
      0x39, 0x41, 0x6c, 0x7e, 0x6f, 0x8d, 0xf2, 0x27,
};
// void ThreadUser(LPVOID deviceids){ //线程入口
void* ThreadUser( void* deviceids ) {
    clock_t start, end; 
    int deviceid=*((int*)deviceids);
    printf("start:%d\n",deviceid);
    uint8_t firstret[32] = {0};
    printf("test\n");
    uint32_t deviceidin=(uint32_t)deviceid;
    // get_hash_init();
    printf("test0\n");
    get_hash(g_msg,g_seed,firstret,&deviceidin);//剔除第一次的时间
    printf("test1\n");
    start = clock();
    for (int i = 0;i<1000; i++) {
      // uint8_t* ret=(uint8_t*)malloc(sizeof(uint8_t)*32);
      uint8_t ret[32] = {0};
      
      uint32_t tmp=(uint32_t)deviceid;
      int diff=get_hash(g_msg,g_seed,ret,&tmp);
          printf("test2\n");
      // for (int j = 0; j < 1; j++) {
      //   printf("%d 0x%02x\n",deviceid,ret[j]);
      // }
    }
    end = clock();
    printf("device %d,Time : %f t/s\n",deviceid,(double)(1000)/((double)(end - start)/CLOCKS_PER_SEC));
}
int main(void)
{
  pthread_t tids[6]; //线程id  
  int device[6]={0,1,2,3,4,5};
  void* status[6];
  // clock_t start, end; 
  // start = clock();
  get_hash_init();
  int devicenum=6;
  for( int i = 0; i < devicenum; ++i )  
  {  
      int ret = pthread_create( &tids[i], NULL, ThreadUser, (void*)&device[i] );
      if( ret!= 0 ) 
      {  
          std::cout << "pthread_create error:error_code=" << ret<< std::endl;  
      }  
  }  
  for(int i=0;i<devicenum;i++){
    pthread_join(tids[i], &status[i]);
  }
  // end = clock();
  // double alltime=(double)(end - start)/CLOCKS_PER_SEC;
  // printf("all Time : %f s\n",alltime);
  // printf("device all,Time : %f t/s\n",(double)(devicenum*1000)/alltime);
  getchar();
  return 0;
}

