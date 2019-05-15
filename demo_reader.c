#include<stdio.h>
#include<fcntl.h>
#include<pthread.h>
#include<unistd.h>
#include<stdlib.h>
#include<sys/time.h>
#include<rdma/rdma_cma.h>
#include<rdma/rdma_verbs.h>
#include<cuda/include/cuda.h>
#include<cuda/include/cuda_runtime_api.h>


unsigned int BUFFER_SIZE;
char *local_ipaddr_string = "10.0.17.2";
char *local_port_string = "10002";
char *remote_ipaddr_string = "10.0.17.2";
char *remote_port_string = "10000";
struct rdma_cm_id* listening;
struct ibv_mr* mr;
char* buf = 0;
unsigned int stop_flag;
struct rdma_cm_id* server_client;
struct rdma_cm_id* client;
struct timeval start_time;
struct timeval end_time;
uint64_t remote_mr_addr;
unsigned int remote_mr_rkey;
int WRITE = 0;
int GPU = 0;


int rdma_server_init() {
  struct rdma_addrinfo* addrinfo;
  struct rdma_addrinfo hint = {};
  hint.ai_port_space = RDMA_PS_TCP;
  hint.ai_flags = RAI_PASSIVE;

  if (rdma_getaddrinfo(local_ipaddr_string, local_port_string, &hint, &addrinfo)) {
    printf("%s %d can not resolve rdma:// %s : %s \n",
                    __func__, __LINE__, local_ipaddr_string, local_port_string);
    return 1;
  }

  struct ibv_qp_init_attr init_attr = {};
  init_attr.qp_type = IBV_QPT_RC;
  init_attr.cap.max_recv_wr = 1024;
  init_attr.cap.max_send_wr = 1;
  init_attr.cap.max_recv_sge = 1;
  init_attr.cap.max_send_sge = 1;


  if (rdma_create_ep(&listening, addrinfo, NULL, &init_attr) ) {
    printf("%s %d can not bind to rdma:// %s : %s \n",
                    __func__, __LINE__, local_ipaddr_string, local_port_string);
    return 1;
  }
  rdma_freeaddrinfo(addrinfo);

  if (rdma_listen(listening, 0)) {
    printf("%s %d can not listen on rdma:// %s : %s \n",
                    __func__, __LINE__, local_ipaddr_string, local_port_string);
    return 1;
  }

  if (listening->verbs == NULL) {
    printf("%s %d unsupported address %s : %s as it dose not bind to a particular rdma device \n",
                    __func__, __LINE__, local_ipaddr_string, local_port_string);
    return 1;
  }

  int flags = fcntl(listening->channel->fd, F_GETFL, 0);
  if (fcntl(listening->channel->fd, F_SETFL, flags | O_NONBLOCK)) {
    printf("%s %d can not set server to non-blocking mode \n",__func__, __LINE__);
    return 1;
  }

  if (GPU) {
    cudaMalloc((void **)&buf, BUFFER_SIZE);
    if (buf == NULL) {
      printf("%s %d can not malloc memory size of %x \n",
                     __func__, __LINE__, BUFFER_SIZE);
      return 1;
    }
    printf("%s %d buf addr %lu size 0x%x\n ",
           __func__, __LINE__, (unsigned long)buf, BUFFER_SIZE);

    cudaMemset(buf, 0x4, BUFFER_SIZE);

  } else {
    buf = malloc(BUFFER_SIZE);
    if (buf == NULL) {
      printf("%s %d can not malloc memory size of %x \n",
                     __func__, __LINE__, BUFFER_SIZE);
      return 1;
    }
    printf("%s %d buf addr %lu size 0x%x\n ",
           __func__, __LINE__, (unsigned long)buf, BUFFER_SIZE);

    memset(buf, 0x4, BUFFER_SIZE);
  }

  if (WRITE) {
    mr = rdma_reg_write(listening, (void *)buf, BUFFER_SIZE);
    if (mr == NULL) {
      printf("%s %d can not register memory region start %p size 0x%x \n",
                      __func__, __LINE__, (void *)buf, BUFFER_SIZE);
      return 1;
    }
  } else {
    mr = rdma_reg_read(listening, (void *)buf, BUFFER_SIZE);
    if (mr == NULL) {
      printf("%s %d can not register memory region start %p size 0x%x \n",
                      __func__, __LINE__, (void *)buf, BUFFER_SIZE);
      return 1;
    }
  }
  printf("%s %d rkey 0x%x \n", __func__, __LINE__, mr->rkey);
  printf("%s success\n", __func__);
  return 0;
}



void rdma_server_deinit() {

  if (listening) {
    rdma_destroy_ep(listening);
    listening = NULL;
  }

  if (server_client) {
    rdma_destroy_ep(server_client);
    server_client = NULL;
  }

  if (client) {
    rdma_destroy_ep(client);
    client = NULL;
  }

  if (mr) {
    rdma_dereg_mr(mr);
    mr = NULL;
  }

  if (buf) {
    cudaFree(buf);
    buf = NULL;
  }
  printf("%s success\n", __func__);
}

void dump_gpu_memory(void) {

  if (GPU) {
    char *host_buf = malloc(BUFFER_SIZE);
    if(!host_buf)
      return;

    cudaMemcpy(host_buf, buf, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    printf("%s %d callback to dump buf. \n", __func__, __LINE__);
    for (int i = 0; i < 50; i++)
    {
      printf("%x", host_buf[i]);
    }
    printf("\r\n");

    free(host_buf);
  } else {
    for (int i = 0; i < 50; i++)
    {
      printf("%x", buf[i]);
    }
    printf("\r\n");
  }

}

void rdma_server_run() {

  stop_flag = 0;

  printf("start the server \n");

  while(!stop_flag) {

    struct rdma_cm_id* id = NULL;
    if (!rdma_get_request(listening, &id)) {

      if (!rdma_accept(id, NULL)) {
        printf("Accepted new RDMA connection\n");
        // ???
        for (int i = 0; i < 1024; i++) {

          if (rdma_post_recvv(id, NULL, NULL, 0)) {
            printf("%s %d rdma_post_recvv failed", __func__, __LINE__);
            if (id) {
              rdma_destroy_ep(id);
            }
            continue;
          }
        }
        server_client = id;
      }
    }


    //server side work
    if (server_client != NULL) {
      struct ibv_wc wc[32];

      int ret = ibv_poll_cq(server_client->recv_cq, 32, wc);
      if (ret < 0) {
        printf("%s %d ibv_poll_cq failed \n", __func__, __LINE__);
        continue;
      }

      if (ret)
        printf("%s %d server side ibv_poll_cq wc %d \n", __func__, __LINE__, ret);

      for (int i = 0; i < ret; i++) {

        if (wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
          printf("%s %d received unknown operation %d\n",
                          __func__, __LINE__, wc[i].opcode);
        }

        if (wc[i].status != 0) {
          printf("%s %d received bad status %s \n",
                          __func__, __LINE__, ibv_wc_status_str(wc[i].status));
        }

        unsigned int key = ntohl(wc[i].imm_data);
        printf("%s %d server index %d key = 0x%x \n", __func__, __LINE__, i, key);

        if (rdma_post_recvv(server_client, NULL, NULL, 0)) {
          printf("%s %d rdma post recv failed", __func__, __LINE__);
        }

        gettimeofday(&end_time, NULL);
        printf("%s %d end_time %ld s %ld.%ld ms\n",
                        __func__, __LINE__,
		        end_time.tv_sec, end_time.tv_usec/1000, end_time.tv_usec%1000);

        dump_gpu_memory();
      }
    }

      //client side work
    if (client != NULL) {
      struct ibv_wc wcc[32];
      int ret = ibv_poll_cq(client->send_cq, 32, wcc);
      if (ret)
        printf("%s %d client side ibv_poll_cq wc %d \n", __func__, __LINE__, ret);

      for (int i = 0; i < ret; i++) {

        if (wcc[i].status) {
          printf("%s %d client received bad status %s\n",
                          __func__, __LINE__, ibv_wc_status_str(wcc[i].status));
        }

        unsigned int key = wcc[i].wr_id;
        printf("%s %d client index %d key = 0x%x \n", __func__, __LINE__, i, key);

        struct ibv_send_wr wr = {};
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.imm_data = htonl(key);
        struct ibv_send_wr* bad_wr;

        if (ibv_post_send(client->qp, &wr, &bad_wr)) {
          printf("%s %d ibv_post_send failed for key\n",
                          __func__, __LINE__);
        }

        gettimeofday(&end_time, NULL);
        printf("%s %d end_time %ld s %ld.%ld ms\n",
                        __func__, __LINE__,
		        end_time.tv_sec, end_time.tv_usec/1000, end_time.tv_usec%1000);

        dump_gpu_memory();
      }
    }
  }
}

int rdma_read_write_test() {

  struct rdma_addrinfo* addrinfo;
  struct rdma_addrinfo hint = {};
  hint.ai_port_space = RDMA_PS_TCP;

  if (rdma_getaddrinfo(remote_ipaddr_string, remote_port_string, &hint, &addrinfo)) {
    printf("%s %d can not connect to  rdma:// %s : %s \n",
                    __func__, __LINE__, remote_ipaddr_string, remote_port_string);
    return 1;
  }

  struct ibv_qp_init_attr init_attr = {};
  init_attr.qp_type = IBV_QPT_RC;
  init_attr.cap.max_recv_wr = 1;
  init_attr.cap.max_send_wr = 1024;
  init_attr.cap.max_recv_sge = 1;
  init_attr.cap.max_send_sge = 1;


  if (rdma_create_ep(&client, addrinfo, NULL, &init_attr) ) {
    rdma_freeaddrinfo(addrinfo);
    printf("%s %d can not create to rdma:// %s : %s \n",
                    __func__, __LINE__, remote_ipaddr_string, remote_port_string);
    return 1;
  }
  rdma_freeaddrinfo(addrinfo);

  if (rdma_connect(client, NULL)) {
    rdma_destroy_ep(client);
    printf("%s %d can not connected to rdma:// %s : %s \n",
                    __func__, __LINE__, remote_ipaddr_string, remote_port_string);

  }

  printf("%s %d RDMA endpoint connect to rdma:// %s : %s \n",
                    __func__, __LINE__, remote_ipaddr_string, remote_port_string);

  gettimeofday(&start_time, NULL);
  printf("%s %d start_time %ld s %ld.%ld ms\n",
         __func__, __LINE__,
         start_time.tv_sec, start_time.tv_usec/1000, start_time.tv_usec%1000);

  if (WRITE) {
    if (rdma_post_write(client, NULL, (void *)buf, BUFFER_SIZE,
                            mr, IBV_SEND_SIGNALED, remote_mr_addr, remote_mr_rkey)) {

      printf("%s %d rdma_post_read failed. \n", __func__, __LINE__);

    }
  } else {
    if (rdma_post_read(client, NULL, (void *)buf, BUFFER_SIZE,
                            mr, IBV_SEND_SIGNALED, remote_mr_addr, remote_mr_rkey)) {

      printf("%s %d rdma_post_write failed. \n", __func__, __LINE__);

    }
  }

  return 0;

}

int main(int argc, char *argv[], char* env[]) {

  pthread_t p_id;


  if (argc < 6) {
    printf("please enter size of buf, addr of remote, rkey of remote, read/write, cpu/gpu \n");
    return 0;
  }

  int size = atoi(argv[1]);
  BUFFER_SIZE = size * 1024 * 1024;

  unsigned long addr = atol(argv[2]);
  remote_mr_addr = (uint64_t)addr;
  printf("get remote mr addr %s 0x%lx \n", argv[2], remote_mr_addr);

  unsigned int rkey = atoi(argv[3]);
  remote_mr_rkey = (unsigned int)rkey;
  printf("get remote mr rkey %s 0x%x \n", argv[3], remote_mr_rkey);

  if (!strcmp(argv[4], "write")) {
    WRITE = 1;
  }

  if (!strcmp(argv[4], "gpu")) {
    GPU = 1;
  }

  if(!rdma_server_init()) {
    int ret = pthread_create(&p_id, NULL, (void *)rdma_server_run, NULL);
    if (ret) {
      printf("create pthread error \n");
      rdma_server_deinit();
      return 1;
    }

    rdma_read_write_test();

    while (!stop_flag)
      sleep(1000);
  }

  rdma_server_deinit();
  return 0;
}


