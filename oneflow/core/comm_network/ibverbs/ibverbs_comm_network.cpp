/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/dl/include/ibv.h"

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

namespace oneflow {

namespace {

std::string GenTokensMsgKey(int64_t machine_id) {
  return "IBVerbsTokensMsg/" + std::to_string(machine_id);
}

std::string GenConnInfoKey(int64_t src_machine_id, int64_t dst_machine_id) {
  return "IBVerbsConnInfo/" + std::to_string(src_machine_id) + "/" + std::to_string(dst_machine_id);
}

void IBVForkInit() {
  if (ibv::IsAvailable()) {
    if (ibv::wrapper.ibv_fork_init() != 0) { std::cerr << "ibv_fork_init failed\n"; }
  } else {
    std::cerr << "libibverbs not available, ibv_fork_init skipped\n";
  }
}

void ParseUserDevicePort(std::string* device_name, int* port) {
  std::string user_device_port = GetStringFromEnv("ONEFLOW_COMM_NET_IB_HCA", "");
  if (user_device_port.empty()) {
    *device_name = "";
    *port = 0;
    return;
  } else {
    const std::string::size_type pos = user_device_port.find(':', 0);
    if (pos == std::string::npos) {
      *device_name = user_device_port;
      *port = 0;
      return;
    } else {
      *device_name = user_device_port.substr(0, pos);
      *port = std::strtol(user_device_port.data() + pos + 1, nullptr, 10);
      return;
    }
  }
}

}  // namespace

IBVerbsCommNet::~IBVerbsCommNet() {
  while (poll_exit_flag_.test_and_set() == true) {}
  poll_thread_.join();
  for (IBVerbsQP* qp : qp_vec_) {
    if (qp) { delete qp; }
  }
  CHECK_EQ(ibv::wrapper.ibv_destroy_cq(cq_), 0);
  CHECK_EQ(ibv::wrapper.ibv_dealloc_pd(pd_), 0);
  CHECK_EQ(ibv::wrapper.ibv_close_device(context_), 0);
}

void IBVerbsCommNet::RegisterMemoryDone() {
  int64_t this_machine_id = GlobalProcessCtx::Rank();
  IBVerbsTokensMsg this_tokens_msg;
  for (IBVerbsMemDesc* mem_desc : mem_descs()) {
    this_tokens_msg.mutable_token2mem_desc()->insert(
        {reinterpret_cast<uint64_t>(mem_desc), mem_desc->ToProto()});
  }
  // TODO(chengcheng): Use Global<Transport> to sync session tokens.
  Global<CtrlClient>::Get()->PushKV(GenTokensMsgKey(this_machine_id), this_tokens_msg);
  for (int64_t peer_id : peer_machine_id()) {
    IBVerbsTokensMsg peer_tokens_msg;
    Global<CtrlClient>::Get()->PullKV(GenTokensMsgKey(peer_id), &peer_tokens_msg);
    for (const auto& pair : peer_tokens_msg.token2mem_desc()) {
      CHECK(token2mem_desc_.at(peer_id)
                .emplace(reinterpret_cast<void*>(pair.first), pair.second)
                .second);
    }
  }
  // TODO(chengcheng): change to OF_ENV_BARRIER
  OF_SESSION_BARRIER();
  Global<CtrlClient>::Get()->ClearKV(GenTokensMsgKey(this_machine_id));
}

void IBVerbsCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  qp_vec_.at(dst_machine_id)->PostSendRequest(msg);
}

IBVerbsCommNet::IBVerbsCommNet()
    : CommNetIf(),
      token2mem_desc_(Global<ResourceDesc, ForEnv>::Get()->process_ranks().size()),
      poll_exit_flag_(ATOMIC_FLAG_INIT) {
  int num_device;
  ibv_device** device_list = ibv::wrapper.ibv_get_device_list(&num_device);
  CHECK_GT(num_device, 0) << "No IB device found";
  PCHECK(device_list);
  std::string user_device;
  int user_port;
  ParseUserDevicePort(&user_device, &user_port);
  ibv_device* device = nullptr;
  if (user_device.empty()) {
    device = device_list[0];
  } else {
    for (int i = 0; i < num_device; ++i) {
      if (device_list[i]->name == user_device) {
        device = device_list[i];
        break;
      }
    }
    CHECK(device != nullptr) << "No IB device match " << user_device;
  }
  context_ = ibv::wrapper.ibv_open_device(device);
  CHECK(context_);
  ibv::wrapper.ibv_free_device_list(device_list);
  pd_ = ibv::wrapper.ibv_alloc_pd(context_);
  CHECK(pd_);
  ibv_device_attr device_attr{};
  CHECK_EQ(ibv::wrapper.ibv_query_device(context_, &device_attr), 0);
  cq_ = ibv::wrapper.ibv_create_cq(context_, device_attr.max_cqe, nullptr, nullptr, 0);
  CHECK(cq_);
  ibv_port_attr port_attr{};
  const uint8_t port = user_port == 0 ? 1 : user_port;
  CHECK_EQ(ibv::wrapper.ibv_query_port_wrap(context_, port, &port_attr), 0);
  ibv_gid gid{};
  const int64_t gid_index = ParseIntegerFromEnv("ONEFLOW_COMM_NET_IB_GID_INDEX", 0);
  CHECK_EQ(ibv::wrapper.ibv_query_gid(context_, port, gid_index, &gid), 0);
  LOG(INFO) << "Using IB device " << device->name << " port " << static_cast<int32_t>(port)
            << " gid index " << gid_index;
  int64_t this_machine_id = GlobalProcessCtx::Rank();
  qp_vec_.assign(Global<ResourceDesc, ForEnv>::Get()->process_ranks().size(), nullptr);
  for (int64_t peer_id : peer_machine_id()) {
    IBVerbsQP* cur_qp = new IBVerbsQP(context_, pd_, port, cq_, cq_);
    qp_vec_.at(peer_id) = cur_qp;
    IBVerbsConnectionInfo conn_info;
    conn_info.set_lid(port_attr.lid);
    conn_info.set_qp_num(cur_qp->qp_num());
    conn_info.set_subnet_prefix(gid.global.subnet_prefix);
    conn_info.set_interface_id(gid.global.interface_id);
    conn_info.set_port_num(port);
    conn_info.set_mtu(static_cast<int>(port_attr.active_mtu));
    Global<CtrlClient>::Get()->PushKV(GenConnInfoKey(this_machine_id, peer_id), conn_info);
  }
  for (int64_t peer_id : peer_machine_id()) {
    IBVerbsConnectionInfo conn_info;
    Global<CtrlClient>::Get()->PullKV(GenConnInfoKey(peer_id, this_machine_id), &conn_info);
    if (conn_info.lid() == 0) {
      LOG(INFO) << "Connecting to peer " << peer_id << " port " << conn_info.port_num() << " qpn "
                << conn_info.qp_num() << " gid index " << gid_index << " spn "
                << conn_info.subnet_prefix() << " iid " << conn_info.interface_id() << " mtu "
                << conn_info.mtu();
    } else {
      LOG(INFO) << "Connecting to peer " << peer_id << " port " << conn_info.port_num() << " qpn "
                << conn_info.qp_num() << " lid " << conn_info.interface_id() << " mtu "
                << conn_info.mtu();
    }
    qp_vec_.at(peer_id)->Connect(conn_info);
    LOG(INFO) << "Connected to peer " << peer_id;
  }
  // TODO(chengcheng): change to OF_ENV_BARRIER
  OF_SESSION_BARRIER();
  for (int64_t peer_id : peer_machine_id()) {
    qp_vec_.at(peer_id)->PostAllRecvRequest();
    Global<CtrlClient>::Get()->ClearKV(GenConnInfoKey(this_machine_id, peer_id));
  }
  // TODO(chengcheng): change to OF_ENV_BARRIER
  OF_SESSION_BARRIER();
  poll_thread_ = std::thread(&IBVerbsCommNet::PollCQ, this);
  // TODO(chengcheng): change to OF_ENV_BARRIER
  OF_SESSION_BARRIER();
}

void IBVerbsCommNet::DoRead(void* read_id, int64_t src_machine_id, void* src_token,
                            void* dst_token) {
  qp_vec_.at(src_machine_id)
      ->PostReadRequest(token2mem_desc_.at(src_machine_id).at(src_token),
                        *static_cast<const IBVerbsMemDesc*>(dst_token), read_id);
}

void IBVerbsCommNet::PollCQ() {
  std::vector<ibv_wc> wc_vec(max_poll_wc_num_);
  while (poll_exit_flag_.test_and_set() == false) {
    poll_exit_flag_.clear();
    int32_t found_wc_num = ibv_poll_cq(cq_, max_poll_wc_num_, wc_vec.data());
    CHECK_GE(found_wc_num, 0);
    FOR_RANGE(int32_t, i, 0, found_wc_num) {
      const ibv_wc& wc = wc_vec.at(i);
      CHECK_EQ(wc.status, IBV_WC_SUCCESS) << wc.opcode;
      WorkRequestId* wr_id = reinterpret_cast<WorkRequestId*>(wc.wr_id);
      IBVerbsQP* qp = wr_id->qp;
      switch (wc.opcode) {
        case IBV_WC_RDMA_READ: {
          qp->ReadDone(wr_id);
          break;
        }
        case IBV_WC_SEND: {
          qp->SendDone(wr_id);
          break;
        }
        case IBV_WC_RECV: {
          qp->RecvDone(wr_id);
          break;
        }
        default: UNIMPLEMENTED();
      }
    }
  }
}

const int32_t IBVerbsCommNet::max_poll_wc_num_ = 32;

COMMAND(IBVForkInit());

}  // namespace oneflow

#endif  // WITH_RDMA && OF_PLATFORM_POSIX
