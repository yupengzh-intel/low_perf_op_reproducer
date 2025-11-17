import torch
import random
import math

def generate_prefill_data(
    q_seq_len, cache_len
):
    q_lens = [q_seq_len]
    accum_q_lens = [0, q_seq_len]
    cache_lens = [cache_len]
    cache_slot_ids = [0]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


class StoreKVCacheOp:

    def __init__(self, mode="prefill"):
        # pre-defined attrs
        self.q_head_num = 8
        self.kv_head_num = 1
        self.head_dim = 128
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num

        self.mode = "prefill"
        if self.mode == "prefill":
            # [q_seq_len, total_head_num, head_dim]
            self.batch_size = 1
            self.q_seq_len = 8192
            self.cache_len = 0

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                generate_prefill_data(
                    self.q_seq_len, 
                    self.cache_len
                )
        
        # accum q_lens
        self.num_tokens = sum(self.q_lens)
        # accum cache_lens
        self.num_cache_tokens = sum(self.cache_lens)
        # max q_len + cache_len
        self.max_kv_len = max(self.kv_lens)
        

        # SRAM 48MB
        cache_size = 48 * (1024 ** 2)
        tensor_size = self.tensor_size()
        max_data_cnt = math.ceil(cache_size / tensor_size)
        self.tensor_list = self.create_tensors(max_data_cnt)
        random.shuffle(self.tensor_list)
    
    def create_tensors(self, max_data_cnt):
        all_tensor_list = []

        for _ in range(max_data_cnt):
            packed_qkv = torch.randn(
                size=[self.num_tokens, self.total_head_num, self.head_dim], 
                dtype=torch.bfloat16, 
                device="cuda"
            )

            q_lens = torch.tensor(
                self.q_lens, 
                dtype=torch.int32, 
                device="cuda"
            )

            accum_q_lens = torch.tensor(
                self.accum_q_lens, 
                dtype=torch.int32, 
                device="cuda"
            )

            cache_lens = torch.tensor(
                self.cache_lens, 
                dtype=torch.int32, 
                device="cuda"
            )

            cache_slot_ids = torch.tensor(
                self.cache_slot_ids, 
                dtype=torch.int32, 
                device="cuda"
            )

            k_cache = torch.randint(
                low=-16,
                high=17,
                size=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=torch.int8, 
                device="cuda"
            )

            v_cache = torch.randint(
                low=-16,
                high=17,
                size=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=torch.int8, 
                device="cuda"
            )

            k_scale = torch.ones(
                size=[self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device="cuda"
            )

            v_scale = torch.ones(
                [self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device="cuda"
            )


            tensors = {
                'packed_qkv': packed_qkv,
                'q_lens': q_lens,
                'accum_q_lens': accum_q_lens,
                'cache_lens': cache_lens,
                'cache_slot_ids': cache_slot_ids,
                'k_cache': k_cache,
                'v_cache': v_cache,
                'k_scale': k_scale,
                'v_scale': v_scale
            }
            all_tensor_list.append(tensors)
            
        return all_tensor_list


    def tensor_size(self):
        size = 0
        
        # packed_qkv: [num_tokens, total_head_num, head_dim], dtype=torch.bfloat16
        size += self.num_tokens * self.total_head_num * self.head_dim * torch.bfloat16.itemsize
        
        # q_lens: [batch_size], dtype=torch.int32
        size += len(self.q_lens) * torch.int32.itemsize
        
        # accum_q_lens: [batch_size + 1], dtype=torch.int32
        size += len(self.accum_q_lens) * torch.int32.itemsize
        
        # cache_lens: [batch_size], dtype=torch.int32
        size += len(self.cache_lens) * torch.int32.itemsize
        
        # cache_slot_ids: [batch_size], dtype=torch.int32
        size += len(self.cache_slot_ids) * torch.int32.itemsize
        
        # k_cache: [batch_size, kv_head_num, max_kv_len, head_dim], dtype=torch.int8
        size += self.batch_size * self.kv_head_num * self.max_kv_len * self.head_dim * torch.int8.itemsize
        
        # v_cache: [batch_size, kv_head_num, max_kv_len, head_dim], dtype=torch.int8
        size += self.batch_size * self.kv_head_num * self.max_kv_len * self.head_dim * torch.int8.itemsize
        
        # k_scale: [kv_head_num, head_dim], dtype=torch.float32
        size += self.kv_head_num * self.head_dim * torch.float32.itemsize
        
        # v_scale: [kv_head_num, head_dim], dtype=torch.float32
        size += self.kv_head_num * self.head_dim * torch.float32.itemsize
        
        return size
    
    def op(self, tensors):
        # get pre-allocated input tensors
        packed_qkv = tensors["packed_qkv"]
        q_lens = tensors["q_lens"]
        accum_q_lens = tensors["accum_q_lens"]
        cache_lens = tensors["cache_lens"]
        cache_slot_ids = tensors["cache_slot_ids"]
        k_cache = tensors["k_cache"]
        v_cache = tensors["v_cache"]
        k_scale = tensors["k_scale"]
        v_scale = tensors["v_scale"]

        # for each batch
        for batch_idx in range(self.batch_size):
            q_len = q_lens[batch_idx]
            q_offset = accum_q_lens[batch_idx]
            cur_cache_len = cache_lens[batch_idx]
            cur_slot_id = cache_slot_ids[batch_idx]

            token_start = q_offset
            token_end = q_offset + q_len

            k_head_start = self.q_head_num
            k_head_end = self.q_head_num + self.kv_head_num
            v_head_start = self.q_head_num + self.kv_head_num
            v_head_end = self.q_head_num + self.kv_head_num * 2

            cache_start = cur_cache_len
            cache_end = cur_cache_len + q_len

            # [num_tokens, total_head_num, head_dim]
            # --> [q_len, kv_head_num, head_dim]
            # --> [kv_head_num, q_len, head_dim]
            cur_k = packed_qkv[token_start:token_end, k_head_start:k_head_end].transpose(0, 1)
            cur_v = packed_qkv[token_start:token_end, v_head_start:v_head_end].transpose(0, 1)

            # [max_batch_size, total_head_num, max_seq_len, head_dim]
            # --> [kv_head_num, q_len, head_dim]

            k_cache[cur_slot_id, k_head_start:k_head_end, cache_start:cache_end] = torch.round(
                torch.mul(cur_k, k_scale.unsqueeze(1))).type(torch.int8)
            v_cache[cur_slot_id, v_head_start:v_head_end, cache_start:cache_end] = torch.round(
                torch.mul(cur_v, v_scale.unsqueeze(1))).type(torch.int8)
    
    def perf(self, iterations, profiling=False):
        if profiling:
            schedule = torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
            )
            prof=torch.profiler.profile(
                schedule=schedule,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(".", use_gzip=True),
                record_shapes=True,
                with_modules=False,
                profile_memory=False,
                with_stack=True,
            )

            prof.start()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(iterations):
            _ = self.op(self.tensor_list[i % len(self.tensor_list)])
        end_event.record()
        end_event.synchronize()

        if profiling:
            prof.stop()

        latency_us = start_event.elapsed_time(end_event) * 1e3 / iterations
        return latency_us
    
    def run(self, profiling=False):
        # warmup
        self.perf(2)
        latency_us = self.perf(32, profiling)
        print(f"{latency_us=:.2f}")
    
if __name__ == "__main__":
    op = StoreKVCacheOp()
    op.run()
