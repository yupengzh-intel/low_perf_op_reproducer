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


def smooth_per_token_dynamic_quant(
    hidden_states, 
    smooth_scale, 
    dst_torch_dtype=torch.int8
):
    smoothed_input = torch.mul(hidden_states, smooth_scale).type(torch.float32)
    per_token_scale = torch.div(torch.max(smoothed_input.abs(), -1, keepdim=False)[0], 127.0)
    quant_tokens = torch.div(smoothed_input, per_token_scale.unsqueeze(-1)).round().type(dst_torch_dtype)
    return quant_tokens, per_token_scale


class RotaryEmbeddingOp:

    def __init__(self, mode="prefill"):
        # pre-defined attrs
        self.q_head_num = 8
        self.kv_head_num = 1
        self.head_dim = 128
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num

        self.rope_offset = 0
        self.rope_dim = 128

        self.mode = mode
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
        

        # SRAM 96MB
        cache_size = 96 * (1024 ** 2)
        tensor_size = self.tensor_size()
        max_data_cnt = math.ceil(cache_size / tensor_size)
        self.tensor_list = self.create_tensors(max_data_cnt)
        random.shuffle(self.tensor_list)
    
    def create_tensors(self, max_data_cnt):
        all_tensor_list = []

        # return cos/sin
        def precompute_freqs_cis(dim, max_seq_len, theta: float = 10000.0):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            t = torch.arange(max_seq_len, device=freqs.device)
            freqs = torch.outer(t, freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            return torch.real(freqs_cis), torch.imag(freqs_cis)

        cos_tensor, sin_tensor = precompute_freqs_cis(
            self.rope_dim, 
            self.max_kv_len
        )
        
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

            cos = cos_tensor.to("cuda")

            sin = sin_tensor.to("cuda")

            y = torch.randn(
                size=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=torch.bfloat16,
                device="cuda"
            )

            tensors = {
                'packed_qkv': packed_qkv,
                'q_lens': q_lens,
                'accum_q_lens': accum_q_lens,
                'cache_lens': cache_lens,
                'cos': cos,
                'sin': sin,
                'y': y
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
        
        # cos: [max_kv_len, rope_dim//2], dtype=torch.float32
        size += self.max_kv_len * (self.rope_dim // 2) * torch.float32.itemsize
        
        # sin: [max_kv_len, rope_dim//2], dtype=torch.float32
        size += self.max_kv_len * (self.rope_dim // 2) * torch.float32.itemsize
        
        # y: [num_tokens, total_head_num, head_dim], dtype=torch.bfloat16
        size += self.num_tokens * self.total_head_num * self.head_dim * torch.bfloat16.itemsize
        
        return size
    
    def op(self, tensors):
        # get pre-allocated input tensors
        packed_qkv = tensors["packed_qkv"]
        q_lens = tensors["q_lens"]
        accum_q_lens = tensors["accum_q_lens"]
        cache_lens = tensors["cache_lens"]
        cos = tensors["cos"]
        sin = tensors["sin"]

        # get pre-allocated output tensors
        y = tensors["y"]


        def rotate(qk, cos, sin):
            # [q_seq_len, q_head_num + kv_head_num, head_dim]
            x1 = qk[..., :self.rope_dim//2]
            x2 = qk[..., self.rope_dim//2:]

            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

            x1_rot = x1 * cos - x2 * sin
            x2_rot = x1 * sin + x2 * cos

            return torch.cat([x1_rot, x2_rot], dim=-1)
            

        # for each batch
        for batch_idx in range(self.batch_size):
            q_len = q_lens[batch_idx]
            q_offset = accum_q_lens[batch_idx]
            cur_cache_len = cache_lens[batch_idx]

            token_start = q_offset
            token_end = q_offset + q_len

            qk_head_start = 0
            qk_head_end = self.q_head_num + self.kv_head_num

            dim_start = self.rope_offset
            dim_end = self.rope_offset + self.rope_dim

            cache_start = cur_cache_len
            cache_end = cur_cache_len + q_len

            y[token_start:token_end, qk_head_start:qk_head_end, dim_start:dim_end] = rotate(
                packed_qkv[token_start:token_end, qk_head_start:qk_head_end, dim_start:dim_end], 
                cos[cache_start : cache_end], 
                sin[cache_start : cache_end]
            )

        return y
    
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
    op = RotaryEmbeddingOp()
    op.run()
