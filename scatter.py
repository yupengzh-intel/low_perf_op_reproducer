import torch
import habana_frameworks.torch as ht
import random
import math

class ScatterOp:

    def __init__(self):
        self.src_batch_size = 16384
        self.dst_batch_size = 16384
        self.dim_size = 1024

        # SRAM 48MB
        cache_size = 48 * (1024 ** 2)
        tensor_size = self.tensor_size()
        max_data_cnt = math.ceil(cache_size / tensor_size)
        self.tensor_list = self.create_tensors(max_data_cnt)
        random.shuffle(self.tensor_list)
    
    def create_tensors(self, max_data_cnt):
        all_tensor_list = []
        for _ in range(max_data_cnt):
            src = torch.randn(
                size=(self.src_batch_size, self.dim_size), 
                dtype=torch.bfloat16, 
                device="hpu"
            )

            random_index = []
            for value in range(self.dst_batch_size):
                random_index.append(value % self.src_batch_size)
            random.shuffle(random_index)
            index = torch.tensor(
                random_index, 
                dtype=torch.int64, 
                device="hpu"
            ).view(self.dst_batch_size, 1).expand(self.dst_batch_size, self.dim_size)

            dst = torch.randn(
                size=(self.dst_batch_size, self.dim_size), 
                dtype=torch.bfloat16, 
                device="hpu"
            )
            tensors = {
                'src': src,
                'index': index,
                'dst': dst
            }
            all_tensor_list.append(tensors)
            ht.core.mark_step()
        torch.hpu.synchronize()
            
        return all_tensor_list


    def tensor_size(self):
        size=0
        # src
        size += self.src_batch_size * self.dim_size * torch.bfloat16.itemsize
        # index
        size += self.dst_batch_size * self.dim_size * torch.int64.itemsize
        # dst
        size += self.dst_batch_size * self.dim_size * torch.bfloat16.itemsize
        return size
    
    def op(self, tensors):
        src = tensors['src']
        index = tensors['index']
        dst = tensors['dst']
        dst.scatter_(
            dim=0, 
            index=index,
            src=src,
        )
        return dst
    
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
                    torch.profiler.ProfilerActivity.HPU,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(".", use_gzip=True),
                record_shapes=True,
                with_modules=False,
                profile_memory=False,
                with_stack=True,
            )

            prof.start()

        start_event = torch.hpu.Event(enable_timing=True)
        end_event = torch.hpu.Event(enable_timing=True)
        start_event.record()
        for i in range(iterations):
            _ = self.op(self.tensor_list[i % len(self.tensor_list)])
            ht.core.mark_step()
        torch.hpu.synchronize()
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
        print("scatter:")
        print(f"src_batch_size={self.src_batch_size}")
        print(f"dst_batch_size={self.dst_batch_size}")
        print(f"dim_size={self.dim_size}")
        print(f"{latency_us=}")
        mem_bw = self.tensor_size()/latency_us/1e3
        print(f"mem_bw(GB/s)={mem_bw}")
    
if __name__ == "__main__":
    op = ScatterOp()
    op.run()