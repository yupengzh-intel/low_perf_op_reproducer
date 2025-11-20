# Operation Performance Comparison

# Basic Ops

## cos op

| Params| dtype | batch_size | dim_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|
| Value | bfloat16 | 131072 | 1024 | 1134.67  | 802.59  |

## sin op

| Params| dtype | batch_size | dim_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|
| Value | bfloat16 | 131072 | 1024 | 1101.52  | 803.36  |

## exp op

| Params| dtype | batch_size | dim_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|
| Value | float16 | 131072 | 1024 | 919.59  | 803.30  |

## gather op

| Params| dtype | src_batch_size | dst_batch_size | dim_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|
| Value | bfloat16 | 16384 | 16384 | 1024 | 1415.52  | 49.89  |

## scatter op

| Params| dtype | src_batch_size | dst_batch_size | dim_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|
| Value | bfloat16 | 16384 | 16384 | 1024 | 45130.37  | 46.50 (120.032)  |

## index_add op

| Params| dtype | src_batch_size | dst_batch_size | dim_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|
| Value | bfloat16 | 16384 | 16384 | 1024 | 412.25  | 193.57 (286.72)  |

## softmax op

| Params| dtype | batch_size| dim_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|
| Value | bfloat16 | 1024 | 16384 | 108.09  | 93.60 (55.968)  |

## topk op

| Params| dtype | batch_size | dim_size | k | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|
| Value | bfloat16 | 1024 | 16384 | 16 | 4073.73  | 194.27  |

# Mocked_model Ops

## rotary_embedding op

| Params| dtype | mode | batch_size | q_seq_len | cache_len | q_head_num | kv_head_num | head_dim | rope_offset | rope_dim | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|-|-|-|-|-|-|
| Value | bfloat16 | prefill | 1 | 8192 | 0 | 8 | 1 | 128 | 0 | 128 | 533.29  | 372.77  |

## moe_quant_group_gemm op

| Params | dtype | dst_dtype | num_tokens | world_size | rank | num_shared_experts | dp_size | sp_size | num_experts | topk | ep_size | hidden_size | new_hidden_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| Value | int8 | bfloat16 | 8192 | 8 | 0 | 0 | 1 | 1 | 64 | 4 | 8 | 4096 | 2048 | 3233.90  | 2010.69 |


## moe_scatter_dynamic_quant op

| Params | dtype | dst_dtype | hidden_size | num_tokens | world_size | rank | num_shared_experts | dp_size | sp_size | num_experts | topk | ep_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| Value | bfloat16 | int8 | 4096 | 8192 | 8 | 0 | 0 | 1 | 1 | 64 | 4 | 8 | 6517.79  | 1287.46  |

## moe_swiglu_dynamic_quant op

| Params | dtype | dst_dtype | hidden_size | num_tokens | world_size | rank | num_shared_experts | dp_size | sp_size | num_experts | topk | ep_size | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| Value | bfloat16 | int8 | 1024 | 8192 | 8 | 0 | 0 | 1 | 1 | 64 | 4 | 8 | 3383.04  | 1066.56  |

## head_rms_norm op

| Params | dtype | num_tokens | total_head_num | norm_head_start | norm_head_num | head_dim | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|-|-|
| Value | bfloat16 | 32768 | 8 | 0 | 8 | 128 | 1050.74  | 522.37  |

## store_kv_cache op

| Params | dtype | num_tokens | total_head_num | norm_head_start | norm_head_num | head_dim | HPU (us) | GPU (us) |
|-|-|-|-|-|-|-|-|-|
| Value | bfloat16 | 32768 | 8 | 0 | 8 | 128 | 644.49  | 167.80 |
