# Operation Performance Comparison

basic op

| Operation Name | HPU (us) | GPU (us) |
|---------------|------------------|------------------|
| cos |1134.67  |  |
| sin |1101.52  |  |
| exp |919.59  |  |
| gather |1415.52  |  |
| scatter |45130.37  |  |
| index_add |412.25  |  |
| softmax |108.09  |  |
| topk |4073.73  |  |

mocked_model op

| Operation Name | HPU (us) | GPU (us) |
|---------------|------------------|------------------|
| rotary_embedding |513.27  |  |
| moe_quant_group_gemm |785.89  |  |
| moe_scatter_dynamic_quant |6517.79  |  |
| moe_swiglu_dynamic_quant |3383.04  |  |
| head_rms_norm |1050.74  |  |