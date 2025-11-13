# Operation Performance Comparison

basic op

| Operation Name | HPU (us) | GPU (us) |
|---------------|------------------|------------------|
| cos |1134.67  |802.59  |
| sin |1101.52  |803.36  |
| exp |919.59  |803.30  |
| gather |1415.52  |49.89  |
| scatter |45130.37  |46.50  |
| index_add |412.25  |193.57  |
| softmax |108.09  |93.60  |
| topk |4073.73  |184.29  |

mocked_model op

| Operation Name | HPU (us) | GPU (us) |
|---------------|------------------|------------------|
| rotary_embedding |513.27  |372.77  |
| moe_quant_group_gemm |785.89  |983.55  |
| moe_scatter_dynamic_quant |6517.79  |1287.46  |
| moe_swiglu_dynamic_quant |3383.04  |1048.03  |
| head_rms_norm |1050.74  |521.25  |