## Pooled over the ten core cells (paired difference in nDG)

| architecture | comparison | quota | before | after |
|---|---|---|---|---|
| MLP | rank(V) - V | 10% | -0.082 [-0.154, -0.003] | -0.110 [-0.189, -0.027] |
| MLP | rank(V) - V | 20% | -0.127 [-0.198, -0.014] | -0.170 [-0.244, -0.048] |
| MLP | rank(V) - V | 30% | -0.139 [-0.208, +0.008] | -0.115 [-0.210, -0.010] **(interval call changes)** |
| MLP | rank(V) - V | 50% | -0.102 [-0.198, +0.038] | -0.102 [-0.194, +0.006] |
| MLP | signed ECDF(V) - V | 10% | -0.056 [-0.109, +0.016] | -0.043 [-0.096, +0.026] |
| MLP | signed ECDF(V) - V | 20% | -0.076 [-0.140, +0.020] | -0.065 [-0.147, +0.051] |
| MLP | signed ECDF(V) - V | 30% | -0.027 [-0.098, +0.069] | -0.003 [-0.105, +0.088] |
| MLP | signed ECDF(V) - V | 50% | +0.037 [-0.035, +0.123] | +0.029 [-0.056, +0.116] |
| MLP | Q - V | 10% | -0.076 [-0.153, +0.000] | -0.071 [-0.148, +0.004] |
| MLP | Q - rank(V) | 10% | +0.006 [-0.058, +0.055] | +0.039 [-0.031, +0.096] |
| MLP | Q - signed ECDF(V) | 10% | -0.020 [-0.088, +0.028] | -0.029 [-0.096, +0.028] |
| MLP | Q - V | 20% | -0.135 [-0.226, -0.022] | -0.139 [-0.244, -0.021] |
| MLP | Q - rank(V) | 20% | -0.008 [-0.101, +0.054] | +0.032 [-0.058, +0.095] |
| MLP | Q - signed ECDF(V) | 20% | -0.059 [-0.157, +0.020] | -0.074 [-0.167, +0.004] |
| MLP | Q - V | 30% | -0.123 [-0.229, +0.008] | -0.104 [-0.237, +0.016] |
| MLP | Q - rank(V) | 30% | +0.017 [-0.110, +0.088] | +0.011 [-0.084, +0.093] |
| MLP | Q - signed ECDF(V) | 30% | -0.096 [-0.193, +0.002] | -0.100 [-0.206, -0.008] **(interval call changes)** |
| MLP | Q - V | 50% | -0.114 [-0.223, +0.009] | -0.123 [-0.228, -0.016] **(interval call changes)** |
| MLP | Q - rank(V) | 50% | -0.012 [-0.138, +0.090] | -0.021 [-0.133, +0.084] |
| MLP | Q - signed ECDF(V) | 50% | -0.151 [-0.263, -0.049] | -0.152 [-0.266, -0.031] |
| MLP | G - V | 10% | -0.063 [-0.120, +0.029] | -0.058 [-0.111, +0.030] |
| MLP | G - rank(V) | 10% | +0.020 [-0.034, +0.097] | +0.052 [-0.000, +0.131] |
| MLP | G - signed ECDF(V) | 10% | -0.007 [-0.062, +0.065] | -0.016 [-0.065, +0.063] |
| MLP | G - V | 20% | -0.027 [-0.118, +0.071] | -0.030 [-0.121, +0.070] |
| MLP | G - rank(V) | 20% | +0.100 [+0.002, +0.181] | +0.140 [+0.032, +0.213] |
| MLP | G - signed ECDF(V) | 20% | +0.049 [-0.058, +0.132] | +0.035 [-0.080, +0.125] |
| MLP | G - V | 30% | -0.066 [-0.131, +0.067] | -0.045 [-0.127, +0.092] |
| MLP | G - rank(V) | 30% | +0.074 [-0.042, +0.186] | +0.070 [-0.012, +0.178] |
| MLP | G - signed ECDF(V) | 30% | -0.039 [-0.123, +0.078] | -0.042 [-0.131, +0.090] |
| MLP | G - V | 50% | -0.047 [-0.146, +0.090] | -0.056 [-0.137, +0.070] |
| MLP | G - rank(V) | 50% | +0.055 [-0.045, +0.167] | +0.047 [-0.046, +0.155] |
| MLP | G - signed ECDF(V) | 50% | -0.084 [-0.181, +0.034] | -0.084 [-0.161, +0.040] |
| GBM | rank(V) - V | 10% | -0.028 [-0.116, +0.082] | -0.001 [-0.100, +0.102] |
| GBM | rank(V) - V | 20% | -0.029 [-0.123, +0.103] | -0.014 [-0.120, +0.128] |
| GBM | rank(V) - V | 30% | -0.002 [-0.113, +0.117] | +0.012 [-0.116, +0.147] |
| GBM | rank(V) - V | 50% | -0.003 [-0.118, +0.119] | -0.020 [-0.114, +0.137] |
| GBM | signed ECDF(V) - V | 10% | +0.063 [-0.021, +0.114] | +0.063 [-0.007, +0.125] |
| GBM | signed ECDF(V) - V | 20% | +0.023 [-0.033, +0.114] | +0.053 [-0.008, +0.138] |
| GBM | signed ECDF(V) - V | 30% | +0.048 [-0.042, +0.109] | +0.059 [-0.037, +0.137] |
| GBM | signed ECDF(V) - V | 50% | +0.036 [-0.037, +0.104] | +0.001 [-0.055, +0.108] |
| GBM | Q - V | 10% | +0.043 [-0.056, +0.098] | +0.045 [-0.055, +0.102] |
| GBM | Q - rank(V) | 10% | +0.071 [-0.064, +0.127] | +0.047 [-0.082, +0.115] |
| GBM | Q - signed ECDF(V) | 10% | -0.020 [-0.111, +0.045] | -0.018 [-0.120, +0.039] |
| GBM | Q - V | 20% | -0.025 [-0.116, +0.106] | -0.007 [-0.101, +0.103] |
| GBM | Q - rank(V) | 20% | +0.003 [-0.137, +0.124] | +0.007 [-0.149, +0.126] |
| GBM | Q - signed ECDF(V) | 20% | -0.048 [-0.151, +0.045] | -0.060 [-0.175, +0.027] |
| GBM | Q - V | 30% | +0.000 [-0.117, +0.135] | -0.002 [-0.111, +0.131] |
| GBM | Q - rank(V) | 30% | +0.002 [-0.137, +0.147] | -0.013 [-0.169, +0.143] |
| GBM | Q - signed ECDF(V) | 30% | -0.047 [-0.149, +0.089] | -0.061 [-0.163, +0.076] |
| GBM | Q - V | 50% | +0.048 [-0.040, +0.158] | +0.025 [-0.054, +0.168] |
| GBM | Q - rank(V) | 50% | +0.051 [-0.072, +0.204] | +0.045 [-0.088, +0.198] |
| GBM | Q - signed ECDF(V) | 50% | +0.012 [-0.070, +0.132] | +0.024 [-0.068, +0.130] |
| GBM | G - V | 10% | +0.064 [-0.011, +0.164] | +0.066 [-0.012, +0.165] |
| GBM | G - rank(V) | 10% | +0.092 [-0.011, +0.182] | +0.067 [-0.029, +0.171] |
| GBM | G - signed ECDF(V) | 10% | +0.001 [-0.059, +0.106] | +0.003 [-0.079, +0.102] |
| GBM | G - V | 20% | +0.088 [-0.018, +0.205] | +0.106 [-0.012, +0.214] |
| GBM | G - rank(V) | 20% | +0.116 [+0.005, +0.215] | +0.120 [-0.005, +0.219] **(interval call changes)** |
| GBM | G - signed ECDF(V) | 20% | +0.065 [-0.041, +0.148] | +0.053 [-0.058, +0.128] |
| GBM | G - V | 30% | +0.072 [-0.045, +0.193] | +0.069 [-0.050, +0.192] |
| GBM | G - rank(V) | 30% | +0.075 [-0.035, +0.185] | +0.058 [-0.060, +0.180] |
| GBM | G - signed ECDF(V) | 30% | +0.025 [-0.067, +0.140] | +0.010 [-0.081, +0.127] |
| GBM | G - V | 50% | -0.008 [-0.105, +0.155] | -0.032 [-0.123, +0.156] |
| GBM | G - rank(V) | 50% | -0.005 [-0.116, +0.140] | -0.012 [-0.133, +0.146] |
| GBM | G - signed ECDF(V) | 50% | -0.044 [-0.137, +0.119] | -0.033 [-0.135, +0.115] |
| MLP | binary Q>0 - V>0 | 10% | -0.043 [-0.112, +0.043] | -0.072 [-0.145, +0.031] |
| MLP | binary Q>0 - V>0 | 20% | -0.051 [-0.140, +0.092] | -0.067 [-0.167, +0.091] |
| MLP | binary Q>0 - V>0 | 30% | -0.076 [-0.162, +0.118] | -0.079 [-0.175, +0.127] |
| MLP | binary Q>0 - V>0 | 50% | -0.059 [-0.172, +0.107] | -0.053 [-0.198, +0.125] |
| MLP | binary G>0 - V>0 | 10% | -0.072 [-0.134, +0.017] | -0.099 [-0.175, +0.003] |
| MLP | binary G>0 - V>0 | 20% | -0.086 [-0.172, +0.063] | -0.101 [-0.193, +0.068] |
| MLP | binary G>0 - V>0 | 30% | -0.107 [-0.191, +0.074] | -0.108 [-0.204, +0.086] |
| MLP | binary G>0 - V>0 | 50% | -0.001 [-0.125, +0.159] | +0.007 [-0.135, +0.169] |
| GBM | binary Q>0 - V>0 | 10% | +0.033 [-0.053, +0.091] | +0.004 [-0.072, +0.078] |
| GBM | binary Q>0 - V>0 | 20% | +0.024 [-0.046, +0.120] | -0.032 [-0.096, +0.086] |
| GBM | binary Q>0 - V>0 | 30% | -0.003 [-0.075, +0.127] | -0.035 [-0.103, +0.103] |
| GBM | binary Q>0 - V>0 | 50% | +0.003 [-0.104, +0.102] | -0.006 [-0.101, +0.107] |
| GBM | binary G>0 - V>0 | 10% | +0.012 [-0.081, +0.085] | -0.017 [-0.095, +0.064] |
| GBM | binary G>0 - V>0 | 20% | -0.078 [-0.150, +0.063] | -0.135 [-0.194, +0.028] |
| GBM | binary G>0 - V>0 | 30% | -0.114 [-0.186, +0.059] | -0.149 [-0.213, +0.033] |
| GBM | binary G>0 - V>0 | 50% | -0.097 [-0.170, +0.049] | -0.108 [-0.184, +0.067] |

## Per-cell intervals against raw V, above / below zero, over the 40 cell-quota pairs

| architecture | label | before | after |
|---|---|---|---|
| GBM | G | 9 / 1 | 9 / 0 |
| GBM | Q | 9 / 2 | 9 / 3 |
| GBM | rank(V) | 3 / 1 | 3 / 1 |
| GBM | signed ECDF(V) | 5 / 0 | 5 / 0 |
| MLP | G | 1 / 0 | 0 / 0 |
| MLP | Q | 0 / 5 | 0 / 4 |
| MLP | rank(V) | 0 / 4 | 0 / 8 |
| MLP | signed ECDF(V) | 0 / 2 | 0 / 1 |

## GBM router at 20 %, per cell (paired difference to raw V)

| cell | label | before | after |
|---|---|---|---|
| KITTI mono brake | G | -0.059 [-0.231, +0.105] | -0.057 [-0.228, +0.108] |
| KITTI mono traj | G | +0.133 [-0.417, +0.812] | +0.128 [-0.433, +0.905] |
| KITTI oracle brake | G | -0.167 [-0.359, +0.184] | -0.164 [-0.366, +0.183] |
| KITTI oracle traj | G | -0.143 [-0.717, +0.121] | -0.158 [-0.716, +0.106] |
| nuScenes mono brake | G | -0.115 [-0.280, +0.157] | -0.116 [-0.309, +0.152] |
| nuScenes mono plan_ade | G | +0.294 [+0.104, +0.498] | +0.294 [+0.104, +0.498] |
| nuScenes mono plan_fde | G | +0.189 [+0.029, +0.476] | +0.189 [+0.029, +0.476] |
| nuScenes oracle brake | G | +0.054 [-0.326, +0.594] | +0.252 [-0.284, +0.614] |
| nuScenes oracle plan_ade | G | +0.277 [+0.014, +0.462] | +0.277 [+0.014, +0.462] |
| nuScenes oracle plan_fde | G | +0.415 [+0.058, +0.779] | +0.415 [+0.058, +0.779] |
| KITTI mono brake | Q | -0.168 [-0.350, +0.041] | -0.166 [-0.338, +0.050] |
| KITTI mono traj | Q | -0.001 [-0.337, +0.579] | +0.000 [-0.346, +0.614] |
| KITTI oracle brake | Q | -0.219 [-0.462, -0.003] | -0.216 [-0.466, -0.008] |
| KITTI oracle traj | Q | -0.246 [-0.885, +0.210] | -0.261 [-0.879, +0.194] |
| nuScenes mono brake | Q | -0.271 [-0.397, +0.034] | -0.273 [-0.431, +0.030] |
| nuScenes mono plan_ade | Q | +0.283 [-0.007, +0.454] | +0.283 [-0.007, +0.454] |
| nuScenes mono plan_fde | Q | +0.253 [+0.053, +0.412] | +0.253 [+0.053, +0.412] |
| nuScenes oracle brake | Q | -0.301 [-0.684, +0.327] | -0.107 [-0.596, +0.315] |
| nuScenes oracle plan_ade | Q | +0.157 [+0.043, +0.301] | +0.157 [+0.043, +0.301] |
| nuScenes oracle plan_fde | Q | +0.261 [-0.002, +0.564] | +0.261 [-0.002, +0.564] |
| KITTI mono brake | rank(V) | +0.019 [-0.044, +0.130] | +0.007 [-0.074, +0.090] |
| KITTI mono traj | rank(V) | -0.551 [-1.285, +0.376] | -0.685 [-1.337, +0.472] |
| KITTI oracle brake | rank(V) | -0.005 [-0.075, +0.102] | -0.018 [-0.129, +0.087] |
| KITTI oracle traj | rank(V) | -0.059 [-0.200, +0.015] | -0.079 [-0.240, -0.005] |
| nuScenes mono brake | rank(V) | -0.122 [-0.491, +0.120] | -0.117 [-0.480, +0.135] |
| nuScenes mono plan_ade | rank(V) | +0.063 [-0.054, +0.290] | +0.063 [-0.054, +0.290] |
| nuScenes mono plan_fde | rank(V) | +0.170 [+0.041, +0.377] | +0.170 [+0.041, +0.377] |
| nuScenes oracle brake | rank(V) | -0.190 [-0.419, +0.101] | +0.134 [-0.301, +0.326] |
| nuScenes oracle plan_ade | rank(V) | +0.173 [-0.041, +0.446] | +0.173 [-0.041, +0.446] |
| nuScenes oracle plan_fde | rank(V) | +0.216 [-0.059, +0.587] | +0.216 [-0.059, +0.587] |
| KITTI mono brake | signed ECDF(V) | -0.008 [-0.035, +0.092] | -0.010 [-0.023, +0.086] |
| KITTI mono traj | signed ECDF(V) | -0.195 [-0.597, +0.356] | -0.076 [-0.523, +0.458] |
| KITTI oracle brake | signed ECDF(V) | -0.007 [-0.009, +0.180] | -0.007 [-0.016, +0.103] |
| KITTI oracle traj | signed ECDF(V) | +0.030 [-0.033, +0.118] | -0.029 [-0.098, +0.083] |
| nuScenes mono brake | signed ECDF(V) | -0.007 [-0.327, +0.209] | +0.042 [-0.233, +0.232] |
| nuScenes mono plan_ade | signed ECDF(V) | +0.122 [+0.003, +0.213] | +0.122 [+0.003, +0.213] |
| nuScenes mono plan_fde | signed ECDF(V) | +0.216 [+0.048, +0.434] | +0.216 [+0.048, +0.434] |
| nuScenes oracle brake | signed ECDF(V) | -0.215 [-0.389, +0.048] | -0.022 [-0.408, +0.069] |
| nuScenes oracle plan_ade | signed ECDF(V) | +0.093 [-0.036, +0.214] | +0.093 [-0.036, +0.214] |
| nuScenes oracle plan_fde | signed ECDF(V) | +0.197 [-0.033, +0.472] | +0.197 [-0.033, +0.472] |

## Seed variation at 20 % (nDG min / median / max over six seeds; winning seeds)

| cell | signal | before | wins | after | wins | shipped win before → after | flag before → after |
|---|---|---|---|---|---|---|---|
| KITTI mono brake | R1_gbm_clf | 0.197 / 0.197 / 0.197 | 6.0 | 0.215 / 0.215 / 0.215 | 6.0 | True → True | False → False |
| KITTI mono brake | R1_gbm_reg | 0.204 / 0.204 / 0.204 | 6.0 | 0.201 / 0.201 / 0.201 | 6.0 | True → True | False → False |
| KITTI mono brake | R1_mlp_clf | 0.151 / 0.162 / 0.185 | 0.0 | 0.148 / 0.174 / 0.195 | 3.0 | False → True | False → True |
| KITTI mono brake | R1_mlp_reg | 0.150 / 0.169 / 0.192 | 1.0 | 0.159 / 0.178 / 0.191 | 2.0 | False → True | False → True |
| KITTI mono brake | gate_gbm | 0.154 / 0.154 / 0.154 | 0.0 | 0.158 / 0.158 / 0.158 | 0.0 | False → False | False → False |
| KITTI mono traj | R1_gbm_clf | -0.192 / -0.192 / -0.192 | 0.0 | 0.059 / 0.059 / 0.059 | 0.0 | False → False | False → False |
| KITTI mono traj | R1_gbm_reg | 0.126 / 0.126 / 0.126 | 0.0 | 0.125 / 0.125 / 0.125 | 0.0 | False → False | False → False |
| KITTI mono traj | R1_mlp_clf | -0.186 / 0.091 / 0.194 | 0.0 | -0.006 / 0.061 / 0.132 | 0.0 | False → False | False → False |
| KITTI mono traj | R1_mlp_reg | 0.041 / 0.117 / 0.236 | 0.0 | 0.029 / 0.106 / 0.186 | 0.0 | False → False | False → False |
| KITTI mono traj | gate_gbm | 0.062 / 0.062 / 0.062 | 0.0 | 0.184 / 0.184 / 0.184 | 0.0 | False → False | False → False |
| KITTI oracle brake | R1_gbm_clf | 0.324 / 0.324 / 0.324 | 6.0 | 0.360 / 0.360 / 0.360 | 6.0 | True → True | False → False |
| KITTI oracle brake | R1_gbm_reg | 0.295 / 0.295 / 0.295 | 0.0 | 0.293 / 0.293 / 0.293 | 0.0 | False → False | False → False |
| KITTI oracle brake | R1_mlp_clf | 0.253 / 0.288 / 0.295 | 5.0 | 0.240 / 0.299 / 0.317 | 2.0 | True → True | False → True |
| KITTI oracle brake | R1_mlp_reg | 0.217 / 0.255 / 0.284 | 2.0 | 0.219 / 0.257 / 0.264 | 2.0 | False → False | False → False |
| KITTI oracle brake | gate_gbm | 0.207 / 0.207 / 0.207 | 0.0 | 0.226 / 0.226 / 0.226 | 0.0 | False → False | False → False |
| KITTI oracle traj | R1_gbm_clf | 0.453 / 0.453 / 0.453 | 6.0 | 0.484 / 0.484 / 0.484 | 6.0 | True → True | False → False |
| KITTI oracle traj | R1_gbm_reg | 0.501 / 0.501 / 0.501 | 6.0 | 0.516 / 0.516 / 0.516 | 6.0 | True → True | False → False |
| KITTI oracle traj | R1_mlp_clf | 0.356 / 0.393 / 0.419 | 6.0 | 0.361 / 0.425 / 0.449 | 6.0 | True → True | False → False |
| KITTI oracle traj | R1_mlp_reg | 0.321 / 0.382 / 0.460 | 3.0 | 0.356 / 0.391 / 0.424 | 4.0 | False → False | False → False |
| KITTI oracle traj | gate_gbm | 0.572 / 0.572 / 0.572 | 6.0 | 0.538 / 0.538 / 0.538 | 6.0 | True → True | False → False |
| nuScenes mono brake | R1_gbm_clf | 0.327 / 0.327 / 0.327 | 0.0 | 0.369 / 0.369 / 0.369 | 0.0 | False → False | False → False |
| nuScenes mono brake | R1_gbm_reg | 0.380 / 0.380 / 0.380 | 6.0 | 0.383 / 0.383 / 0.383 | 6.0 | True → True | False → False |
| nuScenes mono brake | R1_mlp_clf | 0.248 / 0.289 / 0.393 | 1.0 | 0.222 / 0.258 / 0.464 | 1.0 | False → False | False → False |
| nuScenes mono brake | R1_mlp_reg | 0.202 / 0.277 / 0.473 | 3.0 | 0.145 / 0.286 / 0.366 | 2.0 | True → True | True → True |
| nuScenes mono brake | gate_gbm | 0.397 / 0.397 / 0.397 | 6.0 | 0.482 / 0.482 / 0.482 | 6.0 | True → True | False → False |
| nuScenes mono plan_ade | R1_gbm_clf | 0.090 / 0.090 / 0.090 | 0.0 | 0.090 / 0.090 / 0.090 | 0.0 | False → False | False → False |
| nuScenes mono plan_ade | R1_gbm_reg | -0.022 / -0.022 / -0.022 | 0.0 | -0.022 / -0.022 / -0.022 | 0.0 | False → False | False → False |
| nuScenes mono plan_ade | R1_mlp_clf | 0.034 / 0.108 / 0.150 | 0.0 | 0.034 / 0.108 / 0.150 | 0.0 | False → False | False → False |
| nuScenes mono plan_ade | R1_mlp_reg | 0.106 / 0.251 / 0.298 | 0.0 | 0.106 / 0.251 / 0.298 | 0.0 | False → False | False → False |
| nuScenes mono plan_ade | gate_gbm | 0.042 / 0.042 / 0.042 | 0.0 | 0.042 / 0.042 / 0.042 | 0.0 | False → False | False → False |
| nuScenes mono plan_fde | R1_gbm_clf | 0.131 / 0.131 / 0.131 | 0.0 | 0.131 / 0.131 / 0.131 | 0.0 | False → False | False → False |
| nuScenes mono plan_fde | R1_gbm_reg | -0.062 / -0.062 / -0.062 | 0.0 | -0.062 / -0.062 / -0.062 | 0.0 | False → False | False → False |
| nuScenes mono plan_fde | R1_mlp_clf | -0.027 / 0.027 / 0.141 | 0.0 | -0.027 / 0.027 / 0.141 | 0.0 | False → False | False → False |
| nuScenes mono plan_fde | R1_mlp_reg | 0.085 / 0.210 / 0.260 | 0.0 | 0.085 / 0.210 / 0.260 | 0.0 | False → False | False → False |
| nuScenes mono plan_fde | gate_gbm | 0.079 / 0.079 / 0.079 | 0.0 | 0.079 / 0.079 / 0.079 | 0.0 | False → False | False → False |
| nuScenes oracle brake | R1_gbm_clf | 0.247 / 0.247 / 0.247 | 0.0 | 0.421 / 0.421 / 0.421 | 0.0 | False → False | False → False |
| nuScenes oracle brake | R1_gbm_reg | 0.394 / 0.394 / 0.394 | 0.0 | 0.196 / 0.196 / 0.196 | 0.0 | False → False | False → False |
| nuScenes oracle brake | R1_mlp_clf | 0.001 / 0.164 / 0.245 | 0.0 | 0.077 / 0.285 / 0.354 | 0.0 | False → False | False → False |
| nuScenes oracle brake | R1_mlp_reg | 0.091 / 0.217 / 0.335 | 0.0 | 0.130 / 0.186 / 0.354 | 0.0 | False → False | False → False |
| nuScenes oracle brake | gate_gbm | 0.169 / 0.169 / 0.169 | 0.0 | 0.155 / 0.155 / 0.155 | 0.0 | False → False | False → False |
| nuScenes oracle plan_ade | R1_gbm_clf | 0.031 / 0.031 / 0.031 | 0.0 | 0.031 / 0.031 / 0.031 | 0.0 | False → False | False → False |
| nuScenes oracle plan_ade | R1_gbm_reg | -0.083 / -0.083 / -0.083 | 0.0 | -0.083 / -0.083 / -0.083 | 0.0 | False → False | False → False |
| nuScenes oracle plan_ade | R1_mlp_clf | -0.137 / -0.070 / 0.043 | 0.0 | -0.137 / -0.070 / 0.043 | 0.0 | False → False | False → False |
| nuScenes oracle plan_ade | R1_mlp_reg | -0.120 / -0.058 / 0.064 | 0.0 | -0.120 / -0.058 / 0.064 | 0.0 | False → False | False → False |
| nuScenes oracle plan_ade | gate_gbm | -0.141 / -0.141 / -0.141 | 0.0 | -0.141 / -0.141 / -0.141 | 0.0 | False → False | False → False |
| nuScenes oracle plan_fde | R1_gbm_clf | 0.065 / 0.065 / 0.065 | 0.0 | 0.065 / 0.065 / 0.065 | 0.0 | False → False | False → False |
| nuScenes oracle plan_fde | R1_gbm_reg | -0.093 / -0.093 / -0.093 | 0.0 | -0.093 / -0.093 / -0.093 | 0.0 | False → False | False → False |
| nuScenes oracle plan_fde | R1_mlp_clf | -0.047 / 0.056 / 0.164 | 0.0 | -0.047 / 0.056 / 0.164 | 0.0 | False → False | False → False |
| nuScenes oracle plan_fde | R1_mlp_reg | -0.055 / 0.004 / 0.045 | 0.0 | -0.055 / 0.004 / 0.045 | 0.0 | False → False | False → False |
| nuScenes oracle plan_fde | gate_gbm | 0.125 / 0.125 / 0.125 | 0.0 | 0.125 / 0.125 / 0.125 | 0.0 | False → False | False → False |
