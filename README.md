```
/home/zhangran/miniconda3/envs/fusionrl/lib/python3.12/site-packages/accelerate/accelerator.py:530: UserWarning: `log_with=tensorboard` was passed but no supported trackers are currently installed.
  warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
[Config] epochs=10, lr=0.0001, kl_w=1e-05, loss_scale=0.1, mp=bf16
[Dirs] project_dir=./checkpoints/stochastic_policy_stage1_final
[2025-08-31 18:15:57,895] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-08-31 18:15:59,416] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
Epoch 1/10: 100%|███████████████████| 42/42 [00:35<00:00,  1.18it/s, loss=0.4753, fusion=4.75, kld=0.1608]
[Epoch 1] avg_total=0.7121  avg_fusion=7.1209  avg_kld=0.147189

[Eval] Epoch 1 - MSRS
[Metrics][MSRS] VIF=0.8824  Qabf=0.6198  SSIM=0.9586  Reward=0.9236  PSNR=65.5590  MSE=0.0286  CC=0.6191  SCD=1.5449  Nabf=0.0008  MI=2.7922  AG=3.0629  EN=6.3896  SF=5.2420  SD=39.0000

[Eval] Epoch 1 - M3FD
[Metrics][M3FD] VIF=0.7232  Qabf=0.4995  SSIM=0.9291  Reward=0.8005  PSNR=63.1718  MSE=0.0379  CC=0.5066  SCD=1.4525  Nabf=0.0007  MI=3.0842  AG=3.3081  EN=6.8503  SF=5.4299  SD=32.8278

[Eval] Epoch 1 - RS
[Metrics][RS] VIF=0.7326  Qabf=0.5275  SSIM=0.9898  Reward=0.8379  PSNR=63.0035  MSE=0.0419  CC=0.5903  SCD=1.3845  Nabf=0.0007  MI=3.6215  AG=3.4917  EN=7.1783  SF=6.0840  SD=42.6440
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_1
Epoch 2/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.5856, fusion=5.86, kld=0.1557]
[Epoch 2] avg_total=0.5589  avg_fusion=5.5886  avg_kld=0.160992

[Eval] Epoch 2 - MSRS
[Metrics][MSRS] VIF=0.9970  Qabf=0.6741  SSIM=0.9788  Reward=0.9957  PSNR=65.0135  MSE=0.0330  CC=0.6085  SCD=1.5829  Nabf=0.0016  MI=3.5302  AG=3.3728  EN=6.5480  SF=5.5959  SD=40.6482

[Eval] Epoch 2 - M3FD
[Metrics][M3FD] VIF=0.8275  Qabf=0.6052  SSIM=0.9463  Reward=0.8939  PSNR=62.5337  MSE=0.0453  CC=0.4502  SCD=1.2791  Nabf=0.0018  MI=3.4918  AG=3.9252  EN=6.9165  SF=6.0096  SD=33.5842

[Eval] Epoch 2 - RS
[Metrics][RS] VIF=0.7874  Qabf=0.5813  SSIM=0.9768  Reward=0.8787  PSNR=63.0572  MSE=0.0406  CC=0.5734  SCD=1.2640  Nabf=0.0013  MI=3.5384  AG=3.8825  EN=7.1944  SF=6.4281  SD=41.7060
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_2
```

```
home/zhangran/miniconda3/envs/fusionrl/lib/python3.12/site-packages/accelerate/accelerator.py:530: UserWarning: `log_with=tensorboard` was passed but no supported trackers are currently installed.
  warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
[Config] epochs=10, lr=0.0001, kl_w=1e-05, loss_scale=0.1, mp=bf16
[Dirs] project_dir=./checkpoints/stochastic_policy_stage1_final
[2025-08-31 20:35:25,173] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-08-31 20:35:26,887] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
Epoch 1/10: 100%|███████████████████| 42/42 [00:35<00:00,  1.18it/s, loss=0.5583, fusion=5.58, kld=0.6057]
[Epoch 1] avg_total=0.7985  avg_fusion=7.9846  avg_kld=0.595729

[Eval] Epoch 1 - MSRS
[Metrics][MSRS] VIF=0.8584  Qabf=0.5865  SSIM=0.9331  Reward=0.8904  PSNR=65.8885  MSE=0.0260  CC=0.6307  SCD=1.5125  Nabf=0.0008  MI=2.8174  AG=2.9237  EN=6.3043  SF=5.1251  SD=36.8726

[Eval] Epoch 1 - M3FD
[Metrics][M3FD] VIF=0.7707  Qabf=0.4267  SSIM=0.9132  Reward=0.7747  PSNR=62.9618  MSE=0.0391  CC=0.4712  SCD=1.3585  Nabf=0.0004  MI=3.6246  AG=2.7058  EN=6.8513  SF=4.8549  SD=34.6209

[Eval] Epoch 1 - RS
[Metrics][RS] VIF=0.8548  Qabf=0.6007  SSIM=0.9688  Reward=0.9082  PSNR=62.8741  MSE=0.0415  CC=0.5662  SCD=1.3087  Nabf=0.0009  MI=4.5672  AG=4.0687  EN=7.2710  SF=6.5879  SD=45.0463

[Eval] Epoch 1 - PET
[Metrics][PET] VIF=0.8485  Qabf=0.7552  SSIM=1.2233  Reward=1.0682  PSNR=61.0127  MSE=0.0529  CC=0.7712  SCD=1.3206  Nabf=0.0002  MI=3.7030  AG=10.4576  EN=5.4527  SF=9.1516  SD=83.1282

[Eval] Epoch 1 - SPECT
[Metrics][SPECT] VIF=0.8841  Qabf=0.7496  SSIM=1.2575  Reward=1.0887  PSNR=67.2210  MSE=0.0146  CC=0.8596  SCD=1.0322  Nabf=0.0004  MI=3.7195  AG=5.5645  EN=4.5500  SF=7.9038  SD=55.5084
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_1
Epoch 2/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.5349, fusion=5.35, kld=0.6424]
[Epoch 2] avg_total=0.6478  avg_fusion=6.4783  avg_kld=0.686132

[Eval] Epoch 2 - MSRS
[Metrics][MSRS] VIF=0.9391  Qabf=0.6539  SSIM=0.9677  Reward=0.9625  PSNR=65.4244  MSE=0.0293  CC=0.6185  SCD=1.5599  Nabf=0.0013  MI=3.0450  AG=3.2441  EN=6.4671  SF=5.4669  SD=38.7732

[Eval] Epoch 2 - M3FD
[Metrics][M3FD] VIF=0.8374  Qabf=0.5171  SSIM=0.9125  Reward=0.8418  PSNR=62.4563  MSE=0.0439  CC=0.4243  SCD=1.2319  Nabf=0.0015  MI=4.0208  AG=3.2287  EN=6.9741  SF=5.3584  SD=36.0515

[Eval] Epoch 2 - RS
[Metrics][RS] VIF=0.9212  Qabf=0.6262  SSIM=0.9535  Reward=0.9380  PSNR=62.6778  MSE=0.0418  CC=0.5489  SCD=1.2462  Nabf=0.0020  MI=5.3866  AG=4.3746  EN=7.3364  SF=6.8262  SD=46.3510

[Eval] Epoch 2 - PET
[Metrics][PET] VIF=0.8805  Qabf=0.7534  SSIM=1.2472  Reward=1.0859  PSNR=60.8945  MSE=0.0542  CC=0.7824  SCD=1.3824  Nabf=0.0008  MI=3.7086  AG=10.4389  EN=5.3245  SF=8.8806  SD=85.4587

[Eval] Epoch 2 - SPECT
[Metrics][SPECT] VIF=0.9468  Qabf=0.7615  SSIM=1.2527  Reward=1.1139  PSNR=66.9618  MSE=0.0154  CC=0.8530  SCD=1.0156  Nabf=0.0010  MI=4.0306  AG=5.8229  EN=4.4733  SF=8.0259  SD=56.0402
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_2
Epoch 3/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.5894, fusion=5.89, kld=0.6138]
[Epoch 3] avg_total=0.6207  avg_fusion=6.2071  avg_kld=0.660683

[Eval] Epoch 3 - MSRS
[Metrics][MSRS] VIF=0.9595  Qabf=0.6643  SSIM=0.9680  Reward=0.9746  PSNR=65.3555  MSE=0.0299  CC=0.6156  SCD=1.5636  Nabf=0.0015  MI=3.2094  AG=3.2815  EN=6.4840  SF=5.5240  SD=39.1640

[Eval] Epoch 3 - M3FD
[Metrics][M3FD] VIF=0.8134  Qabf=0.5564  SSIM=0.9125  Reward=0.8535  PSNR=62.4790  MSE=0.0436  CC=0.4292  SCD=1.2517  Nabf=0.0017  MI=3.7140  AG=3.5336  EN=7.0121  SF=5.6183  SD=36.6869

[Eval] Epoch 3 - RS
[Metrics][RS] VIF=0.9002  Qabf=0.6245  SSIM=0.9601  Reward=0.9324  PSNR=62.6643  MSE=0.0424  CC=0.5542  SCD=1.2831  Nabf=0.0023  MI=4.9085  AG=4.3596  EN=7.3371  SF=6.8186  SD=46.5727

[Eval] Epoch 3 - PET
[Metrics][PET] VIF=0.9213  Qabf=0.7496  SSIM=1.1992  Reward=1.0816  PSNR=61.0211  MSE=0.0527  CC=0.7815  SCD=1.2705  Nabf=0.0006  MI=3.5446  AG=10.2676  EN=5.4765  SF=8.7205  SD=84.1210

[Eval] Epoch 3 - SPECT
[Metrics][SPECT] VIF=0.9192  Qabf=0.7496  SSIM=1.2577  Reward=1.1004  PSNR=67.0189  MSE=0.0152  CC=0.8576  SCD=1.0732  Nabf=0.0013  MI=3.8844  AG=5.7497  EN=4.4753  SF=7.9915  SD=56.3892
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_3
Epoch 4/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.5240, fusion=5.24, kld=0.5435]
[Epoch 4] avg_total=0.6047  avg_fusion=6.0470  avg_kld=0.586965

[Eval] Epoch 4 - MSRS
[Metrics][MSRS] VIF=0.9735  Qabf=0.6705  SSIM=0.9716  Reward=0.9836  PSNR=65.2127  MSE=0.0311  CC=0.6165  SCD=1.5907  Nabf=0.0020  MI=3.3513  AG=3.3289  EN=6.5128  SF=5.5721  SD=39.8145

[Eval] Epoch 4 - M3FD
[Metrics][M3FD] VIF=0.8372  Qabf=0.5968  SSIM=0.9327  Reward=0.8884  PSNR=62.3765  MSE=0.0456  CC=0.4445  SCD=1.2989  Nabf=0.0019  MI=3.8155  AG=3.7832  EN=7.0014  SF=5.9005  SD=36.1944

[Eval] Epoch 4 - RS
[Metrics][RS] VIF=0.8808  Qabf=0.6184  SSIM=0.9716  Reward=0.9267  PSNR=62.5561  MSE=0.0446  CC=0.5651  SCD=1.3303  Nabf=0.0025  MI=4.5759  AG=4.3069  EN=7.3249  SF=6.7856  SD=46.8190

[Eval] Epoch 4 - PET
[Metrics][PET] VIF=0.9047  Qabf=0.7536  SSIM=1.2662  Reward=1.1004  PSNR=60.8844  MSE=0.0543  CC=0.7781  SCD=1.3317  Nabf=0.0017  MI=3.8183  AG=10.5437  EN=5.2112  SF=8.8732  SD=85.0822

[Eval] Epoch 4 - SPECT
[Metrics][SPECT] VIF=0.9726  Qabf=0.7649  SSIM=1.2647  Reward=1.1282  PSNR=66.8659  MSE=0.0156  CC=0.8511  SCD=0.9856  Nabf=0.0022  MI=4.1434  AG=5.9132  EN=4.2619  SF=8.0624  SD=56.4619
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_4
Epoch 5/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.6643, fusion=6.64, kld=0.4982]
[Epoch 5] avg_total=0.5879  avg_fusion=5.8793  avg_kld=0.530225

[Eval] Epoch 5 - MSRS
[Metrics][MSRS] VIF=0.9648  Qabf=0.6692  SSIM=0.9602  Reward=0.9763  PSNR=65.2982  MSE=0.0301  CC=0.6144  SCD=1.5511  Nabf=0.0023  MI=3.2312  AG=3.3373  EN=6.4874  SF=5.5829  SD=39.2377

[Eval] Epoch 5 - M3FD
[Metrics][M3FD] VIF=0.8334  Qabf=0.6073  SSIM=0.9125  Reward=0.8856  PSNR=62.5613  MSE=0.0434  CC=0.4237  SCD=1.2333  Nabf=0.0021  MI=3.5237  AG=3.8797  EN=7.0166  SF=5.9733  SD=36.4837

[Eval] Epoch 5 - RS
[Metrics][RS] VIF=0.8649  Qabf=0.6295  SSIM=0.9620  Reward=0.9237  PSNR=62.7995  MSE=0.0410  CC=0.5434  SCD=1.2320  Nabf=0.0029  MI=4.3516  AG=4.4620  EN=7.3675  SF=6.9106  SD=46.3953

[Eval] Epoch 5 - PET
[Metrics][PET] VIF=0.8907  Qabf=0.7525  SSIM=1.2633  Reward=1.0943  PSNR=60.9724  MSE=0.0533  CC=0.7746  SCD=1.2395  Nabf=0.0013  MI=3.7040  AG=10.5192  EN=5.2359  SF=8.9082  SD=83.8055

[Eval] Epoch 5 - SPECT
[Metrics][SPECT] VIF=0.9971  Qabf=0.7753  SSIM=1.2651  Reward=1.1417  PSNR=66.8226  MSE=0.0157  CC=0.8463  SCD=0.8607  Nabf=0.0019  MI=4.2641  AG=5.9750  EN=4.2607  SF=8.1102  SD=55.8063
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_5
Epoch 6/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.4849, fusion=4.85, kld=0.4369]
[Epoch 6] avg_total=0.5732  avg_fusion=5.7315  avg_kld=0.475427

[Eval] Epoch 6 - MSRS
[Metrics][MSRS] VIF=0.9739  Qabf=0.6737  SSIM=0.9671  Reward=0.9838  PSNR=65.1686  MSE=0.0315  CC=0.6122  SCD=1.5702  Nabf=0.0024  MI=3.2571  AG=3.3728  EN=6.5094  SF=5.6171  SD=40.0154

[Eval] Epoch 6 - M3FD
[Metrics][M3FD] VIF=0.8491  Qabf=0.6350  SSIM=0.9191  Reward=0.9069  PSNR=62.6263  MSE=0.0436  CC=0.4369  SCD=1.2628  Nabf=0.0024  MI=3.2767  AG=4.0742  EN=7.0060  SF=6.1982  SD=35.0261

[Eval] Epoch 6 - RS
[Metrics][RS] VIF=0.7904  Qabf=0.5992  SSIM=0.9593  Reward=0.8828  PSNR=63.1570  MSE=0.0388  CC=0.5676  SCD=1.2636  Nabf=0.0034  MI=3.2259  AG=4.2447  EN=7.2902  SF=6.6963  SD=42.8338

[Eval] Epoch 6 - PET
[Metrics][PET] VIF=0.9125  Qabf=0.7578  SSIM=1.2536  Reward=1.1009  PSNR=60.9233  MSE=0.0539  CC=0.7730  SCD=1.2442  Nabf=0.0021  MI=3.7789  AG=10.5990  EN=5.2246  SF=8.8712  SD=84.1587

[Eval] Epoch 6 - SPECT
[Metrics][SPECT] VIF=0.9850  Qabf=0.7708  SSIM=1.2590  Reward=1.1334  PSNR=66.8365  MSE=0.0157  CC=0.8487  SCD=0.9150  Nabf=0.0023  MI=4.1907  AG=5.9618  EN=4.2775  SF=8.0967  SD=56.1228
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_6
Epoch 7/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.4067, fusion=4.07, kld=0.3982]
[Epoch 7] avg_total=0.5609  avg_fusion=5.6091  avg_kld=0.431021

[Eval] Epoch 7 - MSRS
[Metrics][MSRS] VIF=0.9734  Qabf=0.6727  SSIM=0.9688  Reward=0.9838  PSNR=65.1376  MSE=0.0317  CC=0.6114  SCD=1.5640  Nabf=0.0025  MI=3.2319  AG=3.3791  EN=6.5379  SF=5.6363  SD=39.8723

[Eval] Epoch 7 - M3FD
[Metrics][M3FD] VIF=0.8442  Qabf=0.5971  SSIM=0.9010  Reward=0.8803  PSNR=62.4433  MSE=0.0441  CC=0.4318  SCD=1.2620  Nabf=0.0031  MI=3.6196  AG=3.8418  EN=7.0404  SF=5.9573  SD=36.5489

[Eval] Epoch 7 - RS
[Metrics][RS] VIF=0.9294  Qabf=0.6452  SSIM=0.9492  Reward=0.9488  PSNR=62.5704  MSE=0.0428  CC=0.5405  SCD=1.2234  Nabf=0.0036  MI=5.4986  AG=4.5950  EN=7.3894  SF=7.0067  SD=47.5631

[Eval] Epoch 7 - PET
[Metrics][PET] VIF=0.9676  Qabf=0.8029  SSIM=1.2541  Reward=1.1420  PSNR=60.8118  MSE=0.0552  CC=0.7466  SCD=1.1156  Nabf=0.0012  MI=4.5448  AG=11.0242  EN=5.0495  SF=9.2161  SD=82.2682

[Eval] Epoch 7 - SPECT
[Metrics][SPECT] VIF=1.0249  Qabf=0.7810  SSIM=1.2584  Reward=1.1516  PSNR=66.7407  MSE=0.0159  CC=0.8422  SCD=0.6563  Nabf=0.0020  MI=4.4265  AG=6.0305  EN=4.1449  SF=8.1464  SD=55.5802
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_7
Epoch 8/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.5666, fusion=5.67, kld=0.3798]
[Epoch 8] avg_total=0.5482  avg_fusion=5.4823  avg_kld=0.396837

[Eval] Epoch 8 - MSRS
[Metrics][MSRS] VIF=0.9564  Qabf=0.6677  SSIM=0.9514  Reward=0.9698  PSNR=65.0678  MSE=0.0324  CC=0.6095  SCD=1.5662  Nabf=0.0026  MI=3.2809  AG=3.3715  EN=6.5449  SF=5.6443  SD=40.1755

[Eval] Epoch 8 - M3FD
[Metrics][M3FD] VIF=0.8335  Qabf=0.6074  SSIM=0.8936  Reward=0.8794  PSNR=62.4255  MSE=0.0450  CC=0.4313  SCD=1.2448  Nabf=0.0031  MI=3.4515  AG=3.9266  EN=7.0447  SF=6.0312  SD=36.3653

[Eval] Epoch 8 - RS
[Metrics][RS] VIF=0.9179  Qabf=0.6439  SSIM=0.9477  Reward=0.9438  PSNR=62.5463  MSE=0.0435  CC=0.5362  SCD=1.2001  Nabf=0.0035  MI=5.2023  AG=4.5856  EN=7.3866  SF=6.9987  SD=47.8541

[Eval] Epoch 8 - PET
[Metrics][PET] VIF=0.9160  Qabf=0.7915  SSIM=1.2581  Reward=1.1204  PSNR=60.8741  MSE=0.0545  CC=0.7544  SCD=1.1767  Nabf=0.0011  MI=4.1850  AG=10.9048  EN=5.1230  SF=9.2139  SD=82.5783

[Eval] Epoch 8 - SPECT
[Metrics][SPECT] VIF=1.0247  Qabf=0.7809  SSIM=1.2565  Reward=1.1509  PSNR=66.7498  MSE=0.0159  CC=0.8424  SCD=0.6744  Nabf=0.0018  MI=4.4488  AG=6.0273  EN=4.1898  SF=8.1449  SD=55.5686
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_8
Epoch 9/10: 100%|███████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.6103, fusion=6.10, kld=0.3547]
[Epoch 9] avg_total=0.5232  avg_fusion=5.2324  avg_kld=0.366805

[Eval] Epoch 9 - MSRS
[Metrics][MSRS] VIF=0.9690  Qabf=0.6733  SSIM=0.9372  Reward=0.9720  PSNR=64.9969  MSE=0.0327  CC=0.6077  SCD=1.5513  Nabf=0.0031  MI=3.4667  AG=3.4152  EN=6.5715  SF=5.7146  SD=40.1293

[Eval] Epoch 9 - M3FD
[Metrics][M3FD] VIF=0.8270  Qabf=0.5898  SSIM=0.8505  Reward=0.8541  PSNR=62.6403  MSE=0.0415  CC=0.4465  SCD=1.2887  Nabf=0.0035  MI=3.3193  AG=3.8589  EN=7.0034  SF=5.9480  SD=35.3375

[Eval] Epoch 9 - RS
[Metrics][RS] VIF=0.9246  Qabf=0.6479  SSIM=0.9440  Reward=0.9468  PSNR=62.6659  MSE=0.0413  CC=0.5367  SCD=1.2080  Nabf=0.0035  MI=5.4747  AG=4.6360  EN=7.3882  SF=7.0489  SD=47.3881

[Eval] Epoch 9 - PET
[Metrics][PET] VIF=0.9768  Qabf=0.8059  SSIM=1.2547  Reward=1.1468  PSNR=60.8056  MSE=0.0553  CC=0.7431  SCD=0.9960  Nabf=0.0008  MI=4.5740  AG=11.0579  EN=5.0448  SF=9.2329  SD=81.8829

[Eval] Epoch 9 - SPECT
[Metrics][SPECT] VIF=1.0286  Qabf=0.7815  SSIM=1.2613  Reward=1.1540  PSNR=66.7291  MSE=0.0159  CC=0.8413  SCD=0.6352  Nabf=0.0019  MI=4.4566  AG=6.0294  EN=4.1344  SF=8.1458  SD=55.5055
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_9
Epoch 10/10: 100%|██████████████████| 42/42 [00:34<00:00,  1.22it/s, loss=0.5703, fusion=5.70, kld=0.3303]
[Epoch 10] avg_total=0.5091  avg_fusion=5.0907  avg_kld=0.342537

[Eval] Epoch 10 - MSRS
[Metrics][MSRS] VIF=0.9924  Qabf=0.6811  SSIM=0.9438  Reward=0.9860  PSNR=64.7559  MSE=0.0350  CC=0.6059  SCD=1.5800  Nabf=0.0033  MI=3.7568  AG=3.4529  EN=6.6097  SF=5.7461  SD=41.2658

[Eval] Epoch 10 - M3FD
[Metrics][M3FD] VIF=0.8591  Qabf=0.6242  SSIM=0.8228  Reward=0.8727  PSNR=62.0704  MSE=0.0489  CC=0.4412  SCD=1.2609  Nabf=0.0035  MI=3.5075  AG=4.0926  EN=7.0485  SF=6.3000  SD=35.5291

[Eval] Epoch 10 - RS
[Metrics][RS] VIF=0.9063  Qabf=0.6454  SSIM=0.9446  Reward=0.9397  PSNR=62.4494  MSE=0.0450  CC=0.5423  SCD=1.2280  Nabf=0.0048  MI=5.0895  AG=4.6287  EN=7.3795  SF=7.0576  SD=47.5844

[Eval] Epoch 10 - PET
[Metrics][PET] VIF=0.9094  Qabf=0.7661  SSIM=1.2624  Reward=1.1070  PSNR=60.8666  MSE=0.0546  CC=0.7683  SCD=1.2591  Nabf=0.0022  MI=3.9850  AG=10.7359  EN=5.1788  SF=9.0187  SD=84.2321

[Eval] Epoch 10 - SPECT
[Metrics][SPECT] VIF=1.0225  Qabf=0.7803  SSIM=1.2602  Reward=1.1510  PSNR=66.7458  MSE=0.0159  CC=0.8433  SCD=0.6858  Nabf=0.0024  MI=4.3893  AG=6.0306  EN=4.1270  SF=8.1447  SD=55.7451
[Save] model -> ./checkpoints/stochastic_policy_stage1_final/epoch_10
[Final] model -> ./checkpoints/stochastic_policy_stage1_final/final
```