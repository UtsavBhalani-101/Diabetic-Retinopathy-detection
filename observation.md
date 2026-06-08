2026-06-08 11:19:37 | INFO     | pipeline.orchestrators.train | --- Epoch 10/10 ---
2026-06-08 11:22:04 | INFO     | pipeline.orchestrators.train | Epoch 10 | Train Loss: 0.6123
2026-06-08 11:22:41 | INFO     | pipeline.evaluation.evaluate | evaluate() | loss=0.8089 | QWK=0.8928
2026-06-08 11:22:41 | INFO     | pipeline.orchestrators.train | Epoch 10 | Val Loss: 0.8089 | Val QWK: 0.8928
2026-06-08 11:22:41 | INFO     | pipeline.orchestrators.train | Confusion Matrix:
[[345  13   3   0   0]
 [  4  48  21   1   0]
 [  0  30 124  38   8]
 [  0   0   6  25   8]
 [  0   5   9  15  30]]
2026-06-08 11:22:41 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | T=10 passes | starting...
2026-06-08 11:23:19 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | done | N=733 samples | mean std=0.0107
2026-06-08 11:23:19 | INFO     | pipeline.orchestrators.train | Epoch 10 | MC uncertainty | entropy=0.5096 | margin=0.6684 | uncertain_frac=0.256
2026-06-08 11:23:19 | INFO     | pipeline.orchestrators.train | Training loop complete.
2026-06-08 11:23:19 | INFO     | pipeline.orchestrators.train | Post-training MC Dropout | T=30
2026-06-08 11:23:19 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | T=30 passes | starting...
2026-06-08 11:24:00 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | done | N=733 samples | mean std=0.0113
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train | Applying temperature calibration...
2026-06-08 11:24:00 | INFO     | pipeline.evaluation.calibration | find_temperature() | searching optimal T in [0.1, 10.0]...
2026-06-08 11:24:00 | INFO     | pipeline.evaluation.calibration | find_temperature() | optimal T = 0.9262
2026-06-08 11:24:00 | INFO     | pipeline.evaluation.calibration | find_temperature() | ECE before scaling (T=1.0): 0.0465
2026-06-08 11:24:00 | INFO     | pipeline.evaluation.calibration | find_temperature() | ECE after  scaling (T=0.93): 0.0453
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train | Triage Summary — post-calibration (first 20 samples):
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   0 | True:0 Pred:0 | H=0.061 M=0.982 MC=0.002 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   1 | True:0 Pred:0 | H=0.097 M=0.976 MC=0.002 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   2 | True:4 Pred:3 | H=0.978 M=0.419 MC=0.022 | HIGH SEVERITY - urgent review
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   3 | True:0 Pred:0 | H=0.014 M=0.998 MC=0.000 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   4 | True:0 Pred:0 | H=0.028 M=0.993 MC=0.001 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   5 | True:0 Pred:0 | H=0.695 M=0.696 MC=0.014 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   6 | True:2 Pred:3 | H=0.651 M=0.486 MC=0.025 | HIGH SEVERITY - urgent review
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   7 | True:3 Pred:3 | H=0.283 M=0.892 MC=0.007 | HIGH SEVERITY - urgent review
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   8 | True:2 Pred:3 | H=1.119 M=0.182 MC=0.017 | UNCERTAIN - refer to specialist
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample   9 | True:1 Pred:2 | H=0.945 M=0.350 MC=0.021 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  10 | True:0 Pred:0 | H=0.005 M=0.999 MC=0.000 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  11 | True:0 Pred:0 | H=0.245 M=0.885 MC=0.012 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  12 | True:0 Pred:0 | H=0.002 M=1.000 MC=0.000 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  13 | True:1 Pred:0 | H=1.483 M=0.035 MC=0.018 | UNCERTAIN - refer to specialist
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  14 | True:0 Pred:0 | H=0.004 M=0.999 MC=0.000 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  15 | True:0 Pred:0 | H=0.656 M=0.712 MC=0.013 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  16 | True:0 Pred:0 | H=0.006 M=0.999 MC=0.000 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  17 | True:0 Pred:0 | H=0.010 M=0.998 MC=0.000 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  18 | True:0 Pred:0 | H=0.001 M=1.000 MC=0.000 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train |   Sample  19 | True:0 Pred:0 | H=0.001 M=1.000 MC=0.000 | ROUTINE
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train | Uncertain fraction (calibrated): 0.237
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train | Calibrated Mean entropy        : 0.4714
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train | Calibrated Mean margin         : 0.6890
2026-06-08 11:24:00 | INFO     | pipeline.orchestrators.train | Calibrated Mean MC std         : 0.0113
2026-06-08 11:24:01 | INFO     | pipeline.orchestrators.train | --- Four Quadrant Uncertainty Breakdown ---
2026-06-08 11:24:01 | INFO     | pipeline.orchestrators.train |   Certain + Wrong (dangerous): 79
2026-06-08 11:24:01 | INFO     | pipeline.orchestrators.train |   Certain + Right (ideal)     : 480
2026-06-08 11:24:01 | INFO     | pipeline.orchestrators.train |   Uncertain + Wrong (caught)  : 82
2026-06-08 11:24:01 | INFO     | pipeline.orchestrators.train |   Uncertain + Right (over-ref): 92
2026-06-08 11:24:01 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_train.png
2026-06-08 11:24:01 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.0285
2026-06-08 11:24:01 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.0388
2026-06-08 11:24:01 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.0809
2026-06-08 11:24:01 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.0588
2026-06-08 11:24:01 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.0195
2026-06-08 11:24:02 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_train.png
2026-06-08 11:24:02 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_train_20260608_112402.png
2026-06-08 11:24:02 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.0285
2026-06-08 11:24:02 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.0388
2026-06-08 11:24:02 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.0809
2026-06-08 11:24:02 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.0588
2026-06-08 11:24:02 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.0195
2026-06-08 11:24:03 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_train_20260608_112402.png
2026-06-08 11:24:03 | INFO     | pipeline.orchestrators.train | Final calibrated val QWK: 0.8928
2026-06-08 11:24:03 | INFO     | pipeline.orchestrators.train | Model saved     → artifacts/weights/aptos_efficientnet.pth (Backup: artifacts/weights/aptos_efficientnet_20260608_112402.pth)
2026-06-08 11:24:03 | INFO     | pipeline.orchestrators.train | Optimal T saved → artifacts/calibration/optimal_T.npy (Backup: artifacts/calibration/optimal_T_20260608_112402.npy) (T=0.9262)
2026-06-08 11:24:03 | INFO     | pipeline.orchestrators.train | Computing per-class reference distributions from val set (for Cosine Similarity)...
2026-06-08 11:24:39 | INFO     | pipeline.orchestrators.train |   Class 0 (No DR): 361 samples, mean norm=12.1414
2026-06-08 11:24:39 | INFO     | pipeline.orchestrators.train |   Class 1 (Mild): 74 samples, mean norm=9.8687
2026-06-08 11:24:39 | INFO     | pipeline.orchestrators.train |   Class 2 (Moderate): 200 samples, mean norm=9.1087
2026-06-08 11:24:39 | INFO     | pipeline.orchestrators.train |   Class 3 (Severe): 39 samples, mean norm=11.4878
2026-06-08 11:24:39 | INFO     | pipeline.orchestrators.train |   Class 4 (Proliferative): 59 samples, mean norm=10.4545
2026-06-08 11:24:39 | INFO     | pipeline.orchestrators.train | Per-class mean (centroids) saved → artifacts/centroids/mean.npy
wandb: ⢿ updating run metadata (0.0s)
wandb: ⣻ updating run metadata (0.0s)
wandb: ⣽ updating run metadata (0.0s)
wandb: ⣾ updating run metadata (0.0s)
wandb: ⣷ uploading summary, console lines 189-194 (0.0s)
wandb: 
wandb: Run history:
wandb:                     epoch ▁▂▃▃▄▅▆▆▇█
wandb:        final_mean_entropy ▁
wandb:         final_mean_margin ▁
wandb: final_mean_mc_uncertainty ▁
wandb:  final_uncertain_fraction ▁
wandb:             final_val_qwk ▁
wandb:             learning_rate ▁▁▁▁▁▁▁▁▁▁
wandb:              mean_entropy █▄▃▂▂▂▁▁▁▁
wandb:               mean_margin ▁▅▆▆▇▇███▇
wandb:       mean_mc_uncertainty █▄▃▂▂▂▁▁▁▁
wandb:                        +9 ...
wandb: 
wandb: Run summary:
wandb:                     epoch 10
wandb:        final_mean_entropy 0.47141
wandb:         final_mean_margin 0.68897
wandb: final_mean_mc_uncertainty 0.01126
wandb:  final_uncertain_fraction 0.23738
wandb:             final_val_qwk 0.89276
wandb:             learning_rate 0.0001
wandb:              mean_entropy 0.50959
wandb:               mean_margin 0.66844
wandb:       mean_mc_uncertainty 0.01069
wandb:                        +9 ...
wandb: 
wandb: 🚀 View run vibrant-jazz-45 at: https://wandb.ai/nova9-/aptos-blindness-detection/runs/w3kw2md2
wandb: ⭐️ View project at: https://wandb.ai/nova9-/aptos-blindness-detection
wandb: Synced 5 W&B file(s), 11 media file(s), 20 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260608_104552-w3kw2md2/logs
2026-06-08 11:24:40 | INFO     | pipeline.orchestrators.train | train_model() complete.
2026-06-08 11:24:40 | INFO     | __main__ | Training complete | optimal_T=0.9262
2026-06-08 11:24:40 | INFO     | __main__ | Starting zero-shot evaluation on external datasets...
2026-06-08 11:24:40 | INFO     | __main__ | ============================================================
2026-06-08 11:24:40 | INFO     | __main__ | STEP 2: Testing on IDRiD | use_test_split=True
2026-06-08 11:24:40 | INFO     | __main__ | ============================================================
2026-06-08 11:24:40 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:24:40 | INFO     | pipeline.orchestrators.test | test_model() | dataset=IDRiD | T=0.9262
2026-06-08 11:24:40 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:24:40 | INFO     | pipeline.setup.config | Device selected: cuda
2026-06-08 11:24:40 | INFO     | pipeline.setup.config | GPU: Tesla T4
2026-06-08 11:24:40 | INFO     | pipeline.training_loop_setup.model | EfficientNetMC | classes=5 | dropout=0.3 | pretrained=False | features=1280
2026-06-08 11:24:40 | INFO     | pipeline.orchestrators.test | [IDRiD] Model loaded from artifacts/weights/aptos_efficientnet.pth
wandb: ⢿ Waiting for wandb.init()...
wandb: ⣻ Waiting for wandb.init()...
wandb: ⣽ Waiting for wandb.init()...
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /kaggle/working/Diabetic-Retinopathy-detection/wandb/run-20260608_112440-zfxvlihk
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run smart-dew-46
wandb: ⭐️ View project at https://wandb.ai/nova9-/aptos-blindness-detection
wandb: 🚀 View run at https://wandb.ai/nova9-/aptos-blindness-detection/runs/zfxvlihk
2026-06-08 11:24:41 | INFO     | pipeline.orchestrators.test | [IDRiD] wandb run initialised | job=test
2026-06-08 11:24:41 | INFO     | pipeline.data.loaders | [IDRiD] Using official TEST split for evaluation
2026-06-08 11:24:41 | INFO     | pipeline.data.dataset | RetinopathyDataset | rows=103 | ext=.jpg
2026-06-08 11:24:41 | INFO     | pipeline.data.loaders | [IDRiD] Test loader ready | samples=103 | batch=32
2026-06-08 11:24:41 | INFO     | pipeline.orchestrators.test | [IDRiD] MC Dropout | T=30
2026-06-08 11:24:41 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | T=30 passes | starting...
2026-06-08 11:24:52 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | done | N=103 samples | mean std=0.0189
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test | [IDRiD] Applying temperature T=0.9262
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test | [IDRiD] QWK: 0.5102
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test | [IDRiD] Confusion Matrix:
[[ 3 25  2  2  2]
 [ 1  4  0  0  0]
 [ 0 10 14  4  4]
 [ 0  2  6 11  0]
 [ 0  1  4  5  3]]
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test | [IDRiD] Mean entropy      : 0.9721
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test | [IDRiD] Mean margin       : 0.3693
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test | [IDRiD] Uncertain fraction: 0.4951
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test | [IDRiD] Triage Summary (first 20 samples):
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   0 | True:4 Pred:3 | H=0.645 M=0.493 MC=0.021 | HIGH SEVERITY - urgent review
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   1 | True:4 Pred:1 | H=0.667 M=0.660 MC=0.016 | ROUTINE
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   2 | True:4 Pred:2 | H=1.087 M=0.138 MC=0.020 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   3 | True:4 Pred:4 | H=1.339 M=0.186 MC=0.025 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   4 | True:4 Pred:4 | H=0.704 M=0.538 MC=0.018 | HIGH SEVERITY - urgent review
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   5 | True:3 Pred:3 | H=1.229 M=0.129 MC=0.022 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   6 | True:3 Pred:3 | H=1.273 M=0.069 MC=0.016 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   7 | True:2 Pred:4 | H=1.181 M=0.008 MC=0.024 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   8 | True:2 Pred:2 | H=0.898 M=0.440 MC=0.023 | ROUTINE
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample   9 | True:2 Pred:3 | H=0.328 M=0.874 MC=0.012 | HIGH SEVERITY - urgent review
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  10 | True:2 Pred:2 | H=1.158 M=0.423 MC=0.019 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  11 | True:2 Pred:1 | H=0.933 M=0.539 MC=0.017 | ROUTINE
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  12 | True:3 Pred:3 | H=0.812 M=0.563 MC=0.015 | HIGH SEVERITY - urgent review
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  13 | True:3 Pred:2 | H=1.130 M=0.189 MC=0.021 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  14 | True:2 Pred:3 | H=1.167 M=0.149 MC=0.022 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  15 | True:3 Pred:3 | H=0.614 M=0.731 MC=0.014 | HIGH SEVERITY - urgent review
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  16 | True:2 Pred:4 | H=1.102 M=0.307 MC=0.016 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  17 | True:3 Pred:3 | H=0.507 M=0.754 MC=0.014 | HIGH SEVERITY - urgent review
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  18 | True:2 Pred:2 | H=1.038 M=0.325 MC=0.016 | UNCERTAIN - refer to specialist
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Sample  19 | True:2 Pred:2 | H=0.934 M=0.565 MC=0.018 | ROUTINE
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test | [IDRiD] --- Four Quadrant Uncertainty Breakdown ---
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Certain + Wrong (dangerous): 25
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Certain + Right (ideal)     : 19
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Uncertain + Wrong (caught)  : 43
2026-06-08 11:24:52 | INFO     | pipeline.orchestrators.test |   Uncertain + Right (over-ref): 16
2026-06-08 11:24:52 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_IDRiD.png
2026-06-08 11:24:52 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.2640
2026-06-08 11:24:52 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.2801
2026-06-08 11:24:52 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.0874
2026-06-08 11:24:52 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.0635
2026-06-08 11:24:52 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.0305
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_IDRiD.png
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_IDRiD_20260608_112453.png
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.2640
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.2801
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.0874
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.0635
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.0305
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_IDRiD_20260608_112453.png
2026-06-08 11:24:53 | INFO     | pipeline.evaluation.cosine_similarity | Extracting features from test set...
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Extracted features for 103 samples, feature dim = 1280
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 0 (No DR)...
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 1 (Mild)...
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 2 (Moderate)...
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 3 (Severe)...
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 4 (Proliferative)...

--- Per-Class Cosine Similarity Summary ---
  [No DR]  min=-0.2466  max=0.5702  avg=0.1001
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity |   Class 0 (No DR) | min=-0.2466  max=0.5702  avg=0.1001
  [Mild]  min=-0.0087  max=0.7218  avg=0.4058
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity |   Class 1 (Mild) | min=-0.0087  max=0.7218  avg=0.4058
  [Moderate]  min=0.0753  max=0.7617  avg=0.4397
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity |   Class 2 (Moderate) | min=0.0753  max=0.7617  avg=0.4397
  [Severe]  min=-0.0604  max=0.8138  avg=0.3253
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity |   Class 3 (Severe) | min=-0.0604  max=0.8138  avg=0.3253
  [Proliferative]  min=-0.0507  max=0.7925  avg=0.3421
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity |   Class 4 (Proliferative) | min=-0.0507  max=0.7925  avg=0.3421

  [Global]  min=-0.2466  max=0.8138  avg=0.3226
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 0 (No DR) similarities → artifacts/cosine_similarity/IDRiD/similarities_class_0.npy
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 1 (Mild) similarities → artifacts/cosine_similarity/IDRiD/similarities_class_1.npy
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 2 (Moderate) similarities → artifacts/cosine_similarity/IDRiD/similarities_class_2.npy
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 3 (Severe) similarities → artifacts/cosine_similarity/IDRiD/similarities_class_3.npy
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 4 (Proliferative) similarities → artifacts/cosine_similarity/IDRiD/similarities_class_4.npy
2026-06-08 11:25:01 | INFO     | pipeline.evaluation.cosine_similarity | Saved test labels → artifacts/cosine_similarity/IDRiD/labels.npy
wandb: ⢿ updating run metadata (0.2s)
wandb: ⢿ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.0s)
wandb: ⣻ updating run metadata (0.2s)
wandb: ⣻ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.0s)
wandb: ⣽ updating run metadata (0.2s)
wandb: ⣽ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.0s)
wandb: ⣾ updating run metadata (0.2s)
wandb: ⣾ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.0s)
wandb: ⣷ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.5s)
wandb: ⣯ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.5s)
wandb: ⣟ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.5s)
wandb: ⡿ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.5s)
wandb: ⢿ uploading artifact run-zfxvlihk-test_confusion_matrix_table (0.5s)
wandb: ⣻ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.0s)
wandb: ⣽ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.0s)
wandb: ⣾ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.0s)
wandb: ⣷ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.0s)
wandb: ⣯ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.0s)
wandb: ⣟ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.5s)
wandb: ⡿ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.5s)
wandb: ⢿ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.5s)
wandb: ⣻ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.5s)
wandb: ⣽ uploading artifact run-zfxvlihk-test_confusion_matrix_table (1.5s)
wandb: ⣾ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.0s)
wandb: ⣷ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.0s)
wandb: ⣯ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.0s)
wandb: ⣟ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.0s)
wandb: ⡿ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.0s)
wandb: ⢿ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.5s)
wandb: ⣻ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.5s)
wandb: ⣽ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.5s)
wandb: ⣾ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.5s)
wandb: ⣷ uploading artifact run-zfxvlihk-test_confusion_matrix_table (2.5s)
wandb: ⣯ uploading media/images/test_calibration_plot_1_b086ef323d861490c787....
wandb: ⣯ uploading media/table/test_confusion_matrix_table_1_6a88ff5a89d7b6a9...
wandb: ⣯ uploading output.log 9.2KB/9.2KB (0.1s)
wandb: ⣯ uploading wandb-summary.json 2.0KB/2.0KB (0.1s)
wandb: ⣯ uploading config.yaml 3.8KB/3.8KB (0.1s)
wandb: ⣟ uploading media/images/test_calibration_plot_1_b086ef323d861490c787....
wandb: ⣟ uploading media/table/test_confusion_matrix_table_1_6a88ff5a89d7b6a9...
wandb: ⣟ uploading output.log 9.2KB/9.2KB (0.1s)
wandb: ⣟ uploading wandb-summary.json 2.0KB/2.0KB (0.1s)
wandb: ⣟ uploading config.yaml 3.8KB/3.8KB (0.1s)
wandb: ⡿ uploading media/images/test_calibration_plot_1_b086ef323d861490c787....
wandb: ⡿ uploading media/table/test_confusion_matrix_table_1_6a88ff5a89d7b6a9...
wandb: ⡿ uploading output.log 9.2KB/9.2KB (0.1s)
wandb: ⡿ uploading wandb-summary.json 2.0KB/2.0KB (0.1s)
wandb: ⡿ uploading config.yaml 3.8KB/3.8KB (0.1s)
wandb: ⢿ uploading media/images/test_calibration_plot_1_b086ef323d861490c787....
wandb: ⢿ uploading media/table/test_confusion_matrix_table_1_6a88ff5a89d7b6a9...
wandb: ⢿ uploading output.log 9.2KB/9.2KB (0.1s)
wandb: ⢿ uploading wandb-summary.json 2.0KB/2.0KB (0.1s)
wandb: ⢿ uploading config.yaml 3.8KB/3.8KB (0.1s)
wandb: ⣻ uploading media/images/test_calibration_plot_1_b086ef323d861490c787....
wandb: ⣻ uploading media/table/test_confusion_matrix_table_1_6a88ff5a89d7b6a9...
wandb: ⣻ uploading output.log 9.2KB/9.2KB (0.1s)
wandb: ⣻ uploading wandb-summary.json 2.0KB/2.0KB (0.1s)
wandb: ⣻ uploading config.yaml 3.8KB/3.8KB (0.1s)
wandb: 
wandb: Run history:
wandb:   CosineSimilarity/Global/Avg ▁
wandb:   CosineSimilarity/Global/Max ▁
wandb:   CosineSimilarity/Global/Min ▁
wandb:     CosineSimilarity/Mild/Avg ▁
wandb:     CosineSimilarity/Mild/Max ▁
wandb:     CosineSimilarity/Mild/Min ▁
wandb: CosineSimilarity/Moderate/Avg ▁
wandb: CosineSimilarity/Moderate/Max ▁
wandb: CosineSimilarity/Moderate/Min ▁
wandb:    CosineSimilarity/No DR/Avg ▁
wandb:                           +17 ...
wandb: 
wandb: Run summary:
wandb:   CosineSimilarity/Global/Avg 0.3226
wandb:   CosineSimilarity/Global/Max 0.8138
wandb:   CosineSimilarity/Global/Min -0.2466
wandb:     CosineSimilarity/Mild/Avg 0.4058
wandb:     CosineSimilarity/Mild/Max 0.7218
wandb:     CosineSimilarity/Mild/Min -0.0087
wandb: CosineSimilarity/Moderate/Avg 0.4397
wandb: CosineSimilarity/Moderate/Max 0.7617
wandb: CosineSimilarity/Moderate/Min 0.0753
wandb:    CosineSimilarity/No DR/Avg 0.1001
wandb:                           +17 ...
wandb: 
wandb: 🚀 View run smart-dew-46 at: https://wandb.ai/nova9-/aptos-blindness-detection/runs/zfxvlihk
wandb: ⭐️ View project at: https://wandb.ai/nova9-/aptos-blindness-detection
wandb: Synced 5 W&B file(s), 2 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260608_112440-zfxvlihk/logs
2026-06-08 11:25:04 | INFO     | pipeline.orchestrators.test | [IDRiD] test_model() complete.
2026-06-08 11:25:04 | INFO     | __main__ | ============================================================
2026-06-08 11:25:04 | INFO     | __main__ | STEP 2: Testing on DDR-China | use_test_split=False
2026-06-08 11:25:04 | INFO     | __main__ | ============================================================
2026-06-08 11:25:04 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:25:04 | INFO     | pipeline.orchestrators.test | test_model() | dataset=DDR-China | T=0.9262
2026-06-08 11:25:04 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:25:04 | INFO     | pipeline.setup.config | Device selected: cuda
2026-06-08 11:25:04 | INFO     | pipeline.setup.config | GPU: Tesla T4
2026-06-08 11:25:04 | INFO     | pipeline.training_loop_setup.model | EfficientNetMC | classes=5 | dropout=0.3 | pretrained=False | features=1280
2026-06-08 11:25:04 | INFO     | pipeline.orchestrators.test | [DDR-China] Model loaded from artifacts/weights/aptos_efficientnet.pth
wandb: ⢿ Waiting for wandb.init()...
wandb: ⣻ Waiting for wandb.init()...
wandb: ⣽ Waiting for wandb.init()...
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /kaggle/working/Diabetic-Retinopathy-detection/wandb/run-20260608_112505-ivelpxxi
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run swept-paper-47
wandb: ⭐️ View project at https://wandb.ai/nova9-/aptos-blindness-detection
wandb: 🚀 View run at https://wandb.ai/nova9-/aptos-blindness-detection/runs/ivelpxxi
2026-06-08 11:25:06 | INFO     | pipeline.orchestrators.test | [DDR-China] wandb run initialised | job=test
2026-06-08 11:25:06 | INFO     | pipeline.data.loaders | [DDR-China] Using full training CSV for evaluation (zero-shot)
2026-06-08 11:25:06 | INFO     | pipeline.data.dataset | RetinopathyDataset | rows=2000 | ext=
2026-06-08 11:25:06 | INFO     | pipeline.data.loaders | [DDR-China] Test loader ready | samples=2000 | batch=32
2026-06-08 11:25:06 | INFO     | pipeline.orchestrators.test | [DDR-China] MC Dropout | T=30
2026-06-08 11:25:06 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | T=30 passes | starting...
2026-06-08 11:26:23 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | done | N=2000 samples | mean std=0.0105
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test | [DDR-China] Applying temperature T=0.9262
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test | [DDR-China] QWK: 0.5760
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test | [DDR-China] Confusion Matrix:
[[966  18  11   2   4]
 [ 88   8   3   0   1]
 [392  92 122  51  58]
 [  2   2   7  19   8]
 [ 29   4  19  19  75]]
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test | [DDR-China] Mean entropy      : 0.4634
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test | [DDR-China] Mean margin       : 0.7212
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test | [DDR-China] Uncertain fraction: 0.2120
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test | [DDR-China] Triage Summary (first 20 samples):
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   0 | True:0 Pred:0 | H=0.082 M=0.973 MC=0.002 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   1 | True:0 Pred:0 | H=0.008 M=0.999 MC=0.000 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   2 | True:2 Pred:0 | H=1.315 M=0.134 MC=0.019 | UNCERTAIN - refer to specialist
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   3 | True:2 Pred:0 | H=0.102 M=0.978 MC=0.003 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   4 | True:0 Pred:0 | H=0.003 M=1.000 MC=0.000 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   5 | True:1 Pred:0 | H=0.621 M=0.622 MC=0.015 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   6 | True:2 Pred:4 | H=1.048 M=0.253 MC=0.027 | UNCERTAIN - refer to specialist
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   7 | True:1 Pred:0 | H=0.208 M=0.925 MC=0.004 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   8 | True:4 Pred:0 | H=1.125 M=0.460 MC=0.025 | UNCERTAIN - refer to specialist
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample   9 | True:0 Pred:0 | H=0.053 M=0.985 MC=0.002 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  10 | True:0 Pred:2 | H=1.384 M=0.098 MC=0.020 | UNCERTAIN - refer to specialist
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  11 | True:2 Pred:2 | H=0.942 M=0.590 MC=0.019 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  12 | True:0 Pred:0 | H=0.011 M=0.997 MC=0.000 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  13 | True:0 Pred:0 | H=0.150 M=0.943 MC=0.005 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  14 | True:0 Pred:0 | H=0.001 M=1.000 MC=0.000 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  15 | True:0 Pred:0 | H=0.000 M=1.000 MC=0.000 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  16 | True:2 Pred:0 | H=1.008 M=0.386 MC=0.022 | UNCERTAIN - refer to specialist
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  17 | True:0 Pred:0 | H=0.000 M=1.000 MC=0.000 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  18 | True:0 Pred:0 | H=0.261 M=0.876 MC=0.010 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Sample  19 | True:0 Pred:0 | H=0.008 M=0.998 MC=0.000 | ROUTINE
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test | [DDR-China] --- Four Quadrant Uncertainty Breakdown ---
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Certain + Wrong (dangerous): 512
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Certain + Right (ideal)     : 1007
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Uncertain + Wrong (caught)  : 298
2026-06-08 11:26:23 | INFO     | pipeline.orchestrators.test |   Uncertain + Right (over-ref): 183
2026-06-08 11:26:23 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_DDR-China.png
2026-06-08 11:26:23 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.2031
2026-06-08 11:26:23 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.0852
2026-06-08 11:26:23 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.2802
2026-06-08 11:26:23 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.0412
2026-06-08 11:26:23 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.0199
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_DDR-China.png
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_DDR-China_20260608_112624.png
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.2031
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.0852
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.2802
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.0412
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.0199
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_DDR-China_20260608_112624.png
2026-06-08 11:26:24 | INFO     | pipeline.evaluation.cosine_similarity | Extracting features from test set...
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Extracted features for 2000 samples, feature dim = 1280
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 0 (No DR)...
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 1 (Mild)...
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 2 (Moderate)...
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 3 (Severe)...
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 4 (Proliferative)...

--- Per-Class Cosine Similarity Summary ---
  [No DR]  min=-0.2896  max=0.8731  avg=0.5294
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity |   Class 0 (No DR) | min=-0.2896  max=0.8731  avg=0.5294
  [Mild]  min=-0.0820  max=0.7093  avg=0.2342
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity |   Class 1 (Mild) | min=-0.0820  max=0.7093  avg=0.2342
  [Moderate]  min=-0.1842  max=0.7322  avg=0.0962
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity |   Class 2 (Moderate) | min=-0.1842  max=0.7322  avg=0.0962
  [Severe]  min=-0.2134  max=0.8590  avg=0.0353
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity |   Class 3 (Severe) | min=-0.2134  max=0.8590  avg=0.0353
  [Proliferative]  min=-0.1999  max=0.7576  avg=0.0599
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity |   Class 4 (Proliferative) | min=-0.1999  max=0.7576  avg=0.0599

  [Global]  min=-0.2896  max=0.8731  avg=0.1910
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 0 (No DR) similarities → artifacts/cosine_similarity/DDR-China/similarities_class_0.npy
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 1 (Mild) similarities → artifacts/cosine_similarity/DDR-China/similarities_class_1.npy
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 2 (Moderate) similarities → artifacts/cosine_similarity/DDR-China/similarities_class_2.npy
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 3 (Severe) similarities → artifacts/cosine_similarity/DDR-China/similarities_class_3.npy
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 4 (Proliferative) similarities → artifacts/cosine_similarity/DDR-China/similarities_class_4.npy
2026-06-08 11:26:39 | INFO     | pipeline.evaluation.cosine_similarity | Saved test labels → artifacts/cosine_similarity/DDR-China/labels.npy
wandb: ⢿ updating run metadata (0.2s)
wandb: ⢿ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.0s)
wandb: ⣻ updating run metadata (0.2s)
wandb: ⣻ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.0s)
wandb: ⣽ updating run metadata (0.2s)
wandb: ⣽ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.0s)
wandb: ⣾ updating run metadata (0.2s)
wandb: ⣾ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.0s)
wandb: ⣷ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⣷ test_confusion_matrix_table.table.json 970B/970B (0.0s)
wandb: ⣯ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⣯ test_confusion_matrix_table.table.json 970B/970B (0.0s)
wandb: ⣟ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⣟ test_confusion_matrix_table.table.json 970B/970B (0.0s)
wandb: ⡿ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⡿ test_confusion_matrix_table.table.json 970B/970B (0.0s)
wandb: ⢿ uploading artifact run-ivelpxxi-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⢿ test_confusion_matrix_table.table.json 970B/970B (0.0s)
wandb: ⣻ uploading artifact run-ivelpxxi-test_confusion_matrix_table (1.0s)
wandb: ⣽ uploading artifact run-ivelpxxi-test_confusion_matrix_table (1.0s)
wandb: ⣾ uploading artifact run-ivelpxxi-test_confusion_matrix_table (1.0s)
wandb: ⣷ uploading artifact run-ivelpxxi-test_confusion_matrix_table (1.0s)
wandb: ⣯ uploading artifact run-ivelpxxi-test_confusion_matrix_table (1.0s)
wandb: 
wandb: Run history:
wandb:   CosineSimilarity/Global/Avg ▁
wandb:   CosineSimilarity/Global/Max ▁
wandb:   CosineSimilarity/Global/Min ▁
wandb:     CosineSimilarity/Mild/Avg ▁
wandb:     CosineSimilarity/Mild/Max ▁
wandb:     CosineSimilarity/Mild/Min ▁
wandb: CosineSimilarity/Moderate/Avg ▁
wandb: CosineSimilarity/Moderate/Max ▁
wandb: CosineSimilarity/Moderate/Min ▁
wandb:    CosineSimilarity/No DR/Avg ▁
wandb:                           +17 ...
wandb: 
wandb: Run summary:
wandb:   CosineSimilarity/Global/Avg 0.191
wandb:   CosineSimilarity/Global/Max 0.8731
wandb:   CosineSimilarity/Global/Min -0.2896
wandb:     CosineSimilarity/Mild/Avg 0.2342
wandb:     CosineSimilarity/Mild/Max 0.7093
wandb:     CosineSimilarity/Mild/Min -0.082
wandb: CosineSimilarity/Moderate/Avg 0.0962
wandb: CosineSimilarity/Moderate/Max 0.7322
wandb: CosineSimilarity/Moderate/Min -0.1842
wandb:    CosineSimilarity/No DR/Avg 0.5294
wandb:                           +17 ...
wandb: 
wandb: 🚀 View run swept-paper-47 at: https://wandb.ai/nova9-/aptos-blindness-detection/runs/ivelpxxi
wandb: ⭐️ View project at: https://wandb.ai/nova9-/aptos-blindness-detection
wandb: Synced 5 W&B file(s), 2 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260608_112505-ivelpxxi/logs
2026-06-08 11:26:40 | INFO     | pipeline.orchestrators.test | [DDR-China] test_model() complete.
2026-06-08 11:26:40 | INFO     | __main__ | ============================================================
2026-06-08 11:26:40 | INFO     | __main__ | STEP 2: Testing on Messidor-Grp1 | use_test_split=False
2026-06-08 11:26:40 | INFO     | __main__ | ============================================================
2026-06-08 11:26:40 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:26:40 | INFO     | pipeline.orchestrators.test | test_model() | dataset=Messidor-Grp1 | T=0.9262
2026-06-08 11:26:40 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:26:40 | INFO     | pipeline.setup.config | Device selected: cuda
2026-06-08 11:26:40 | INFO     | pipeline.setup.config | GPU: Tesla T4
2026-06-08 11:26:41 | INFO     | pipeline.training_loop_setup.model | EfficientNetMC | classes=5 | dropout=0.3 | pretrained=False | features=1280
2026-06-08 11:26:41 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] Model loaded from artifacts/weights/aptos_efficientnet.pth
wandb: ⢿ Waiting for wandb.init()...
wandb: ⣻ Waiting for wandb.init()...
wandb: ⣽ Waiting for wandb.init()...
wandb: ⣾ setting up run cfweb8g1 (0.4s)
wandb: ⣷ setting up run cfweb8g1 (0.4s)
wandb: ⣯ setting up run cfweb8g1 (0.4s)
wandb: ⣟ setting up run cfweb8g1 (0.4s)
wandb: ⡿ setting up run cfweb8g1 (0.4s)
wandb: ⢿ setting up run cfweb8g1 (0.9s)
wandb: ⣻ setting up run cfweb8g1 (0.9s)
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /kaggle/working/Diabetic-Retinopathy-detection/wandb/run-20260608_112641-cfweb8g1
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run clean-durian-48
wandb: ⭐️ View project at https://wandb.ai/nova9-/aptos-blindness-detection
wandb: 🚀 View run at https://wandb.ai/nova9-/aptos-blindness-detection/runs/cfweb8g1
2026-06-08 11:26:43 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] wandb run initialised | job=test
2026-06-08 11:26:43 | INFO     | pipeline.data.loaders | [Messidor-Grp1] Using full training CSV for evaluation (zero-shot)
2026-06-08 11:26:43 | INFO     | pipeline.data.dataset | RetinopathyDataset | rows=400 | ext=
2026-06-08 11:26:43 | INFO     | pipeline.data.loaders | [Messidor-Grp1] Test loader ready | samples=400 | batch=32
2026-06-08 11:26:43 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] MC Dropout | T=30
2026-06-08 11:26:43 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | T=30 passes | starting...
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | done | N=400 samples | mean std=0.0133
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] Applying temperature T=0.9262
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] QWK: 0.5094
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] Confusion Matrix:
[[150   0   0   1   0]
 [ 27   1   0   2   0]
 [ 58   0   7   5   0]
 [  0   0   0   0   0]
 [ 46   0  16  85   2]]
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] Mean entropy      : 0.5981
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] Mean margin       : 0.6823
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] Uncertain fraction: 0.2225
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] Triage Summary (first 20 samples):
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   0 | True:4 Pred:0 | H=1.011 M=0.311 MC=0.026 | UNCERTAIN - refer to specialist
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   1 | True:0 Pred:0 | H=0.060 M=0.987 MC=0.001 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   2 | True:1 Pred:0 | H=0.189 M=0.950 MC=0.004 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   3 | True:2 Pred:0 | H=0.299 M=0.893 MC=0.012 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   4 | True:4 Pred:0 | H=0.169 M=0.954 MC=0.005 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   5 | True:0 Pred:0 | H=0.183 M=0.947 MC=0.005 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   6 | True:0 Pred:0 | H=0.426 M=0.817 MC=0.010 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   7 | True:0 Pred:0 | H=0.363 M=0.850 MC=0.008 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   8 | True:4 Pred:3 | H=1.489 M=0.180 MC=0.021 | UNCERTAIN - refer to specialist
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample   9 | True:0 Pred:0 | H=0.185 M=0.943 MC=0.004 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  10 | True:0 Pred:0 | H=0.293 M=0.917 MC=0.006 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  11 | True:4 Pred:0 | H=0.107 M=0.971 MC=0.003 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  12 | True:4 Pred:0 | H=0.520 M=0.784 MC=0.014 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  13 | True:0 Pred:0 | H=0.395 M=0.888 MC=0.007 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  14 | True:0 Pred:0 | H=0.889 M=0.551 MC=0.021 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  15 | True:0 Pred:0 | H=0.042 M=0.991 MC=0.001 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  16 | True:1 Pred:0 | H=0.069 M=0.985 MC=0.002 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  17 | True:4 Pred:3 | H=0.607 M=0.644 MC=0.016 | HIGH SEVERITY - urgent review
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  18 | True:2 Pred:3 | H=1.354 M=0.146 MC=0.027 | UNCERTAIN - refer to specialist
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Sample  19 | True:2 Pred:0 | H=0.805 M=0.564 MC=0.019 | ROUTINE
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] --- Four Quadrant Uncertainty Breakdown ---
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Certain + Wrong (dangerous): 153
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Certain + Right (ideal)     : 149
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Uncertain + Wrong (caught)  : 87
2026-06-08 11:27:03 | INFO     | pipeline.orchestrators.test |   Uncertain + Right (over-ref): 11
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_Messidor-Grp1.png
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.2648
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.0506
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.1644
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.2112
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.3520
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_Messidor-Grp1.png
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_Messidor-Grp1_20260608_112703.png
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.2648
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.0506
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.1644
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.2112
2026-06-08 11:27:03 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.3520
2026-06-08 11:27:04 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_Messidor-Grp1_20260608_112703.png
2026-06-08 11:27:04 | INFO     | pipeline.evaluation.cosine_similarity | Extracting features from test set...
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Extracted features for 400 samples, feature dim = 1280
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 0 (No DR)...
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 1 (Mild)...
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 2 (Moderate)...
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 3 (Severe)...
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 4 (Proliferative)...

--- Per-Class Cosine Similarity Summary ---
  [No DR]  min=-0.1287  max=0.7996  avg=0.4717
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity |   Class 0 (No DR) | min=-0.1287  max=0.7996  avg=0.4717
  [Mild]  min=-0.0940  max=0.3054  avg=0.1284
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity |   Class 1 (Mild) | min=-0.0940  max=0.3054  avg=0.1284
  [Moderate]  min=-0.1243  max=0.5766  avg=0.1203
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity |   Class 2 (Moderate) | min=-0.1243  max=0.5766  avg=0.1203
  [Severe]  min=-0.1600  max=0.6689  avg=0.0887
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity |   Class 3 (Severe) | min=-0.1600  max=0.6689  avg=0.0887
  [Proliferative]  min=-0.1623  max=0.5856  avg=0.0513
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity |   Class 4 (Proliferative) | min=-0.1623  max=0.5856  avg=0.0513

  [Global]  min=-0.1623  max=0.7996  avg=0.1721
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 0 (No DR) similarities → artifacts/cosine_similarity/Messidor-Grp1/similarities_class_0.npy
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 1 (Mild) similarities → artifacts/cosine_similarity/Messidor-Grp1/similarities_class_1.npy
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 2 (Moderate) similarities → artifacts/cosine_similarity/Messidor-Grp1/similarities_class_2.npy
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 3 (Severe) similarities → artifacts/cosine_similarity/Messidor-Grp1/similarities_class_3.npy
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 4 (Proliferative) similarities → artifacts/cosine_similarity/Messidor-Grp1/similarities_class_4.npy
2026-06-08 11:27:13 | INFO     | pipeline.evaluation.cosine_similarity | Saved test labels → artifacts/cosine_similarity/Messidor-Grp1/labels.npy
wandb: ⢿ updating run metadata (0.2s)
wandb: ⢿ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.0s)
wandb: ⣻ updating run metadata (0.2s)
wandb: ⣻ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.0s)
wandb: ⣽ updating run metadata (0.2s)
wandb: ⣽ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.0s)
wandb: ⣾ updating run metadata (0.2s)
wandb: ⣾ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.0s)
wandb: ⣷ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.5s)
wandb: ⣯ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.5s)
wandb: ⣟ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.5s)
wandb: ⡿ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.5s)
wandb: ⢿ uploading artifact run-cfweb8g1-test_confusion_matrix_table (0.5s)
wandb: ⣻ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.0s)
wandb: ⣽ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.0s)
wandb: ⣾ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.0s)
wandb: ⣷ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.0s)
wandb: ⣯ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.0s)
wandb: ⣟ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.5s)
wandb: ⡿ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.5s)
wandb: ⢿ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.5s)
wandb: ⣻ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.5s)
wandb: ⣽ uploading artifact run-cfweb8g1-test_confusion_matrix_table (1.5s)
wandb: ⣾ uploading history steps 0-1, summary, console lines 59-84 (0.1s)
wandb: ⣷ uploading history steps 0-1, summary, console lines 59-84 (0.1s)
wandb: ⣯ uploading history steps 0-1, summary, console lines 59-84 (0.1s)
wandb: 
wandb: Run history:
wandb:   CosineSimilarity/Global/Avg ▁
wandb:   CosineSimilarity/Global/Max ▁
wandb:   CosineSimilarity/Global/Min ▁
wandb:     CosineSimilarity/Mild/Avg ▁
wandb:     CosineSimilarity/Mild/Max ▁
wandb:     CosineSimilarity/Mild/Min ▁
wandb: CosineSimilarity/Moderate/Avg ▁
wandb: CosineSimilarity/Moderate/Max ▁
wandb: CosineSimilarity/Moderate/Min ▁
wandb:    CosineSimilarity/No DR/Avg ▁
wandb:                           +17 ...
wandb: 
wandb: Run summary:
wandb:   CosineSimilarity/Global/Avg 0.1721
wandb:   CosineSimilarity/Global/Max 0.7996
wandb:   CosineSimilarity/Global/Min -0.1623
wandb:     CosineSimilarity/Mild/Avg 0.1284
wandb:     CosineSimilarity/Mild/Max 0.3054
wandb:     CosineSimilarity/Mild/Min -0.094
wandb: CosineSimilarity/Moderate/Avg 0.1203
wandb: CosineSimilarity/Moderate/Max 0.5766
wandb: CosineSimilarity/Moderate/Min -0.1243
wandb:    CosineSimilarity/No DR/Avg 0.4717
wandb:                           +17 ...
wandb: 
wandb: 🚀 View run clean-durian-48 at: https://wandb.ai/nova9-/aptos-blindness-detection/runs/cfweb8g1
wandb: ⭐️ View project at: https://wandb.ai/nova9-/aptos-blindness-detection
wandb: Synced 5 W&B file(s), 2 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260608_112641-cfweb8g1/logs
2026-06-08 11:27:16 | INFO     | pipeline.orchestrators.test | [Messidor-Grp1] test_model() complete.
2026-06-08 11:27:16 | INFO     | __main__ | ============================================================
2026-06-08 11:27:16 | INFO     | __main__ | STEP 2: Testing on Messidor-Grp2 | use_test_split=False
2026-06-08 11:27:16 | INFO     | __main__ | ============================================================
2026-06-08 11:27:16 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:27:16 | INFO     | pipeline.orchestrators.test | test_model() | dataset=Messidor-Grp2 | T=0.9262
2026-06-08 11:27:16 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:27:16 | INFO     | pipeline.setup.config | Device selected: cuda
2026-06-08 11:27:16 | INFO     | pipeline.setup.config | GPU: Tesla T4
2026-06-08 11:27:16 | INFO     | pipeline.training_loop_setup.model | EfficientNetMC | classes=5 | dropout=0.3 | pretrained=False | features=1280
2026-06-08 11:27:16 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] Model loaded from artifacts/weights/aptos_efficientnet.pth
wandb: ⢿ Waiting for wandb.init()...
wandb: ⣻ Waiting for wandb.init()...
wandb: ⣽ Waiting for wandb.init()...
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /kaggle/working/Diabetic-Retinopathy-detection/wandb/run-20260608_112716-220i0ua5
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run glamorous-plasma-49
wandb: ⭐️ View project at https://wandb.ai/nova9-/aptos-blindness-detection
wandb: 🚀 View run at https://wandb.ai/nova9-/aptos-blindness-detection/runs/220i0ua5
2026-06-08 11:27:18 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] wandb run initialised | job=test
2026-06-08 11:27:18 | INFO     | pipeline.data.loaders | [Messidor-Grp2] Using full training CSV for evaluation (zero-shot)
2026-06-08 11:27:18 | INFO     | pipeline.data.dataset | RetinopathyDataset | rows=400 | ext=
2026-06-08 11:27:18 | INFO     | pipeline.data.loaders | [Messidor-Grp2] Test loader ready | samples=400 | batch=32
2026-06-08 11:27:18 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] MC Dropout | T=30
2026-06-08 11:27:18 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | T=30 passes | starting...
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | done | N=400 samples | mean std=0.0101
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] Applying temperature T=0.9262
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] QWK: 0.4685
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] Confusion Matrix:
[[184   0   0   2   0]
 [ 67   0   0   4   0]
 [ 79   0   4   8   0]
 [  0   0   0   0   0]
 [ 15   0   6  31   0]]
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] Mean entropy      : 0.4630
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] Mean margin       : 0.7665
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] Uncertain fraction: 0.1700
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] Triage Summary (first 20 samples):
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   0 | True:0 Pred:0 | H=0.342 M=0.870 MC=0.012 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   1 | True:0 Pred:0 | H=0.394 M=0.871 MC=0.009 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   2 | True:0 Pred:0 | H=0.414 M=0.846 MC=0.009 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   3 | True:0 Pred:0 | H=1.143 M=0.196 MC=0.024 | UNCERTAIN - refer to specialist
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   4 | True:0 Pred:0 | H=0.750 M=0.673 MC=0.019 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   5 | True:0 Pred:0 | H=0.127 M=0.967 MC=0.004 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   6 | True:0 Pred:0 | H=0.076 M=0.982 MC=0.002 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   7 | True:0 Pred:0 | H=0.043 M=0.991 MC=0.001 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   8 | True:0 Pred:0 | H=0.133 M=0.965 MC=0.004 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample   9 | True:0 Pred:0 | H=0.181 M=0.953 MC=0.004 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  10 | True:0 Pred:0 | H=0.271 M=0.913 MC=0.008 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  11 | True:0 Pred:0 | H=0.078 M=0.979 MC=0.002 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  12 | True:2 Pred:0 | H=0.298 M=0.894 MC=0.008 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  13 | True:2 Pred:0 | H=0.998 M=0.415 MC=0.020 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  14 | True:0 Pred:0 | H=0.546 M=0.804 MC=0.011 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  15 | True:1 Pred:0 | H=0.236 M=0.936 MC=0.006 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  16 | True:4 Pred:2 | H=1.285 M=0.005 MC=0.020 | UNCERTAIN - refer to specialist
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  17 | True:4 Pred:2 | H=0.921 M=0.491 MC=0.018 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  18 | True:2 Pred:0 | H=1.261 M=0.150 MC=0.028 | UNCERTAIN - refer to specialist
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Sample  19 | True:1 Pred:0 | H=0.808 M=0.670 MC=0.014 | ROUTINE
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] --- Four Quadrant Uncertainty Breakdown ---
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Certain + Wrong (dangerous): 153
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Certain + Right (ideal)     : 175
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Uncertain + Wrong (caught)  : 59
2026-06-08 11:27:36 | INFO     | pipeline.orchestrators.test |   Uncertain + Right (over-ref): 13
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_Messidor-Grp2.png
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.3255
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.1516
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.1945
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.1138
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.1185
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_Messidor-Grp2.png
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_Messidor-Grp2_20260608_112736.png
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.3255
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.1516
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.1945
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.1138
2026-06-08 11:27:36 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.1185
2026-06-08 11:27:37 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_Messidor-Grp2_20260608_112736.png
2026-06-08 11:27:37 | INFO     | pipeline.evaluation.cosine_similarity | Extracting features from test set...
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Extracted features for 400 samples, feature dim = 1280
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 0 (No DR)...
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 1 (Mild)...
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 2 (Moderate)...
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 3 (Severe)...
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 4 (Proliferative)...

--- Per-Class Cosine Similarity Summary ---
  [No DR]  min=-0.0662  max=0.8162  avg=0.5658
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity |   Class 0 (No DR) | min=-0.0662  max=0.8162  avg=0.5658
  [Mild]  min=-0.0125  max=0.2586  avg=0.1173
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity |   Class 1 (Mild) | min=-0.0125  max=0.2586  avg=0.1173
  [Moderate]  min=-0.1169  max=0.5572  avg=0.0589
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity |   Class 2 (Moderate) | min=-0.1169  max=0.5572  avg=0.0589
  [Severe]  min=-0.1381  max=0.5529  avg=0.0208
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity |   Class 3 (Severe) | min=-0.1381  max=0.5529  avg=0.0208
  [Proliferative]  min=-0.1499  max=0.5197  avg=-0.0088
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity |   Class 4 (Proliferative) | min=-0.1499  max=0.5197  avg=-0.0088

  [Global]  min=-0.1499  max=0.8162  avg=0.1508
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 0 (No DR) similarities → artifacts/cosine_similarity/Messidor-Grp2/similarities_class_0.npy
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 1 (Mild) similarities → artifacts/cosine_similarity/Messidor-Grp2/similarities_class_1.npy
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 2 (Moderate) similarities → artifacts/cosine_similarity/Messidor-Grp2/similarities_class_2.npy
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 3 (Severe) similarities → artifacts/cosine_similarity/Messidor-Grp2/similarities_class_3.npy
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 4 (Proliferative) similarities → artifacts/cosine_similarity/Messidor-Grp2/similarities_class_4.npy
2026-06-08 11:27:41 | INFO     | pipeline.evaluation.cosine_similarity | Saved test labels → artifacts/cosine_similarity/Messidor-Grp2/labels.npy
wandb: ⢿ updating run metadata (0.2s)
wandb: ⢿ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.0s)
wandb: ⣻ updating run metadata (0.2s)
wandb: ⣻ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.0s)
wandb: ⣽ updating run metadata (0.2s)
wandb: ⣽ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.0s)
wandb: ⣾ updating run metadata (0.2s)
wandb: ⣾ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.0s)
wandb: ⣷ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.5s)
wandb: ⣯ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.5s)
wandb: ⣟ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.5s)
wandb: ⡿ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.5s)
wandb: ⢿ uploading artifact run-220i0ua5-test_confusion_matrix_table (0.5s)
wandb: ⣻ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.0s)
wandb: ⣽ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.0s)
wandb: ⣾ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.0s)
wandb: ⣷ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.0s)
wandb: ⣯ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.0s)
wandb: ⣟ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.5s)
wandb: ⡿ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.5s)
wandb: ⢿ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.5s)
wandb: ⣻ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.5s)
wandb: ⣽ uploading artifact run-220i0ua5-test_confusion_matrix_table (1.5s)
wandb: ⣾ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.0s)
wandb: ⣷ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.0s)
wandb: ⣯ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.0s)
wandb: ⣟ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.0s)
wandb: ⡿ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.0s)
wandb: ⢿ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.5s)
wandb: ⣻ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.5s)
wandb: ⣽ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.5s)
wandb: ⣾ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.5s)
wandb: ⣷ uploading artifact run-220i0ua5-test_confusion_matrix_table (2.5s)
wandb: ⣯ uploading media/images/test_calibration_plot_1_5cd77fbac4ef1ae3528e....
wandb: ⣯ uploading media/table/test_confusion_matrix_table_1_52d98c93a11b3bd0...
wandb: ⣟ uploading media/images/test_calibration_plot_1_5cd77fbac4ef1ae3528e....
wandb: ⣟ uploading media/table/test_confusion_matrix_table_1_52d98c93a11b3bd0...
wandb: 
wandb: Run history:
wandb:   CosineSimilarity/Global/Avg ▁
wandb:   CosineSimilarity/Global/Max ▁
wandb:   CosineSimilarity/Global/Min ▁
wandb:     CosineSimilarity/Mild/Avg ▁
wandb:     CosineSimilarity/Mild/Max ▁
wandb:     CosineSimilarity/Mild/Min ▁
wandb: CosineSimilarity/Moderate/Avg ▁
wandb: CosineSimilarity/Moderate/Max ▁
wandb: CosineSimilarity/Moderate/Min ▁
wandb:    CosineSimilarity/No DR/Avg ▁
wandb:                           +17 ...
wandb: 
wandb: Run summary:
wandb:   CosineSimilarity/Global/Avg 0.1508
wandb:   CosineSimilarity/Global/Max 0.8162
wandb:   CosineSimilarity/Global/Min -0.1499
wandb:     CosineSimilarity/Mild/Avg 0.1173
wandb:     CosineSimilarity/Mild/Max 0.2586
wandb:     CosineSimilarity/Mild/Min -0.0125
wandb: CosineSimilarity/Moderate/Avg 0.0589
wandb: CosineSimilarity/Moderate/Max 0.5572
wandb: CosineSimilarity/Moderate/Min -0.1169
wandb:    CosineSimilarity/No DR/Avg 0.5658
wandb:                           +17 ...
wandb: 
wandb: 🚀 View run glamorous-plasma-49 at: https://wandb.ai/nova9-/aptos-blindness-detection/runs/220i0ua5
wandb: ⭐️ View project at: https://wandb.ai/nova9-/aptos-blindness-detection
wandb: Synced 5 W&B file(s), 2 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260608_112716-220i0ua5/logs
2026-06-08 11:27:44 | INFO     | pipeline.orchestrators.test | [Messidor-Grp2] test_model() complete.
2026-06-08 11:27:44 | INFO     | __main__ | ============================================================
2026-06-08 11:27:44 | INFO     | __main__ | STEP 2: Testing on Messidor-Grp3 | use_test_split=False
2026-06-08 11:27:44 | INFO     | __main__ | ============================================================
2026-06-08 11:27:44 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:27:44 | INFO     | pipeline.orchestrators.test | test_model() | dataset=Messidor-Grp3 | T=0.9262
2026-06-08 11:27:44 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:27:44 | INFO     | pipeline.setup.config | Device selected: cuda
2026-06-08 11:27:44 | INFO     | pipeline.setup.config | GPU: Tesla T4
2026-06-08 11:27:44 | INFO     | pipeline.training_loop_setup.model | EfficientNetMC | classes=5 | dropout=0.3 | pretrained=False | features=1280
2026-06-08 11:27:44 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] Model loaded from artifacts/weights/aptos_efficientnet.pth
wandb: ⢿ Waiting for wandb.init()...
wandb: ⣻ Waiting for wandb.init()...
wandb: ⣽ Waiting for wandb.init()...
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /kaggle/working/Diabetic-Retinopathy-detection/wandb/run-20260608_112744-6yg166yn
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run scarlet-durian-50
wandb: ⭐️ View project at https://wandb.ai/nova9-/aptos-blindness-detection
wandb: 🚀 View run at https://wandb.ai/nova9-/aptos-blindness-detection/runs/6yg166yn
2026-06-08 11:27:45 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] wandb run initialised | job=test
2026-06-08 11:27:45 | INFO     | pipeline.data.loaders | [Messidor-Grp3] Using full training CSV for evaluation (zero-shot)
2026-06-08 11:27:45 | INFO     | pipeline.data.dataset | RetinopathyDataset | rows=400 | ext=
2026-06-08 11:27:45 | INFO     | pipeline.data.loaders | [Messidor-Grp3] Test loader ready | samples=400 | batch=32
2026-06-08 11:27:45 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] MC Dropout | T=30
2026-06-08 11:27:45 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | T=30 passes | starting...
2026-06-08 11:28:03 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | done | N=400 samples | mean std=0.0132
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] Applying temperature T=0.9262
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] QWK: 0.4514
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] Confusion Matrix:
[[204   0   0   5   0]
 [ 50   0   1   1   0]
 [ 71   1   2  12   0]
 [  0   0   0   0   0]
 [ 18   1   5  29   0]]
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] Mean entropy      : 0.6146
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] Mean margin       : 0.6869
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] Uncertain fraction: 0.2300
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] Triage Summary (first 20 samples):
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   0 | True:0 Pred:0 | H=0.399 M=0.870 MC=0.011 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   1 | True:0 Pred:0 | H=0.276 M=0.914 MC=0.008 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   2 | True:0 Pred:0 | H=0.177 M=0.948 MC=0.005 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   3 | True:0 Pred:0 | H=0.260 M=0.927 MC=0.005 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   4 | True:0 Pred:0 | H=0.414 M=0.880 MC=0.010 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   5 | True:0 Pred:0 | H=0.154 M=0.955 MC=0.004 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   6 | True:0 Pred:0 | H=0.269 M=0.908 MC=0.006 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   7 | True:0 Pred:0 | H=0.488 M=0.825 MC=0.009 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   8 | True:0 Pred:0 | H=0.367 M=0.878 MC=0.009 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample   9 | True:0 Pred:0 | H=0.291 M=0.920 MC=0.006 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  10 | True:0 Pred:0 | H=0.335 M=0.906 MC=0.007 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  11 | True:0 Pred:0 | H=0.630 M=0.749 MC=0.011 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  12 | True:0 Pred:0 | H=0.525 M=0.786 MC=0.013 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  13 | True:0 Pred:0 | H=0.305 M=0.904 MC=0.006 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  14 | True:0 Pred:0 | H=0.228 M=0.931 MC=0.006 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  15 | True:0 Pred:0 | H=0.810 M=0.614 MC=0.018 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  16 | True:0 Pred:0 | H=0.088 M=0.977 MC=0.002 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  17 | True:0 Pred:0 | H=0.121 M=0.968 MC=0.004 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  18 | True:0 Pred:0 | H=0.296 M=0.909 MC=0.007 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Sample  19 | True:0 Pred:0 | H=0.905 M=0.565 MC=0.022 | ROUTINE
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] --- Four Quadrant Uncertainty Breakdown ---
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Certain + Wrong (dangerous): 120
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Certain + Right (ideal)     : 185
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Uncertain + Wrong (caught)  : 74
2026-06-08 11:28:03 | INFO     | pipeline.orchestrators.test |   Uncertain + Right (over-ref): 21
2026-06-08 11:28:03 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_Messidor-Grp3.png
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.2153
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.1047
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.1826
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.1404
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.1211
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_Messidor-Grp3.png
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_Messidor-Grp3_20260608_112804.png
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.2153
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.1047
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.1826
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.1404
2026-06-08 11:28:04 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.1211
2026-06-08 11:28:05 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_Messidor-Grp3_20260608_112804.png
2026-06-08 11:28:05 | INFO     | pipeline.evaluation.cosine_similarity | Extracting features from test set...
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Extracted features for 400 samples, feature dim = 1280
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 0 (No DR)...
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 1 (Mild)...
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 2 (Moderate)...
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 3 (Severe)...
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 4 (Proliferative)...

--- Per-Class Cosine Similarity Summary ---
  [No DR]  min=-0.0119  max=0.7674  avg=0.5201
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity |   Class 0 (No DR) | min=-0.0119  max=0.7674  avg=0.5201
  [Mild]  min=-0.0020  max=0.3547  avg=0.1305
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity |   Class 1 (Mild) | min=-0.0020  max=0.3547  avg=0.1305
  [Moderate]  min=-0.1122  max=0.5230  avg=0.0721
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity |   Class 2 (Moderate) | min=-0.1122  max=0.5230  avg=0.0721
  [Severe]  min=-0.1407  max=0.4591  avg=0.0285
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity |   Class 3 (Severe) | min=-0.1407  max=0.4591  avg=0.0285
  [Proliferative]  min=-0.1655  max=0.4433  avg=-0.0066
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity |   Class 4 (Proliferative) | min=-0.1655  max=0.4433  avg=-0.0066

  [Global]  min=-0.1655  max=0.7674  avg=0.1489
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 0 (No DR) similarities → artifacts/cosine_similarity/Messidor-Grp3/similarities_class_0.npy
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 1 (Mild) similarities → artifacts/cosine_similarity/Messidor-Grp3/similarities_class_1.npy
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 2 (Moderate) similarities → artifacts/cosine_similarity/Messidor-Grp3/similarities_class_2.npy
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 3 (Severe) similarities → artifacts/cosine_similarity/Messidor-Grp3/similarities_class_3.npy
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 4 (Proliferative) similarities → artifacts/cosine_similarity/Messidor-Grp3/similarities_class_4.npy
2026-06-08 11:28:11 | INFO     | pipeline.evaluation.cosine_similarity | Saved test labels → artifacts/cosine_similarity/Messidor-Grp3/labels.npy
wandb: ⢿ updating run metadata (0.2s)
wandb: ⢿ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.0s)
wandb: ⣻ updating run metadata (0.2s)
wandb: ⣻ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.0s)
wandb: ⣽ updating run metadata (0.2s)
wandb: ⣽ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.0s)
wandb: ⣾ updating run metadata (0.2s)
wandb: ⣾ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.0s)
wandb: ⣷ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⣷ test_confusion_matrix_table.table.json 960B/960B (0.1s)
wandb: ⣯ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⣯ test_confusion_matrix_table.table.json 960B/960B (0.1s)
wandb: ⣟ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⣟ test_confusion_matrix_table.table.json 960B/960B (0.1s)
wandb: ⡿ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⡿ test_confusion_matrix_table.table.json 960B/960B (0.1s)
wandb: ⢿ uploading artifact run-6yg166yn-test_confusion_matrix_table (0.5s)
wandb:   ↳ ⢿ test_confusion_matrix_table.table.json 960B/960B (0.1s)
wandb: ⣻ uploading artifact run-6yg166yn-test_confusion_matrix_table (1.0s)
wandb: ⣽ uploading artifact run-6yg166yn-test_confusion_matrix_table (1.0s)
wandb: ⣾ uploading artifact run-6yg166yn-test_confusion_matrix_table (1.0s)
wandb: ⣷ uploading artifact run-6yg166yn-test_confusion_matrix_table (1.0s)
wandb: ⣯ uploading artifact run-6yg166yn-test_confusion_matrix_table (1.0s)
wandb: 
wandb: Run history:
wandb:   CosineSimilarity/Global/Avg ▁
wandb:   CosineSimilarity/Global/Max ▁
wandb:   CosineSimilarity/Global/Min ▁
wandb:     CosineSimilarity/Mild/Avg ▁
wandb:     CosineSimilarity/Mild/Max ▁
wandb:     CosineSimilarity/Mild/Min ▁
wandb: CosineSimilarity/Moderate/Avg ▁
wandb: CosineSimilarity/Moderate/Max ▁
wandb: CosineSimilarity/Moderate/Min ▁
wandb:    CosineSimilarity/No DR/Avg ▁
wandb:                           +17 ...
wandb: 
wandb: Run summary:
wandb:   CosineSimilarity/Global/Avg 0.1489
wandb:   CosineSimilarity/Global/Max 0.7674
wandb:   CosineSimilarity/Global/Min -0.1655
wandb:     CosineSimilarity/Mild/Avg 0.1305
wandb:     CosineSimilarity/Mild/Max 0.3547
wandb:     CosineSimilarity/Mild/Min -0.002
wandb: CosineSimilarity/Moderate/Avg 0.0721
wandb: CosineSimilarity/Moderate/Max 0.523
wandb: CosineSimilarity/Moderate/Min -0.1122
wandb:    CosineSimilarity/No DR/Avg 0.5201
wandb:                           +17 ...
wandb: 
wandb: 🚀 View run scarlet-durian-50 at: https://wandb.ai/nova9-/aptos-blindness-detection/runs/6yg166yn
wandb: ⭐️ View project at: https://wandb.ai/nova9-/aptos-blindness-detection
wandb: Synced 5 W&B file(s), 2 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260608_112744-6yg166yn/logs
2026-06-08 11:28:13 | INFO     | pipeline.orchestrators.test | [Messidor-Grp3] test_model() complete.
2026-06-08 11:28:13 | INFO     | __main__ | ============================================================
2026-06-08 11:28:13 | INFO     | __main__ | STEP 2: Testing on EyePACS-Resized | use_test_split=False
2026-06-08 11:28:13 | INFO     | __main__ | ============================================================
2026-06-08 11:28:13 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:28:13 | INFO     | pipeline.orchestrators.test | test_model() | dataset=EyePACS-Resized | T=0.9262
2026-06-08 11:28:13 | INFO     | pipeline.orchestrators.test | ============================================================
2026-06-08 11:28:13 | INFO     | pipeline.setup.config | Device selected: cuda
2026-06-08 11:28:13 | INFO     | pipeline.setup.config | GPU: Tesla T4
2026-06-08 11:28:13 | INFO     | pipeline.training_loop_setup.model | EfficientNetMC | classes=5 | dropout=0.3 | pretrained=False | features=1280
2026-06-08 11:28:13 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] Model loaded from artifacts/weights/aptos_efficientnet.pth
wandb: ⢿ Waiting for wandb.init()...
wandb: ⣻ Waiting for wandb.init()...
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /kaggle/working/Diabetic-Retinopathy-detection/wandb/run-20260608_112813-req009gc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run amber-sun-51
wandb: ⭐️ View project at https://wandb.ai/nova9-/aptos-blindness-detection
wandb: 🚀 View run at https://wandb.ai/nova9-/aptos-blindness-detection/runs/req009gc
2026-06-08 11:28:14 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] wandb run initialised | job=test
2026-06-08 11:28:14 | INFO     | pipeline.data.loaders | [EyePACS-Resized] Using full training CSV for evaluation (zero-shot)
2026-06-08 11:28:14 | INFO     | pipeline.data.dataset | RetinopathyDataset | rows=2000 | ext=.jpeg
2026-06-08 11:28:14 | INFO     | pipeline.data.loaders | [EyePACS-Resized] Test loader ready | samples=2000 | batch=32
2026-06-08 11:28:14 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] MC Dropout | T=30
2026-06-08 11:28:14 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | T=30 passes | starting...
2026-06-08 11:30:15 | INFO     | pipeline.evaluation.evaluate | mc_evaluate_full() | done | N=2000 samples | mean std=0.0129
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] Applying temperature T=0.9262
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] QWK: 0.4303
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] Confusion Matrix:
[[1250   90   67   29   34]
 [ 113   12    7    6    1]
 [ 173   30   65   26    7]
 [   8    4   14   18    6]
 [   3    3    4   11   19]]
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] Mean entropy      : 0.5350
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] Mean margin       : 0.6911
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] Uncertain fraction: 0.2340
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] Triage Summary (first 20 samples):
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   0 | True:0 Pred:0 | H=0.110 M=0.977 MC=0.003 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   1 | True:0 Pred:0 | H=1.210 M=0.263 MC=0.026 | UNCERTAIN - refer to specialist
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   2 | True:0 Pred:0 | H=1.227 M=0.173 MC=0.019 | UNCERTAIN - refer to specialist
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   3 | True:0 Pred:0 | H=0.074 M=0.983 MC=0.002 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   4 | True:0 Pred:0 | H=0.009 M=0.998 MC=0.000 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   5 | True:0 Pred:0 | H=0.408 M=0.782 MC=0.015 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   6 | True:2 Pred:2 | H=1.410 M=0.031 MC=0.021 | UNCERTAIN - refer to specialist
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   7 | True:0 Pred:0 | H=1.076 M=0.457 MC=0.022 | UNCERTAIN - refer to specialist
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   8 | True:4 Pred:3 | H=0.556 M=0.746 MC=0.021 | HIGH SEVERITY - urgent review
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample   9 | True:0 Pred:0 | H=0.421 M=0.864 MC=0.009 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  10 | True:0 Pred:0 | H=0.985 M=0.425 MC=0.023 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  11 | True:1 Pred:2 | H=1.217 M=0.087 MC=0.020 | UNCERTAIN - refer to specialist
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  12 | True:0 Pred:0 | H=0.836 M=0.509 MC=0.023 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  13 | True:0 Pred:0 | H=0.046 M=0.987 MC=0.001 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  14 | True:0 Pred:0 | H=0.002 M=1.000 MC=0.000 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  15 | True:0 Pred:1 | H=1.039 M=0.093 MC=0.023 | UNCERTAIN - refer to specialist
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  16 | True:0 Pred:0 | H=1.348 M=0.147 MC=0.020 | UNCERTAIN - refer to specialist
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  17 | True:0 Pred:0 | H=0.726 M=0.576 MC=0.019 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  18 | True:0 Pred:0 | H=0.903 M=0.631 MC=0.022 | ROUTINE
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Sample  19 | True:2 Pred:3 | H=0.442 M=0.821 MC=0.016 | HIGH SEVERITY - urgent review
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test | [EyePACS-Resized] --- Four Quadrant Uncertainty Breakdown ---
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Certain + Wrong (dangerous): 322
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Certain + Right (ideal)     : 1138
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Uncertain + Wrong (caught)  : 314
2026-06-08 11:30:15 | INFO     | pipeline.orchestrators.test |   Uncertain + Right (over-ref): 226
2026-06-08 11:30:15 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_EyePACS-Resized.png
2026-06-08 11:30:15 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.1929
2026-06-08 11:30:15 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.0999
2026-06-08 11:30:15 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.0874
2026-06-08 11:30:15 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.0366
2026-06-08 11:30:15 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.0278
2026-06-08 11:30:16 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_EyePACS-Resized.png
2026-06-08 11:30:16 | INFO     | pipeline.evaluation.calibration | per_class_calibration() | n_classes=5 | save_path=artifacts/calibration/plots/calibration_EyePACS-Resized_20260608_113016.png
2026-06-08 11:30:16 | INFO     | pipeline.evaluation.calibration |   Class 0 (No DR) ECE: 0.1929
2026-06-08 11:30:16 | INFO     | pipeline.evaluation.calibration |   Class 1 (Mild) ECE: 0.0999
2026-06-08 11:30:16 | INFO     | pipeline.evaluation.calibration |   Class 2 (Moderate) ECE: 0.0874
2026-06-08 11:30:16 | INFO     | pipeline.evaluation.calibration |   Class 3 (Severe) ECE: 0.0366
2026-06-08 11:30:16 | INFO     | pipeline.evaluation.calibration |   Class 4 (Proliferative) ECE: 0.0278
2026-06-08 11:30:17 | INFO     | pipeline.evaluation.calibration | Calibration plot saved → artifacts/calibration/plots/calibration_EyePACS-Resized_20260608_113016.png
2026-06-08 11:30:17 | INFO     | pipeline.evaluation.cosine_similarity | Extracting features from test set...
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Extracted features for 2000 samples, feature dim = 1280
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 0 (No DR)...
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 1 (Mild)...
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 2 (Moderate)...
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 3 (Severe)...
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Computing Cosine similarity to class 4 (Proliferative)...

--- Per-Class Cosine Similarity Summary ---
  [No DR]  min=-0.1394  max=0.8636  avg=0.5204
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity |   Class 0 (No DR) | min=-0.1394  max=0.8636  avg=0.5204
  [Mild]  min=-0.0435  max=0.7665  avg=0.2236
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity |   Class 1 (Mild) | min=-0.0435  max=0.7665  avg=0.2236
  [Moderate]  min=-0.1605  max=0.8048  avg=0.1152
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity |   Class 2 (Moderate) | min=-0.1605  max=0.8048  avg=0.1152
  [Severe]  min=-0.1713  max=0.8110  avg=0.0600
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity |   Class 3 (Severe) | min=-0.1713  max=0.8110  avg=0.0600
  [Proliferative]  min=-0.1806  max=0.7660  avg=0.0675
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity |   Class 4 (Proliferative) | min=-0.1806  max=0.7660  avg=0.0675

  [Global]  min=-0.1806  max=0.8636  avg=0.1973
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 0 (No DR) similarities → artifacts/cosine_similarity/EyePACS-Resized/similarities_class_0.npy
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 1 (Mild) similarities → artifacts/cosine_similarity/EyePACS-Resized/similarities_class_1.npy
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 2 (Moderate) similarities → artifacts/cosine_similarity/EyePACS-Resized/similarities_class_2.npy
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 3 (Severe) similarities → artifacts/cosine_similarity/EyePACS-Resized/similarities_class_3.npy
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Saved class 4 (Proliferative) similarities → artifacts/cosine_similarity/EyePACS-Resized/similarities_class_4.npy
2026-06-08 11:31:59 | INFO     | pipeline.evaluation.cosine_similarity | Saved test labels → artifacts/cosine_similarity/EyePACS-Resized/labels.npy


so from this heatmap I noticed these thing there are 3 groups being formed - ddr and eyepacs as first, all the messidor as another group and idrid as it's own seperate entityfor ddr and eyepacs, the model is decent around no DR and prolific and it's more like these are the 2 dominant political parties and all the other ones are in some way supporter of either of these.for the messidor there is just one ruler that pulls everything to it's side, no DRfor idrid, it's almost balanced out, everyone has their own take and the cluster is more dense and broader then group 1now the reason for this is happening in messidor is because of the imbalance in data and there is no severe category so the cluster around the right bottom corner is never formednow i think that the model is able to get the class 4 and 0 so right because they are extremes and they are clearly distinguishable, there is a smooth progression curve between classes and model is struggling to draw a decision boundary to separate them and it's easy to misdiagnoise to adjacent classes or cluster around the extremes - this are my 2 hypotheses from what I saw

Claude responded: Good observations.Good observations. Let me validate what's right, push back where needed, and sharpen the hypotheses.

What's Solid
Observation 1 — Three groups is correct and meaningful. IDRiD behaves fundamentally differently from everything else. The fact that it forms its own group despite being geographically closest to APTOS (Indian population, similar equipment) is actually the most interesting finding in the whole analysis. It suggests the grouping isn't about population proximity — it's about something else. More on this below.
Observation 2 — DDR/EyePACS two-party system is accurate. The feature space for these datasets is being pulled between Class 0 and Class 4 as the two dominant attractors, with Classes 1, 2, 3 losing their distinct identity. This is exactly H1 — the DR-invariant features that survive distribution shift are the extreme ones (healthy vs proliferative), and the middle severity features collapse.
Observation 3 — Messidor single ruler is correct and the strongest H1 evidence in the data. Not two parties, just one. Everything points toward Class 0. This connects directly to the silent failure — the model isn't even uncertain because there's no competing signal from any other class.
Observation 6 — Extremes are distinguishable, middle classes struggle is partially right and worth sharpening.

Where to Push Back
Observation 5 — Messidor's collapse is because of data imbalance and missing Class 3 is partially right but misses the main driver.
The missing Class 3 contributes — there's no Class 3 centroid representation in Messidor, so that region of feature space is unanchored. But this doesn't explain why everything collapses to Class 0 specifically rather than spreading across the other classes. The real driver is the combination of two things: the Class 0 centroid in APTOS feature space is the largest and most dominant (1805 training samples vs 193 for Class 4), AND Messidor's scanner produces images that after EfficientNet feature extraction land closer to the APTOS Class 0 region than any other. The missing Class 3 makes it worse but isn't the primary cause.
Observation 6 needs splitting into two distinct claims:
The first claim — extremes are distinguishable because they're visually distinctive — is correct. Class 0 has a clear visual signature (clean retina, no lesions). Class 4 has a clear visual signature (neovascularization, large hemorrhages, extensive damage). These are visually far apart and the model learned robust representations for both on APTOS.
The second claim — middle classes cluster around extremes — needs more precision. What you're observing isn't just a smooth progression problem. It's two separate phenomena:
On APTOS itself, the confusion is bidirectional on middle classes — this is the genuine clinical ambiguity of adjacent grades. Classes 1, 2, 3 share overlapping visual features and even ophthalmologists disagree on these boundaries. This is a real property of the data.
On external datasets, the middle classes don't just spread between extremes — they specifically collapse toward Class 0. If it were purely a smooth progression issue, you'd expect them to land between Class 0 and Class 4 proportionally. Instead Classes 1, 2, 3 on DDR and Messidor land overwhelmingly near Class 0. That's not ambiguity — that's directional shift. The mild and moderate DR features from non-APTOS scanners don't look like mild and moderate DR to this model. They look like healthy retinas.

The IDRiD Anomaly Is the Key Insight
You called it a separate entity but didn't explain why. This is actually the most theoretically important observation.
IDRiD's heatmap is balanced — each true class has reasonably high similarity to its own centroid, not just Class 0. This means IDRiD's DR features transferred to APTOS's feature space. The model's learned representations for each severity grade are directionally aligned with the corresponding IDRiD images.
Why IDRiD specifically? Three reasons compounding: same Indian patient population as APTOS, similar clinical equipment, and identical label semantics. The distribution shift is in scanner-artifact dimensions (hence high Mahalanobis distance in EXP_011) but NOT in DR-feature dimensions (hence cosine similarity stays aligned per class). This is the clearest evidence you have for the benign vs malignant shift distinction.
The contrast is what makes it interesting: IDRiD is geometrically far from APTOS in feature space (high Mahalanobis distance) but directionally aligned per class (balanced cosine similarity). Messidor is geometrically close to APTOS (lower Mahalanobis distance after BG) but directionally collapsed toward Class 0 (cosine similarity dominated by Class 0). This directly and cleanly separates H1 from H2 and from H3 all at once.

Refined Hypotheses
Your two hypotheses are on the right track but here's a sharper formulation:
H1 confirmed — directional feature corruption, not just boundary mismatch. The cosine similarity heatmaps show that for DDR, EyePACS, and all Messidor groups, images from non-Class-0 true labels have their highest cosine similarity pointing toward Class 0. The features are genuinely in the wrong direction, not just near the wrong boundary. This rules out H2 as the primary explanation for Messidor and DDR.
Severity determines feature robustness across domains. Class 0 and Class 4 have distinctive enough visual signatures that their features survive distribution shift — the model can still recognize them directionally even on foreign scanners. Classes 1, 2, 3 have ambiguous, overlapping features that don't survive — under distribution shift they lose their directional identity and get absorbed into the majority class attractor. This explains the two-party system in DDR/EyePACS and the single-party collapse in Messidor.
IDRiD is the control case. It's the only dataset where DR features transferred per-class, which explains why it's the only dataset where uncertainty correctly increased and Certain+Wrong dropped. When features are directionally correct, the internal uncertainty signals work. When features are directionally corrupted (Messidor), internal signals fail.
