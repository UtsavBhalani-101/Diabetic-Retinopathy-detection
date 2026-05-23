train dataset


QWK on new dataset: 0.7579
[[64 65  1  2  2]
 [ 7 12  1  0  0]
 [ 5 37 87  0  7]
 [ 0  4 48  2 20]
 [ 0  0  9 11 29]]
Mean entropy      : 0.9194
Mean margin       : 0.4023
Uncertain fraction: 0.4600
Certain + Wrong (dangerous): 85
Certain + Right (ideal)     : 103
Uncertain + Wrong (caught)  : 134
Uncertain + Right (over-ref): 91
Class 0 (No DR) ECE: 0.1410
Class 1 (Mild) ECE: 0.2058
Class 2 (Moderate) ECE: 0.0797
Class 3 (Severe) ECE: 0.1030
Class 4 (Proliferative) ECE: 0.0457


test dataset:

Final QWK on Official IDRiD Test Set: 0.6238

Test Set Confusion Matrix:
[[ 7 21  6  0  0]
 [ 1  4  0  0  0]
 [ 1  8 22  1  0]
 [ 0  2  9  4  4]
 [ 0  2  5  0  6]]

Mean entropy      : 0.9964
Mean margin       : 0.3603
Uncertain fraction: 0.5534

--- Four Quadrant Uncertainty Breakdown ---
Certain + Wrong (dangerous): 23
Certain + Right (ideal)     : 21
Uncertain + Wrong (caught)  : 37
Uncertain + Right (over-ref): 22
Class 0 (No DR) ECE: 0.2323
Class 1 (Mild) ECE: 0.2553
Class 2 (Moderate) ECE: 0.1224
Class 3 (Severe) ECE: 0.0996
Class 4 (Proliferative) ECE: 0.0583
