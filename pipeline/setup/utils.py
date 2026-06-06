DATASET_REGISTRY = {
    "IDRiD": {
        "image_path": r"/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading/1. Original Images/a. Training Set",
        "target_path": r"/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        # official IDRiD test set (separate from training set)
        "test_image_path": r"/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set",
        "test_target_path": r"/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
        "image_col": "Image name",
        "diagnosis_col": "Retinopathy grade",
        "extension": ".jpg",
        "num_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    },
    "DDR-China": {
        "image_path": r"/kaggle/input/datasets/mariaherrerot/ddrdataset/DR_grading/DR_grading",
        "target_path": r"/kaggle/input/datasets/mariaherrerot/ddrdataset/DR_grading.csv",
        "image_col": "id_code",
        "diagnosis_col": "diagnosis",
        "extension": "",
        "num_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    },
    "Messidor-Grp1": {
        "image_path": r"/kaggle/input/datasets/antiti/messidor-grp-dataset/messidor_grp/messidor_grp/grp_1/images",
        "target_path": r"/kaggle/input/datasets/antiti/messidor-grp-dataset/messidor_grp/messidor_grp/grp_1/base1.csv",
        "image_col": "Image name",
        "diagnosis_col": "Retinopathy grade",
        "extension": ".tif",  # Adjust extension if Messidor uses tif/png instead of jpg
        "num_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    },
    "Messidor-Grp2": {
        "image_path": r"/kaggle/input/datasets/antiti/messidor-grp-dataset/messidor_grp/messidor_grp/grp_2/images",
        "target_path": r"/kaggle/input/datasets/antiti/messidor-grp-dataset/messidor_grp/messidor_grp/grp_2/base2.csv",
        "image_col": "Image name",
        "diagnosis_col": "Retinopathy grade",
        "extension": ".tif",
        "num_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    },
    "Messidor-Grp3": {
        "image_path": r"/kaggle/input/datasets/antiti/messidor-grp-dataset/messidor_grp/messidor_grp/grp_3/images",
        "target_path": r"/kaggle/input/datasets/antiti/messidor-grp-dataset/messidor_grp/messidor_grp/grp_3/base3.csv",
        "image_col": "Image name",
        "diagnosis_col": "Retinopathy grade",
        "extension": ".tif",
        "num_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    },
    "EyePACS-Resized": {
        "image_path": r"/kaggle/input/datasets/dreamer07/eyepacs/data/data",
        "target_path": r"/kaggle/input/datasets/dreamer07/eyepacs/trainLabels.csv/trainLabels.csv", 
        "image_col": "image",
        "diagnosis_col": "level",
        "extension": ".jpg",
        "num_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    },
    "APTOS_2019": {
        "image_path": r"/kaggle/input/competitions/aptos2019-blindness-detection/train_images",
        "target_path": r"/kaggle/input/competitions/aptos2019-blindness-detection/train.csv",
        "image_col": "id_code",         # APTOS uses id_code
        "diagnosis_col": "diagnosis",   # APTOS uses diagnosis
        "extension": ".png",             # APTOS images are saved natively as .png
        "num_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        # real label distribution from train.csv — used for wandb config logging
        "class_distribution": {"0": 1805, "1": 370, "2": 999, "3": 193, "4": 295}
    }
}