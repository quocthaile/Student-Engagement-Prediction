# Model training: cach su dung train, valid, test

Tai lieu nay mo ta rieng giai doan huan luyen mo hinh trong pipeline hien tai cua du an, tap trung vao cach ba tap `train_smote.csv`, `valid_original.csv`, `test_original.csv` duoc tao ra va duoc dung trong model training/evaluation.

## 1. Cac file du lieu lien quan

Pipeline thuc nghiem dung cac file trong `dataset/model_data/`:

| File | Vai tro | Cach tao |
| --- | --- | --- |
| `train_smote.csv` | Tap train dung de fit mo hinh | Sinh tu 80% du lieu goc, sau do chi tap train duoc ap dung SMOTE/SMOTENC |
| `valid_original.csv` | Tap validation de chon mo hinh va hieu chinh | Sinh tu 10% du lieu goc, giu phan phoi that, khong SMOTE |
| `test_original.csv` | Tap test cuoi de bao cao ket qua sau cung | Sinh tu 10% du lieu goc, giu phan phoi that, khong SMOTE |

So lieu hien tai trong repo:

| Tap | So dong | Phan phoi nhan |
| --- | ---: | --- |
| Train sau SMOTE | 105,684 | High: 35,228; Low: 35,228; Medium: 35,228 |
| Valid goc | 12,952 | Medium: 4,404; Low: 4,274; High: 4,274 |
| Test goc | 12,952 | Medium: 4,404; High: 4,274; Low: 4,274 |

Luu y: README hien tai co cho vi du kich thuoc 60K/35K/35K, nhung artifact dang nam trong repo cho thay kich thuoc thuc te la 105,684/12,952/12,952.

## 2. Stage 3 tao train/valid/test nhu the nao

File chinh: `experiment/stage_3_split_and_smote.py`.

Quy trinh:

1. Doc bang dac trung da gan nhan tu `dataset/user_features_and_wes.csv`.
2. Kiem tra cot dich `target_label`.
3. Chuan hoa mot so cot dau vao nhu `age`, `gender_encoded`, `school_encoded`.
4. Chia stratified 8:1:1:
   - lan 1: `train_test_split(..., test_size=0.2, stratify=df["target_label"])` tao `df_train` 80% va `df_temp` 20%;
   - lan 2: chia `df_temp` thanh `df_valid` 10% va `df_test` 10%, tiep tuc stratify theo `target_label`.
5. Fit `LabelEncoder` theo thu tu `["Low_Engagement", "Medium_Engagement", "High_Engagement"]` va luu `label_encoder.pkl`.
6. Tao ma tran dac trung cho train/valid/test bang cung danh sach 12 feature:
   - `school_encoded`
   - `seq`
   - `speed`
   - `rep_counts`
   - `cmt_counts`
   - `age`
   - `gender_encoded`
   - `num_courses`
   - `attempts_4w`
   - `is_correct_4w`
   - `score_4w`
   - `accuracy_rate_4w`
7. Chi ap dung SMOTE/SMOTENC tren `X_train`, `y_train`.
8. Luu:
   - train da can bang vao `train_smote.csv`;
   - valid/test goc vao `valid_original.csv`, `test_original.csv`, co giu them metadata nhu `school`, `year_of_birth`, `gender`.

Y nghia quan trong: valid va test khong bi SMOTE. Dieu nay dung, vi validation/test phai phan anh du lieu that. SMOTE chi duoc phep tac dong vao train de mo hinh hoc duoc lop thieu.

## 3. Stage 4 dung train/valid/test nhu the nao

File chinh: `experiment/stage_4_model_training_eval.py`.

Quy trinh hien tai:

1. Nap ca ba tap tu stage 3:
   - `X_train`, `y_train`
   - `X_valid`, `y_valid`
   - `X_test`, `y_test`
2. Lay `feature_columns` tu cot cua `train_smote.csv` tru `target_label`.
3. Reindex valid/test theo dung cot train. Cac cot metadata khong co trong train se bi loai bo, cac cot thieu duoc fill 0.
4. Huan luyen 5 thuat toan tren train:
   - Logistic Regression dang `OneVsRestClassifier`
   - Linear SVC
   - Decision Tree
   - Random Forest
   - XGBoost
5. Moi mo hinh sau khi fit train se duoc danh gia tren valid.
6. Ket qua valid duoc ghi vao `experiment/deployment_models/evaluation_metrics.csv`.
7. Chon best model theo thu tu uu tien:
   - `Recall_Low_Engagement`
   - `F1_Macro`
   - `Balanced_Accuracy`
   - `Accuracy`
8. Chi sau khi da chon best model moi chay final evaluation tren test.
9. Ket qua test duoc ghi vao:
   - `final_test_metrics.csv`
   - `<BestModel>_test_predictions.csv`
   - `best_model_test_classification_report.csv`
   - confusion matrix trong `experiment/output_images_4w/`
10. Dong goi mo hinh vao:
   - `best_model_4w.pkl`
   - `deployment_bundle.pkl`
   - `best_model_metadata.json`

Ket qua hien tai:

| Tap | Model | Accuracy | Balanced Accuracy | F1 Macro | Recall Low | Precision Low | AUC OVR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Valid | Logistic Regression | 0.6150 | 0.6122 | 0.5045 | 0.0108 | 0.2044 | 0.7801 |
| Test | Logistic Regression | 0.6177 | 0.6149 | 0.5064 | 0.0098 | 0.2049 | 0.7811 |

Diem can nhan manh: trong pipeline thuc nghiem, test dang duoc dung dung vai tro la tap danh gia cuoi cung. Viec chon Logistic Regression den tu valid, khong den tu test.

## 4. Khac biet voi `dashboard/trainer.py`

File `dashboard/trainer.py` co pipeline rieng de phuc vu dashboard:

1. Doc `train_smote.csv` va `test_original.csv`.
2. Co khai bao `VALID_PATH` nhung khong su dung `valid_original.csv`.
3. Fit `SimpleImputer` va `StandardScaler` tren train.
4. Train mot model theo lua chon UI.
5. Danh gia truc tiep tren test.
6. Luu bundle va metadata vao `dashboard/models/`.

Viem de: neu dung dashboard de thu qua nhieu model, doi tham so, xem ket qua test roi chon model, thi test da bi bien thanh validation ngầm. Khi do ket qua test khong con la uoc luong khach quan cho kha nang tong quat hoa.

Voi muc tieu bao cao thuc nghiem, nen xem `experiment/stage_4_model_training_eval.py` la pipeline chinh. `dashboard/trainer.py` nen chi dung de demo hoac can sua lai de danh gia tren valid truoc, test sau.

## 5. Van de hien tai trong ket qua

Ket qua moi nhat cho thay Accuracy khoang 61-62%, AUC khoang 0.78, nhung `Recall_Low_Engagement` rat thap:

- valid: 0.0108
- test: 0.0098

Voi bai toan canh bao som sinh vien co nguy co, recall lop `Low_Engagement` rat quan trong. Recall gan 1% co nghia la trong 100 sinh vien Low thuc su, mo hinh chi bat duoc khoang 1 sinh vien. Vi vay, du accuracy nhin kha on, model chua dat yeu cau nghiep vu neu muc tieu la canh bao som.

Nguyen nhan co the:

1. Tieu chi label va feature 4 tuan dau chua tach lop Low du manh.
2. Mo hinh dang toi uu bien quyet dinh mac dinh, chua hieu chinh nguong cho muc tieu recall Low.
3. SMOTE lam can bang train nhung chua chac giup mo hinh uu tien Low tren phan phoi that.
4. Tieu chi chon model uu tien Recall Low, nhung cac ung vien deu co Recall Low qua thap, nen best model chi la "it kem nhat" theo metric nay.

## 6. De xuat cach chay lai va hieu chinh dung vai tro valid/test

Nguyen tac:

- Train: chi de fit model va fit cac bien doi co hoc tham so.
- Valid: de chon model, chon hyperparameter, chon class weight, chon threshold, chon cau hinh label/SMOTE.
- Test: chi dung mot lan sau cung de xac nhan ket qua cua pipeline da khoa.

### Buoc 1: Chay lai split neu thay doi label/feature/SMOTE

Neu chi sua model o stage 4 thi khong can chay lai stage 1-3. Neu thay doi cach gan nhan, cua so 4w, SMOTE, feature set thi can chay lai tu stage lien quan.

Lenh chay lai tu split den training:

```bash
python experiment/run_pipeline.py --from-step 3 --to-step 4
```

Lenh chay lai toan bo pipeline 1-4:

```bash
python experiment/run_pipeline.py --from-step 1 --to-step 4
```

Lenh chay auto-search cac preset hien co:

```bash
python experiment/run_pipeline.py --auto-select-config
```

Luu y ky thuat: `run_pipeline.py --auto-select-config` co thu nhieu preset trong `config.py`, nhung stage 3 hien tai chi dung `ENABLE_SMOTE`; cac tham so `TRAIN_CLASS_RATIOS`, `TRAIN_TARGET_TOTAL_SAMPLES`, `MAX_TRAIN_SAMPLES_PER_CLASS` dang duoc khai bao trong config nhung chua that su duoc ap dung trong `stage_3_split_and_smote.py`. Neu muon auto-search ti le train co y nghia, can sua stage 3 de dung cac tham so nay.

### Buoc 2: Hieu chinh tren valid, khong dung test

Nen thu cac huong sau va moi lan chi so sanh bang valid:

1. Class weight:
   - Logistic Regression: thu `class_weight="balanced"` hoac dict tang trong so cho `Low_Engagement`.
   - Linear SVC: thu `class_weight="balanced"`.
   - Random Forest da co `class_weight="balanced"` nhung co the thu `balanced_subsample`.
2. Hyperparameter:
   - Logistic Regression: thu `C`, `penalty`, `solver`, `max_iter`.
   - Decision Tree/Random Forest/XGBoost: thu `max_depth`, `min_samples_leaf`, `n_estimators`, `learning_rate`.
3. Threshold/risk policy:
   - Với model co `predict_proba` hoac `decision_function`, khong bat buoc phai dung `argmax`.
   - Co the quy dinh neu score/probability cua `Low_Engagement` vuot nguong `t` thi gan Low.
   - Quet `t` tren valid va chon nguong dat recall Low muc tieu, vi du 0.50, 0.70, dong thoi kiem soat precision va false positive.
4. Selection metric:
   - Neu muc tieu nghiep vu la canh bao som, nen dung `Recall_Low_Engagement` hoac `F_beta` voi beta > 1 cho lop Low.
   - Khong nen de Accuracy quyet dinh model neu recall Low la yeu cau chinh.
5. Feature/label:
   - Thu them feature xu huong theo thoi gian: so lan hoc moi tuan, do giam/tang attempts, ngay hoc gan nhat, so ngay active.
   - Xem lai cach dinh nghia `Low_Engagement`: neu Low la nhom can canh bao, label nen phu hop voi nguy co bo hoc/ket qua kem, khong chi la phan vi tong hop.

Moi cau hinh nen ghi lai:

- model name;
- hyperparameter;
- threshold neu co;
- valid Accuracy;
- valid Recall Low;
- valid Precision Low;
- valid F1 Low;
- valid F1 Macro;
- valid Balanced Accuracy.

### Buoc 3: Khoa cau hinh tot nhat theo valid

Sau khi co ket qua valid, chon cau hinh theo mot quy tac ro rang, vi du:

1. Loc cac model co `Recall_Low_Engagement >= 0.60`.
2. Trong nhom do, chon model co `Precision_Low_Engagement` cao nhat.
3. Neu khong co model nao dat recall muc tieu, chon model co `F2_Low_Engagement` cao nhat va ghi ro chua dat yeu cau canh bao som.

Quy tac nay nen duoc ghi trong code va bao cao de tranh viec chon model theo cam tinh.

### Buoc 4: Chay test duy nhat sau khi da khoa model

Khi da khoa:

- feature set;
- cach label;
- SMOTE/class weight;
- model class;
- hyperparameter;
- threshold;
- tieu chi chon model;

moi chay test:

```bash
python experiment/stage_4_model_training_eval.py
```

Neu stage 4 da duoc sua de chay nhieu cau hinh, can dam bao no khong dung test trong vong lap chon model. Test chi nen duoc dung o doan cuoi, tuong tu logic hien tai o buoc `[3/4]`.

Sau test, bao cao cac file:

- `experiment/deployment_models/final_test_metrics.csv`
- `experiment/deployment_models/best_model_test_classification_report.csv`
- `experiment/deployment_models/<BestModel>_test_predictions.csv`
- `experiment/output_images_4w/CM_TEST_<BestModel>.png`

## 7. De xuat sua code de ho tro hieu chinh tot hon

### 7.1 Them evaluation tren valid va test cho dashboard

`dashboard/trainer.py` nen nap ca valid va test:

- train model tren train;
- hien metric valid de nguoi dung so sanh model;
- chi tao metric test khi bam nut "final evaluate" hoac khi model da duoc chon;
- metadata nen tach `validation_metrics` va `test_metrics`.

Như vậy dashboard khong vo tinh khuyen khich chon model theo test.

### 7.2 Them threshold tuning trong stage 4

Nen bo sung ham:

```python
def tune_low_threshold(model, X_valid, y_valid, label_encoder, candidate_thresholds):
    ...
```

Ham nay quet threshold cho lop `Low_Engagement` tren valid. Sau khi chon threshold, luu vao `best_model_metadata.json` va ap dung dung threshold do khi danh gia test va khi predict production.

### 7.3 Sua stage 3 neu muon dung `TRAIN_CLASS_RATIOS`

Hien `config.py` co `TRAIN_CLASS_RATIOS`, `TRAIN_TARGET_TOTAL_SAMPLES`, `MAX_TRAIN_SAMPLES_PER_CLASS`, nhung stage 3 dang dung SMOTE mac dinh theo cach can bang ve lop lon nhat. Neu bao cao noi da thu cac ti le train khac nhau thi can sua stage 3 de:

1. Tinh so mau muc tieu moi lop tu `TRAIN_CLASS_RATIOS`.
2. Oversample/undersample train theo so mau do.
3. Khong thay doi valid/test.
4. Ghi phan phoi train sau resampling vao log/artifact.

## 8. Ket luan ngan gon

Pipeline thuc nghiem hien tai da dung vai tro train/valid/test tuong doi dung:

- train duoc SMOTE va dung de fit model;
- valid duoc dung de chon model;
- test duoc dung de danh gia cuoi.

Van de chinh khong phai la test leakage trong `experiment/stage_4_model_training_eval.py`, ma la:

1. dashboard dang danh gia truc tiep tren test va khong dung valid;
2. Recall Low qua thap nen model chua phu hop muc tieu canh bao som;
3. cac tham so train ratio trong config chua duoc stage 3 ap dung that su;
4. chua co buoc threshold tuning tren valid cho lop `Low_Engagement`.

Huong chay lai nen la: lap lai train/hieu chinh tren valid, khoa cau hinh, roi moi bao cao test mot lan. Tuyet doi khong dung test de chon model, chon threshold, hay chon preset.

## 9. Cap nhat sau khi sua pipeline

Pipeline `experiment/stage_4_model_training_eval.py` da duoc sua theo dung nguyen tac tren:

1. Mo rong danh sach ung vien model/hyperparameter:
   - Logistic Regression C1
   - Logistic Regression C0.3/C1/C3 voi `class_weight="balanced"`
   - Linear SVC mac dinh va balanced
   - Decision Tree mac dinh va balanced
   - Random Forest balanced
   - Random Forest Deep Balanced
   - XGBoost
2. Moi ung vien chi duoc fit tren `train_smote.csv`.
3. Moi ung vien duoc tune threshold cho lop `Low_Engagement` tren `valid_original.csv`.
4. Selection score moi:
   - uu tien ung vien dat `Recall_Low_Engagement >= 0.60`;
   - trong cac ung vien dat muc recall, uu tien `Precision_Low_Engagement`;
   - tiep theo moi xet `F2_Low_Engagement`, `Recall_Low_Engagement`, `F1_Macro`, `Balanced_Accuracy`, `Accuracy`.
5. Test chi duoc chay sau khi da chon xong model va threshold tu validation.
6. `deployment_bundle.pkl` va `best_model_metadata.json` da luu them:
   - `decision_policy`
   - `low_threshold`
   - `low_class_index`
   - `score_source`

Ket qua chay lai stage 4 sau khi sua:

| Tap | Model | Policy | Threshold | Accuracy | Balanced Acc | F1 Macro | Precision Low | Recall Low | F2 Low | AUC OVR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Valid | XGBoost | `low_threshold_predict_proba` | 0.35 | 0.5751 | 0.5809 | 0.4683 | 0.4519 | 0.7501 | 0.6627 | 0.7952 |
| Test | XGBoost | `low_threshold_predict_proba` | 0.35 | 0.5764 | 0.5822 | 0.4701 | 0.4503 | 0.7550 | 0.6650 | 0.7958 |

So voi baseline cu:

| Phien ban | Model | Test Accuracy | Test Precision Low | Test Recall Low | Test F2 Low |
| --- | --- | ---: | ---: | ---: | ---: |
| Truoc khi sua | Logistic Regression mac dinh | 0.6177 | 0.2049 | 0.0098 | chua tinh |
| Sau khi sua | XGBoost + threshold Low 0.35 | 0.5764 | 0.4503 | 0.7550 | 0.6650 |

Dien giai: accuracy giam tu 61.77% xuong 57.64%, nhung Recall Low tang tu 0.98% len 75.50%. Voi bai toan canh bao som, day la trade-off hop ly hon vi model bat duoc phan lon sinh vien nguy co thay vi gan nhu bo sot toan bo lop Low.
