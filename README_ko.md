# AntiBinder
AntiBinder: 항체-항원 결합 관계를 예측하는 서열-구조 하이브리드 모델입니다. 양방향 교차-어텐션 메커니즘을 기반으로 하며, 이번 버전에서는 예측 정확도 향상을 위해 **중쇄(VH)**와 **경쇄(VL)** 데이터를 모두 지원합니다.

![framework](./figures/model_all.png)

## 소개
이 프로젝트는 단백질 유형의 항원-항체 친화도를 예측하는 데 사용됩니다. 모델은 서열 데이터만으로도 훈련 및 사용이 가능합니다. 또한, 모듈을 쌓아 모델의 파라미터를 크게 늘리고 플러그 앤 플레이 방식으로 훈련할 수도 있습니다. 업데이트된 모델은 항체의 VH 및 VL 영역을 모두 활용하여 더욱 포괄적인 결합 예측을 제공합니다.

## 필수 환경
python 3.11

## 설치 가이드
프로젝트 설치 및 설정에 대한 자세한 지침입니다:

### 저장소 복제
git clone https://github.com/brisingr10/Antibinder3.git

### 의존성 설치
pip install -r requirements.txt

## 사용 방법

### 1. 데이터 준비
훈련 또는 예측을 시작하기 전에, 원시 항체-항원 결합 데이터를 처리하여 필요한 영역을 추출하고 하나의 데이터셋으로 통합해야 합니다.

**단계:**

1.  **데이터 전처리:**
    `process_all_data.py` 스크립트를 사용하여 다음을 포함한 모든 필요한 데이터 전처리를 수행합니다:
    *   `datasets/raw_data/`에서 raw 데이터 읽기.
    *   알려진 데이터셋에 대한 내부 구성에 따라 열 이름 변경(예: `Heavy`를 `vh`로, `Light`를 `vl`로, `antigen`을 `Antigen Sequence`로, `Label`을 `ANT_Binding`으로).
    *   중쇄(VH) 및 경쇄(VL) 서열을 해당 프레임워크(FR) 및 상보성 결정 영역(CDR) 세그먼트(H-FR1, H-CDR1 등, L-FR1, L-CDR1 등)로 분할.
    *   모든 처리된 데이터 파일을 단일 데이터셋으로 결합.
    *   데이터 유효성 검사 수행(누락된 VH 또는 VL 서열이 있는 행 및 중복 행 제거).
    *   결합된 데이터를 훈련 및 검증을 위해 `combined_training_data.csv` 및 `test_data.csv`로 분할.

    `datasets/raw_data/`의 raw 데이터 CSV 파일에는 `process_all_data.py`가 예상하는 중쇄, 경쇄, 항원 서열 및 결합 라벨에 필요한 열이 포함되어 있는지 확인하십시오(각 데이터셋에 대한 특정 열 매핑은 스크립트의 `if __name__ == "__main__:"` 블록을 참조).

    ```bash
    python process_all_data.py
    # 이 스크립트는 datasets/raw_data/에서 데이터를 읽고, 중간 처리된 파일을 datasets/process_data/에 출력하며,
    # 최종 combined_training_data.csv 및 test_data.csv를 datasets/ 디렉토리에 저장합니다.
    ```

    **출력 데이터 구조:**
    `combined_training_data.csv` 및 `test_data.csv` 파일에는 다음 열이 포함됩니다:
    `vh`, `vl`, `Antigen Sequence`,
    `H-FR1`, `H-CDR1`, `H-FR2`, `H-CDR2`, `H-FR3`, `H-CDR3`, `H-FR4`,
    `L-FR1`, `L-CDR1`, `L-FR2`, `L-CDR2`, `L-FR3`, `L-CDR3`, `L-FR4`,
    `ANT_Binding`

### 2. 모델 훈련
데이터 준비 및 결합이 완료되면 `main_trainer.py`를 사용하여 AntiBinder 모델을 훈련할 수 있습니다.

```bash
python main_trainer.py \
    --batch_size 32 \
    --latent_dim 32 \
    --epochs 50 \
    --lr 1e-4 \
    --model_name AntiBinderV2 \
    --device 0 # CUDA 장치 사용 가능 시 지정 (예: 0, 1 등)

# 기타 선택적 인수:
# --no_cuda: CUDA 훈련 비활성화 (CPU 사용)
# --seed: 재현성을 위한 랜덤 시드 (기본값: 42)
```

*   **설정:** 모델 파라미터 및 아키텍처 설정(예: 중쇄 및 경쇄의 `max_position_embeddings`, 영역 유형 인덱싱)은 `cfg_ab.py` 파일에 정의되어 있습니다.
*   **데이터 로딩 및 임베딩:** `antigen_antibody_emb.py` 스크립트는 결합된 데이터를 로드하고, 항원에 대한 ESM 임베딩 및 항체 사슬에 대한 IgFold 구조 임베딩을 생성합니다. 또한 효율적인 데이터 검색을 위해 LMDB 캐시를 관리합니다.

### 3. 모델로 예측
훈련된 모델을 사용하여 예측하려면 `main_test.py`를 사용합니다.

```bash
python main_test.py \
    --input_path "path/to/your/prediction_data.csv" \
    --checkpoint_path "path/to/your/trained_model.pth" \
    --batch_size 64

# 기타 선택적 인수:
# --no_cuda: CUDA 비활성화 (CPU 사용)
# --seed: 재현성을 위한 랜덤 시드 (기본값: 42)
```

*   **예측을 위한 입력 데이터:** `--input_path`는 `vh`, `vl`, `Antigen Sequence` 열을 포함하는 CSV 파일을 가리켜야 합니다. 이 파일은 훈련 데이터와 동일한 방식으로 처리되어야 합니다(즉, `process_all_data.py`를 사용하여 FR/CDR 영역으로 분할되어야 합니다). `ANT_Binding` 열은 예측에 필수는 아니지만, 스크립트는 동일한 데이터 구조를 예상합니다.
*   **출력:** 스크립트는 `predictions/output/` 디렉토리에 새 CSV 파일을 생성하며, 입력 파일 이름에 `_results.csv`를 추가합니다. 이 파일에는 `predicted_probability` 및 `predicted_label` 열이 포함됩니다.

## 모델 아키텍처
핵심 모델 아키텍처는 `antibinder_model.py`에 정의되어 있으며, 다음을 포함합니다:
*   `Combine_Embedding`: 중쇄 및 경쇄 모두에 대한 서열 및 구조 임베딩의 결합을 처리합니다.
*   `BiCrossAttentionBlock`: 항체(결합된 VH+VL)와 항원 임베딩 간의 양방향 교차-어텐션 메커니즘을 구현합니다.
*   `AntiBinder`: 임베딩 결합, 교차-어텐션 및 최종 분류 계층을 총괄하는 주 모델 클래스입니다.

## 캐시 관리
ESM 및 IgFold 임베딩은 LMDB를 사용하여 캐시되므로 후속 실행 시 속도가 향상됩니다. `datasets/fold_emb/` 및 `antigen_esm/` 디렉토리 내에 중쇄 구조, 경쇄 구조 및 항원 ESM 임베딩을 위한 별도의 캐시 디렉토리가 유지 관리됩니다.