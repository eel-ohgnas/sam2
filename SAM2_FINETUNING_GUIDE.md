# SAM2 파인튜닝 프로젝트 가이드

## 프로젝트 개요

이 문서는 SAM2(Segment Anything Model 2)를 파인튜닝하고 테스트하는 전체 과정을 정리한 가이드입니다.

---

## 1. 환경 설정

### 1.1 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate
```

**목적**: 프로젝트별 독립적인 Python 환경을 만들어 패키지 충돌을 방지합니다.

**출처**: Python 공식 문서 - [venv 모듈](https://docs.python.org/3/library/venv.html)

---

### 1.2 SAM2 설치

```bash
# SAM2 저장소 클론
git clone https://github.com/facebookresearch/sam2.git

# 개발 모드로 설치
pip install -e .
```

**목적**:
- `git clone`: Meta의 공식 SAM2 코드를 로컬에 다운로드
- `pip install -e .`: 편집 가능 모드로 설치하여 코드 수정 시 재설치 불필요

**출처**: [SAM2 GitHub Repository](https://github.com/facebookresearch/sam2)

---

### 1.3 체크포인트 다운로드

```bash
# 체크포인트 폴더 생성
mkdir -p checkpoints

# SAM2.1 Hiera Small 모델 다운로드 (185MB)
curl -L -o checkpoints/sam2.1_hiera_small.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
```

**목적**: 사전 학습된 모델 가중치를 다운로드하여 파인튜닝의 시작점으로 사용합니다.

**출처**: [SAM2 Model Checkpoints](https://github.com/facebookresearch/sam2#download-checkpoints)

**모델 옵션**:
| 모델 | 크기 | 용도 |
|------|------|------|
| `sam2.1_hiera_tiny.pt` | 39MB | 빠른 추론, 제한된 리소스 |
| `sam2.1_hiera_small.pt` | 185MB | 균형잡힌 성능 (권장) |
| `sam2.1_hiera_base_plus.pt` | 324MB | 높은 정확도 |
| `sam2.1_hiera_large.pt` | 898MB | 최고 정확도 |

---

## 2. 데이터셋 준비

### 2.1 Kvasir-SEG 데이터셋 다운로드

```bash
# 데이터셋 다운로드
curl -L -o kvasir-seg.zip \
  "https://datasets.simula.no/downloads/kvasir-seg.zip"

# 압축 해제
unzip kvasir-seg.zip
```

**목적**: 의료 영상(위장관 폴립) 세그멘테이션을 위한 공개 데이터셋을 다운로드합니다.

**출처**: [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)

**데이터셋 구성**:
- 1,000개의 폴립 이미지
- 해당 이진 마스크
- 학습/검증 분할: 800/200

---

### 2.2 데이터 구조 정리

```bash
# 폴더 구조 생성
mkdir -p data/images/train data/images/val
mkdir -p data/masks/train data/masks/val

# 이미지 분할 (80% 학습, 20% 검증)
# Python 스크립트로 자동 분할
```

**목적**: SAM2 학습 스크립트가 요구하는 폴더 구조로 데이터를 정리합니다.

**폴더 구조**:
```
data/
├── images/
│   ├── train/    # 학습 이미지 (800개)
│   └── val/      # 검증 이미지 (200개)
└── masks/
    ├── train/    # 학습 마스크
    └── val/      # 검증 마스크
```

---

## 3. 파인튜닝

### 3.1 학습 스크립트 실행

```bash
source venv/bin/activate
python train_sam2_proper.py
```

**목적**: SAM2 모델을 Kvasir-SEG 데이터셋으로 파인튜닝합니다.

---

### 3.2 핵심 학습 코드 설명

#### 모델 로드
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 모델 빌드 (device 파라미터 중요!)
sam2_model = build_sam2(
    'configs/sam2.1/sam2.1_hiera_s.yaml',  # 모델 설정
    'checkpoints/sam2.1_hiera_small.pt',    # 체크포인트
    device='mps'  # Apple Silicon: 'mps', NVIDIA: 'cuda', CPU: 'cpu'
)
predictor = SAM2ImagePredictor(sam2_model)
```

**출처**: [sagieppel의 60줄 파인튜닝 코드](https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code)

**핵심 포인트**:
- `device` 파라미터를 명시적으로 전달해야 함 (MPS 환경에서 CUDA 에러 방지)
- 설정 파일은 전체 경로 사용: `'configs/sam2.1/sam2.1_hiera_s.yaml'`

---

#### 학습 모드 설정
```python
# Mask Decoder와 Prompt Encoder만 학습
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)
# Image Encoder는 고정 (메모리 절약)
```

**목적**:
- 전체 모델 대신 일부만 학습하여 메모리 사용량 감소
- 사전 학습된 이미지 인코더의 강력한 특징 추출 능력 유지

---

#### Forward Pass (Gradient 전파 핵심)
```python
# 이미지 인코딩
predictor.set_image(image)

# Prompt 준비
mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
)

# Prompt 인코딩
sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
    points=(unnorm_coords, labels),
    boxes=None,
    masks=None,
)

# Mask Decoder (핵심!)
high_res_features = [feat_level[-1].unsqueeze(0)
                     for feat_level in predictor._features["high_res_feats"]]

low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=True,
    repeat_image=batched_mode,
    high_res_features=high_res_features,
)
```

**핵심 포인트**:
- `predictor.predict()` 대신 내부 컴포넌트를 직접 호출해야 Gradient 전파 가능
- `predictor._features`를 통해 인코딩된 이미지 특징에 접근

**출처**: sagieppel의 코드에서 이 접근법을 발견. 공식 SAM2 학습 코드(A100 80GB 필요)의 대안입니다.

---

#### 손실 함수
```python
# 마스크 업스케일
prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

# Ground Truth
gt_mask = torch.tensor(mask.astype(np.float32), device=device)

# Sigmoid로 확률 변환
prd_mask = torch.sigmoid(prd_masks[:, 0])

# Binary Cross Entropy Loss
seg_loss = (
    -gt_mask * torch.log(prd_mask + 1e-5)
    - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)
).mean()

# IoU Score Loss
inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1, 2))
union = gt_mask.sum(dim=(1, 2)) + (prd_mask > 0.5).sum(dim=(1, 2)) - inter
iou = inter / (union + 1e-5)
score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

# 총 손실
loss = seg_loss + score_loss * 0.05
```

**목적**:
- `seg_loss`: 픽셀 단위 마스크 정확도
- `score_loss`: 모델의 자체 신뢰도 점수와 실제 IoU 일치도

---

## 4. 모델 비교 및 평가

### 4.1 비교 스크립트 실행

```bash
python compare_models.py
```

**출력 예시**:
```
┌─────────────────────────────────────────┐
│           평균 IoU 비교                  │
├─────────────────────────────────────────┤
│  원본 SAM2:      65.92%               │
│  파인튜닝 SAM2:  83.33%               │
├─────────────────────────────────────────┤
│  🚀 개선율:      +17.4%p              │
└─────────────────────────────────────────┘
```

---

### 4.2 IoU (Intersection over Union) 계산

```python
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-6)
```

**목적**: 세그멘테이션 품질을 정량적으로 측정하는 표준 지표입니다.

**해석**:
- 1.0 (100%): 완벽한 일치
- 0.5 (50%): 절반 일치
- 0.0 (0%): 전혀 일치하지 않음

---

## 5. 테스트 및 추론

### 5.1 좌표 지정 방식

```bash
python test_my_image.py --image test_images/dog.jpg --point "320,200"
```

**파라미터**:
- `--image`: 테스트할 이미지 경로
- `--point`: 세그멘트할 위치 좌표 (x,y)

---

### 5.2 인터랙티브 방식

```bash
python test_my_image.py --image test_images/dog.jpg
```

**사용법**:
- 좌클릭: 전경 포인트 추가 (세그멘트할 영역)
- 우클릭: 배경 포인트 추가 (제외할 영역)
- 창 닫기: 세그멘테이션 실행

---

### 5.3 원본 모델과 비교

```bash
# 파인튜닝 모델 (기본)
python test_my_image.py --image test_images/dog.jpg --point "320,200"

# 원본 모델
python test_my_image.py --image test_images/dog.jpg --point "320,200" --original
```

---

## 6. 생성된 파일 구조

```
sam2/
├── checkpoints/
│   └── sam2.1_hiera_small.pt      # 원본 체크포인트 (185MB)
├── data/
│   ├── images/
│   │   ├── train/                  # 학습 이미지 800개
│   │   └── val/                    # 검증 이미지 200개
│   └── masks/
│       ├── train/                  # 학습 마스크
│       └── val/                    # 검증 마스크
├── output/
│   ├── sam2_best.pt               # 파인튜닝 모델 (176MB)
│   ├── sam2_final.pt              # 최종 모델
│   ├── comparison/                 # 비교 이미지
│   └── my_results/                 # 테스트 결과
├── test_images/                    # 테스트용 샘플 이미지
├── train_sam2_proper.py           # 파인튜닝 스크립트
├── compare_models.py              # 모델 비교 스크립트
└── test_my_image.py               # 테스트 스크립트
```

---

## 7. 문제 해결 (Troubleshooting)

### 7.1 Config 파일을 찾을 수 없음

```
MissingConfigException: Cannot find primary config 'sam2.1_hiera_s'
```

**해결**: 전체 경로 사용
```python
# ❌ 잘못된 방법
config = 'sam2.1_hiera_s'

# ✅ 올바른 방법
config = 'configs/sam2.1/sam2.1_hiera_s.yaml'
```

---

### 7.2 CUDA 컴파일 에러 (Apple Silicon)

```
AssertionError: Torch not compiled with CUDA enabled
```

**해결**: device 파라미터 명시
```python
# ❌ 잘못된 방법
sam2_model = build_sam2(config, checkpoint)

# ✅ 올바른 방법
sam2_model = build_sam2(config, checkpoint, device='mps')
```

---

### 7.3 Gradient가 흐르지 않음 (Loss=0.0000)

**원인**: `predictor.predict()`는 `torch.no_grad()` 컨텍스트에서 실행됨

**해결**: 내부 컴포넌트를 직접 호출 (섹션 3.2 참조)

---

## 8. Windows 호환성 가이드

### 8.1 호환성 분석 결과

현재 코드는 macOS (Apple Silicon)에서 개발되었으며, Windows 호환성을 분석한 결과는 다음과 같습니다.

| 문제 유형 | 심각도 | 결론 |
|-----------|--------|------|
| MPS 디바이스 (Apple Silicon 전용) | 낮음 | 자동으로 CUDA/CPU로 전환됨 |
| 경로 구분자 (`/` vs `\`) | 낮음 | Python에서 `/`는 Windows에서도 동작 |
| `curl` 명령어 의존 | 중간 | Windows 10 이상에서 기본 포함 |
| 콘솔 한글/이모지 출력 | 중간 | PowerShell 또는 Windows Terminal 사용 권장 |

---

### 8.2 디바이스 자동 선택

코드에 이미 플랫폼별 분기가 포함되어 있어 Windows에서도 자동으로 동작합니다.

```python
# 모든 스크립트에서 동일한 패턴
if torch.backends.mps.is_available():     # macOS → True
    device = "mps"
elif torch.cuda.is_available():            # NVIDIA GPU → True
    device = "cuda"
else:                                      # GPU 없음 → True
    device = "cpu"
```

| 환경 | 선택되는 디바이스 | 비고 |
|------|-------------------|------|
| macOS (M1/M2/M3/M4) | `mps` | Apple Silicon GPU |
| Windows + NVIDIA GPU | `cuda` | CUDA 설치 필요 |
| Windows (GPU 없음) | `cpu` | 학습 속도 느림 |

---

### 8.3 Windows 실행 방법

#### Step 1: 사전 준비

- [Python 3.10+](https://www.python.org/downloads/) 설치 시 **"Add to PATH" 체크 필수**
- [Git](https://git-scm.com/download/win) 설치
- (선택) [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 설치

```powershell
# PowerShell에서 설치 확인
python --version
git --version
nvidia-smi           # NVIDIA GPU 확인 (선택)
```

#### Step 2: 프로젝트 셋업

```powershell
# 저장소 클론
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate          # ← macOS와 다른 부분!

# SAM2 및 의존성 설치
pip install -e .
pip install opencv-python matplotlib tqdm
```

> **주의**: macOS에서는 `source venv/bin/activate`, Windows에서는 `venv\Scripts\activate`

#### Step 3: CUDA 지원 PyTorch 설치 (NVIDIA GPU 사용 시)

```powershell
# CUDA 버전 확인
nvidia-smi

# CUDA 12.1 기준 PyTorch 재설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Step 4: 체크포인트 다운로드

```powershell
mkdir checkpoints
curl -L -o checkpoints\sam2.1_hiera_small.pt ^
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
```

#### Step 5: 데이터셋 다운로드 및 학습

```powershell
python download_dataset.py
python train_sam2_proper.py
```

#### Step 6: 테스트

```powershell
python test_my_image.py --image test_images\dog.jpg --point "320,200"
python compare_models.py
```

---

### 8.4 macOS vs Windows 명령어 비교

| 작업 | macOS (Terminal) | Windows (PowerShell) |
|------|------------------|---------------------|
| 가상환경 활성화 | `source venv/bin/activate` | `venv\Scripts\activate` |
| 폴더 생성 | `mkdir -p checkpoints` | `mkdir checkpoints` |
| 파일 다운로드 | `curl -L -o file url` | `curl -L -o file url` |
| 경로 구분자 | `/` | `\` (Python에서 `/`도 가능) |
| GPU | MPS (Apple Silicon) | CUDA (NVIDIA) |
| Mixed Precision | 미지원 (MPS) | 지원 (CUDA) |
| 권장 터미널 | Terminal / iTerm2 | Windows Terminal / PowerShell |

---

### 8.5 Windows 문제 해결

#### 한글/이모지가 깨져서 출력될 때

```powershell
# PowerShell에서 UTF-8 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

또는 **Windows Terminal** 앱을 사용하면 기본적으로 UTF-8을 지원합니다.

#### curl이 없다는 에러

Windows 10 (1803 이상)에는 curl이 기본 포함되어 있습니다. 이전 버전이라면:

```powershell
# 브라우저에서 직접 다운로드
# https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
# → checkpoints 폴더에 저장
```

#### CUDA 관련 에러

```
RuntimeError: No CUDA GPUs are available
```

NVIDIA GPU가 없는 PC에서는 CPU로 동작합니다. 학습 속도가 느리지만 기능은 동일합니다.
GPU 없이 테스트만 하려면 `test_my_image.py`를 사용하세요.

---

## 9. 참고 자료

| 리소스 | 설명 | 링크 |
|--------|------|------|
| SAM2 공식 저장소 | Meta의 SAM2 코드 | [GitHub](https://github.com/facebookresearch/sam2) |
| 60줄 파인튜닝 | sagieppel의 간단한 파인튜닝 코드 | [GitHub](https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code) |
| Kvasir-SEG | 의료 영상 데이터셋 | [Simula](https://datasets.simula.no/kvasir-seg/) |
| SAM2 논문 | 모델 아키텍처 상세 설명 | [arXiv](https://arxiv.org/abs/2408.00714) |

---

## 10. 핵심 요약

| 단계 | 명령어 | 목적 |
|------|--------|------|
| 환경 설정 | `source venv/bin/activate` | 가상환경 활성화 |
| 학습 | `python train_sam2_proper.py` | 파인튜닝 실행 |
| 비교 | `python compare_models.py` | 성능 비교 |
| 테스트 | `python test_my_image.py --image <path>` | 추론 실행 |

**핵심 성과**:
- 원본 SAM2 IoU: 65.92%
- 파인튜닝 SAM2 IoU: 83.33%
- **개선율: +17.4%p**
