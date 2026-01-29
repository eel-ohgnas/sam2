# SAM2 파인튜닝 프로젝트 가이드

## 프로젝트 개요

SAM2(Segment Anything Model 2)를 Kvasir-SEG 데이터셋으로 파인튜닝하고 테스트하는 프로젝트입니다.

**핵심 성과**: 원본 IoU 65.92% → 파인튜닝 IoU 83.33% (**+17.4%p 개선**)

---

## 프로젝트 구조

```
sam2/                              ← 프로젝트 루트 (GitHub: eel-ohgnas/sam2)
├── train_sam2_proper.py           ← 파인튜닝 스크립트 (메인)
├── compare_models.py              ← 원본 vs 파인튜닝 비교
├── test_my_image.py               ← 이미지 테스트 도구
├── download_dataset.py            ← Kvasir-SEG 다운로드
├── prepare_data.py                ← 데이터 전처리
├── SAM2_FINETUNING_GUIDE.md       ← 이 문서
├── .gitignore
│
├── checkpoints/                   ← 원본 모델 (Git 제외)
│   └── sam2.1_hiera_small.pt      (185MB)
├── output/                        ← 학습 결과물 (Git 제외)
│   ├── sam2_best.pt               (176MB, 파인튜닝 모델)
│   ├── sam2_final.pt
│   └── comparison/                (비교 이미지)
├── data/                          ← 데이터셋 (Git 제외)
│   ├── images/train/              (800개)
│   ├── images/val/                (200개)
│   ├── masks/train/
│   └── masks/val/
├── test_images/                   ← 테스트 이미지 (Git 제외)
├── venv/                          ← Python 가상환경 (Git 제외)
├── sam2_official/                  ← Meta 공식 코드 참고용
└── Kvasir-SEG/                    ← 원본 데이터셋 (Git 제외)
```

**중요**: 이 저장소에는 Python 스크립트와 문서만 포함됩니다.
체크포인트, 데이터셋, 가상환경 등 대용량 파일은 `.gitignore`로 제외되어 있어
각 환경에서 별도로 준비해야 합니다.

---

## 1단계: 환경 설정

### macOS

```bash
# 프로젝트 클론
git clone https://github.com/eel-ohgnas/sam2.git
cd sam2

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate

# SAM2 패키지 설치 (Meta 공식)
pip install sam2

# 추가 패키지 설치
pip install opencv-python matplotlib tqdm
```

### Windows

```powershell
# 프로젝트 클론
git clone https://github.com/eel-ohgnas/sam2.git
cd sam2

# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate

# PyTorch 설치 (NVIDIA GPU 있는 경우)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# PyTorch 설치 (GPU 없는 경우)
# pip install torch torchvision

# SAM2 패키지 설치
pip install sam2

# 추가 패키지 설치
pip install opencv-python matplotlib tqdm
```

> **주의 - 폴더 이름 충돌**:
> 프로젝트 폴더 이름이 `sam2`이면 Python이 `sam2` 패키지와 혼동할 수 있습니다.
> 에러 발생 시 폴더 이름을 변경하세요:
> ```powershell
> cd ..
> ren sam2 sam2-project    # Windows
> mv sam2 sam2-project     # macOS
> cd sam2-project
> ```

### 설치 확인

```bash
python -c "from sam2.build_sam import build_sam2; print('SAM2 설치 완료!')"
```

---

## 2단계: 체크포인트 다운로드

```bash
mkdir checkpoints
curl -L -o checkpoints/sam2.1_hiera_small.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
```

Windows PowerShell:
```powershell
mkdir checkpoints
curl -L -o checkpoints\sam2.1_hiera_small.pt `
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
```

**모델 옵션**:
| 모델 | 크기 | 용도 |
|------|------|------|
| `sam2.1_hiera_tiny.pt` | 39MB | 빠른 추론, 제한된 리소스 |
| `sam2.1_hiera_small.pt` | 185MB | 균형잡힌 성능 (이 프로젝트에서 사용) |
| `sam2.1_hiera_base_plus.pt` | 324MB | 높은 정확도 |
| `sam2.1_hiera_large.pt` | 898MB | 최고 정확도 |

**출처**: [SAM2 Model Checkpoints](https://github.com/facebookresearch/sam2#download-checkpoints)

---

## 3단계: 데이터셋 준비

```bash
python download_dataset.py
```

이 스크립트가 하는 일:
1. Kvasir-SEG 데이터셋 다운로드 (1,000개 폴립 이미지 + 마스크)
2. 학습/검증 분할 (800/200)
3. `data/images/`, `data/masks/` 폴더 구조 생성

**데이터셋 출처**: [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)

---

## 4단계: 파인튜닝

```bash
python train_sam2_proper.py
```

### 학습 설정 (train_sam2_proper.py)

```python
CONFIG = {
    'checkpoint': 'checkpoints/sam2.1_hiera_small.pt',
    'model_cfg': 'configs/sam2.1/sam2.1_hiera_s.yaml',
    'image_dir': 'data/images/train',
    'mask_dir': 'data/masks/train',
    'iterations': 3000,
    'save_every': 500,
    'lr': 1e-5,
    'weight_decay': 4e-5,
}
```

- `model_cfg`: SAM2 패키지 내부의 Hydra 설정 파일 (패키지 설치 시 자동 포함)
- `checkpoint`: 사전 학습 모델 가중치 (2단계에서 다운로드)
- `iterations`: 총 학습 반복 횟수
- `lr`: 학습률 (Learning Rate)

### 핵심 학습 코드 설명

#### 학습 대상 선택

```python
predictor.model.sam_mask_decoder.train(True)   # Mask Decoder 학습
predictor.model.sam_prompt_encoder.train(True)  # Prompt Encoder 학습
# Image Encoder는 고정 (메모리 절약 + 사전 학습 특징 유지)
```

전체 모델이 아닌 Decoder와 Prompt Encoder만 학습합니다.
이미지 인코더의 사전 학습된 특징 추출 능력을 그대로 유지하면서,
새로운 도메인(의료 영상)에 맞게 마스크 생성 부분만 미세 조정합니다.

#### Forward Pass (Gradient 전파)

```python
# 이미지 인코딩 (gradient 없이)
predictor.set_image(image)

# Prompt 인코딩
sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
    points=(unnorm_coords, labels), boxes=None, masks=None,
)

# Mask Decoder (gradient 있음 - 여기가 학습되는 부분!)
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

**핵심**: `predictor.predict()`를 사용하면 `torch.no_grad()` 내부에서 실행되어
Gradient가 흐르지 않습니다. 대신 `predictor._features`에 접근하여
모델 컴포넌트를 직접 호출해야 학습이 가능합니다.

**출처**: [sagieppel - Fine-tune SAM2 in 60 lines](https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code)

#### 손실 함수

```python
# Binary Cross Entropy (픽셀 단위 마스크 정확도)
seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5)
            - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)).mean()

# IoU Score Loss (모델 신뢰도 ↔ 실제 IoU 일치도)
score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

# 총 손실 (seg_loss가 주, score_loss는 보조)
loss = seg_loss + score_loss * 0.05
```

### 학습 결과

- 학습 시간: 약 5분 30초 (M4 Max MPS)
- IoU 변화: 0.9% → 80.6%
- 저장 파일: `output/sam2_best.pt`, `output/sam2_final.pt`

---

## 5단계: 모델 비교

```bash
python compare_models.py
```

원본 SAM2와 파인튜닝 SAM2를 검증 데이터셋(200개)에서 비교합니다.

**출력 예시**:
```
┌─────────────────────────────────────────┐
│           평균 IoU 비교                  │
├─────────────────────────────────────────┤
│  원본 SAM2:      65.92%               │
│  파인튜닝 SAM2:  83.33%               │
├─────────────────────────────────────────┤
│  개선율:         +17.4%p              │
└─────────────────────────────────────────┘
```

비교 이미지 5장이 `output/comparison/` 폴더에 저장됩니다.

### IoU (Intersection over Union)

```python
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-6)
```

세그멘테이션 품질을 측정하는 표준 지표입니다.
- 1.0 (100%): 완벽한 일치
- 0.5 (50%): 절반 일치
- 0.0 (0%): 전혀 일치하지 않음

---

## 6단계: 테스트

### 좌표 지정 방식

```bash
python test_my_image.py --image test_images/dog.jpg --point "320,200"
```

### 인터랙티브 방식 (마우스 클릭)

```bash
python test_my_image.py --image test_images/dog.jpg
```

- 좌클릭: 전경 포인트 (세그멘트할 영역)
- 우클릭: 배경 포인트 (제외할 영역)
- 창 닫기: 세그멘테이션 실행

### 옵션

| 옵션 | 설명 |
|------|------|
| `--image`, `-i` | 이미지 경로 (필수) |
| `--original` | 원본 SAM2 사용 (기본: 파인튜닝 모델) |
| `--point`, `-p` | 좌표 지정 (예: `"100,200"` 또는 `"100,200,1;300,400,0"`) |
| `--output`, `-o` | 출력 폴더 (기본: `output/my_results`) |

### 결과 파일

- `output/my_results/이미지명_result.png` - 시각화 결과
- `output/my_results/이미지명_mask.png` - 마스크 파일

---

## 7. 문제 해결 (Troubleshooting)

### 7.1 SAM2 import 실패

```
ModuleNotFoundError: No module named 'sam2'
```

**해결**: SAM2를 pip로 설치합니다:
```bash
pip install sam2
```

### 7.2 폴더 이름 충돌

```
RuntimeError: You're likely running python from the parent directory
of the sam2 repository
```

**원인**: 프로젝트 폴더 이름이 `sam2`이면 Python이 `sam2` 패키지 대신
현재 폴더를 import합니다.

**해결**: 폴더 이름을 변경합니다:
```bash
cd ..
mv sam2 sam2-project     # macOS
ren sam2 sam2-project     # Windows
cd sam2-project
```

### 7.3 Config 파일을 찾을 수 없음

```
MissingConfigException: Cannot find primary config 'sam2.1_hiera_s'
```

**해결**: 전체 경로를 사용합니다:
```python
# ❌
config = 'sam2.1_hiera_s'

# ✅
config = 'configs/sam2.1/sam2.1_hiera_s.yaml'
```

이 경로는 SAM2 패키지 내부의 Hydra 설정을 가리킵니다.
`pip install sam2` 시 자동으로 포함됩니다.

### 7.4 CUDA 에러 (Apple Silicon)

```
AssertionError: Torch not compiled with CUDA enabled
```

**해결**: device 파라미터를 명시합니다:
```python
# ❌
sam2_model = build_sam2(config, checkpoint)

# ✅
sam2_model = build_sam2(config, checkpoint, device='mps')   # macOS
sam2_model = build_sam2(config, checkpoint, device='cuda')   # Windows + NVIDIA
sam2_model = build_sam2(config, checkpoint, device='cpu')    # GPU 없음
```

스크립트에는 자동 감지 코드가 포함되어 있으므로 직접 수정할 필요 없습니다.

### 7.5 Gradient가 흐르지 않음 (Loss=0.0000)

**원인**: `predictor.predict()`는 `torch.no_grad()` 컨텍스트에서 실행됩니다.

**해결**: `predictor._features`를 통해 모델 컴포넌트를 직접 호출합니다 (4단계 참조).

### 7.6 No module named 'cv2'

```bash
pip install opencv-python
```

### 7.7 Windows에서 한글/이모지 깨짐

PowerShell에서:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

또는 **Windows Terminal** 앱을 사용하세요 (기본 UTF-8 지원).

### 7.8 Visual C++ Build Tools 필요 (Windows)

```
error: Microsoft Visual C++ 14.0 or greater is required
```

[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 설치 시
**"C++를 사용한 데스크톱 개발"** 워크로드를 선택합니다.

---

## 8. macOS vs Windows 비교

| 항목 | macOS | Windows |
|------|-------|---------|
| 가상환경 활성화 | `source venv/bin/activate` | `venv\Scripts\activate` |
| GPU | MPS (Apple Silicon) | CUDA (NVIDIA) |
| Mixed Precision | 미지원 (MPS) | 지원 (CUDA) |
| 터미널 | Terminal / zsh | PowerShell / Windows Terminal |
| 경로 구분자 | `/` | `\` (Python에서 `/`도 동작) |
| PyTorch 설치 | `pip install torch` | `pip install torch --index-url .../cu121` |

---

## 9. 참고 자료

| 리소스 | 설명 |
|--------|------|
| [SAM2 GitHub](https://github.com/facebookresearch/sam2) | Meta 공식 SAM2 코드 |
| [sagieppel 60줄 파인튜닝](https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code) | 이 프로젝트의 학습 코드 기반 |
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | 의료 영상 데이터셋 |
| [SAM2 논문](https://arxiv.org/abs/2408.00714) | 모델 아키텍처 논문 |
