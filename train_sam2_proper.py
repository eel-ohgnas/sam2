"""
SAM2 파인튜닝 - 검증된 방식 (sagieppel 기반)
Kvasir-SEG 데이터셋 + Apple Silicon MPS 지원

참고: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code
"""

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ============ 설정 ============
CONFIG = {
    'checkpoint': 'checkpoints/sam2.1_hiera_small.pt',
    'model_cfg': 'configs/sam2.1/sam2.1_hiera_s.yaml',
    'image_dir': 'data/images/train',
    'mask_dir': 'data/masks/train',
    'output_dir': 'output',
    'iterations': 3000,  # 학습 반복 횟수
    'save_every': 500,   # 저장 주기
    'lr': 1e-5,
    'weight_decay': 4e-5,
}


def load_data(image_dir, mask_dir):
    """데이터셋 로드"""
    data = []
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # 마스크 찾기
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + ".jpg")
            if mask_path.exists():
                data.append({
                    'image': str(img_path),
                    'mask': str(mask_path)
                })

    print(f"📁 로드된 데이터: {len(data)}개")
    return data


def read_batch(data, device):
    """랜덤 이미지-마스크 쌍 읽기"""
    while True:
        # 랜덤 선택
        entry = data[np.random.randint(len(data))]

        # 이미지 읽기 (BGR -> RGB)
        img = cv2.imread(entry['image'])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 마스크 읽기
        mask = cv2.imread(entry['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # 리사이즈 (최대 1024)
        r = min(1024 / img.shape[1], 1024 / img.shape[0])
        if r < 1:
            img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
            mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)),
                            interpolation=cv2.INTER_NEAREST)

        # 이진화
        mask = (mask > 127).astype(np.uint8)

        # 마스크가 비어있으면 스킵
        if mask.sum() < 100:
            continue

        # 마스크에서 랜덤 포인트 선택
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            continue

        yx = coords[np.random.randint(len(coords))]
        point = np.array([[[yx[1], yx[0]]]])  # (1, 1, 2) - x, y
        label = np.array([[1]])  # foreground

        # 마스크를 (1, H, W) 형태로
        mask = mask[np.newaxis, :, :]

        return img, mask, point, label


def main():
    print("\n" + "="*60)
    print("🚀 SAM2 파인튜닝 (검증된 방식)")
    print("="*60)

    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon GPU (MPS) 사용")
        use_amp = False  # MPS는 AMP 미지원
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ NVIDIA GPU (CUDA) 사용")
        use_amp = True
    else:
        device = torch.device("cpu")
        print("⚠️ CPU 사용")
        use_amp = False

    # 데이터 로드
    data = load_data(CONFIG['image_dir'], CONFIG['mask_dir'])
    if len(data) == 0:
        print("❌ 데이터가 없습니다!")
        return

    # 모델 로드
    print("\n📦 SAM2 모델 로드 중...")
    device_str = str(device).split(':')[0]  # "mps" or "cuda" or "cpu"

    sam2_model = build_sam2(
        CONFIG['model_cfg'],
        CONFIG['checkpoint'],
        device=device_str
    )
    predictor = SAM2ImagePredictor(sam2_model)

    # 학습 모드 설정
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    # Image encoder는 학습하지 않음 (메모리 절약)

    # 학습 가능한 파라미터 확인
    trainable = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in predictor.model.parameters())
    print(f"📊 학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    # Mixed Precision (CUDA only)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # 출력 폴더
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 학습 루프
    print(f"\n🎯 학습 시작: {CONFIG['iterations']} iterations")
    print("-" * 60)

    mean_iou = 0
    best_iou = 0

    pbar = tqdm(range(CONFIG['iterations']), desc="Training")
    for itr in pbar:
        try:
            # 데이터 로드
            image, mask, input_point, input_label = read_batch(data, device)

            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, iou = forward_pass(predictor, image, mask, input_point, input_label, device)
            else:
                loss, iou = forward_pass(predictor, image, mask, input_point, input_label, device)

            if loss is None:
                continue

            # Backward pass
            predictor.model.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # 통계 업데이트
            current_iou = iou.mean().item()
            mean_iou = mean_iou * 0.99 + 0.01 * current_iou

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'IoU': f'{mean_iou:.4f}'
            })

            # 모델 저장
            if (itr + 1) % CONFIG['save_every'] == 0:
                save_path = os.path.join(CONFIG['output_dir'], f'sam2_iter_{itr+1}.pt')
                torch.save(predictor.model.state_dict(), save_path)
                print(f"\n💾 저장: {save_path} (IoU: {mean_iou:.4f})")

                if mean_iou > best_iou:
                    best_iou = mean_iou
                    best_path = os.path.join(CONFIG['output_dir'], 'sam2_best.pt')
                    torch.save(predictor.model.state_dict(), best_path)
                    print(f"🏆 베스트 모델 저장: {best_path}")

        except Exception as e:
            print(f"\n⚠️ 오류: {e}")
            continue

    # 최종 저장
    final_path = os.path.join(CONFIG['output_dir'], 'sam2_final.pt')
    torch.save(predictor.model.state_dict(), final_path)

    print("\n" + "="*60)
    print("✅ 학습 완료!")
    print("="*60)
    print(f"""
📊 결과:
   - 최종 IoU: {mean_iou:.4f}
   - 베스트 IoU: {best_iou:.4f}
   - 저장 위치: {CONFIG['output_dir']}/

📁 생성된 파일:
   - sam2_best.pt (베스트 모델)
   - sam2_final.pt (최종 모델)
""")


def forward_pass(predictor, image, mask, input_point, input_label, device):
    """Forward pass with gradient"""

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

    # Mask Decoder
    batched_mode = unnorm_coords.shape[0] > 1
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=batched_mode,
        high_res_features=high_res_features,
    )

    # 마스크 업스케일
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

    # Ground truth
    gt_mask = torch.tensor(mask.astype(np.float32), device=device)

    # 예측 마스크 (sigmoid 적용)
    prd_mask = torch.sigmoid(prd_masks[:, 0])

    # Segmentation Loss (Binary Cross Entropy)
    seg_loss = (
        -gt_mask * torch.log(prd_mask + 1e-5)
        - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)
    ).mean()

    # IoU Score Loss
    inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1, 2))
    union = gt_mask.sum(dim=(1, 2)) + (prd_mask > 0.5).sum(dim=(1, 2)) - inter
    iou = inter / (union + 1e-5)

    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

    # Total Loss
    loss = seg_loss + score_loss * 0.05

    return loss, iou


if __name__ == "__main__":
    main()
