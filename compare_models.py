"""
원본 SAM2 vs 파인튜닝 SAM2 비교 스크립트
"""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def calculate_iou(pred_mask, gt_mask):
    """IoU 계산"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-6)


def test_model(predictor, image_paths, mask_paths, desc="Testing"):
    """모델 테스트 및 IoU 계산"""
    ious = []

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc=desc):
        # 이미지 로드
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # GT 마스크 로드
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # 마스크에서 포인트 샘플링
        coords = np.argwhere(gt_mask > 0)
        if len(coords) == 0:
            continue
        yx = coords[len(coords) // 2]  # 중앙 포인트
        point = np.array([[yx[1], yx[0]]])
        label = np.array([1])

        # 예측
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True
        )

        # 가장 좋은 마스크 선택
        best_idx = np.argmax(scores)
        pred_mask = masks[best_idx]

        # GT 마스크 리사이즈
        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

        # IoU 계산
        iou = calculate_iou(pred_mask, gt_mask > 0)
        ious.append(iou)

    return np.mean(ious), ious


def visualize_comparison(image_path, mask_path, predictor_original, predictor_finetuned, output_path):
    """비교 시각화"""
    # 이미지 로드
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # GT 마스크 로드
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 127)

    # 포인트 선택
    coords = np.argwhere(gt_mask > 0)
    if len(coords) == 0:
        return
    yx = coords[len(coords) // 2]
    point = np.array([[yx[1], yx[0]]])
    label = np.array([1])

    # 원본 모델 예측
    predictor_original.set_image(image)
    masks_orig, scores_orig, _ = predictor_original.predict(
        point_coords=point, point_labels=label, multimask_output=True
    )
    best_orig = masks_orig[np.argmax(scores_orig)]

    # 파인튜닝 모델 예측
    predictor_finetuned.set_image(image)
    masks_ft, scores_ft, _ = predictor_finetuned.predict(
        point_coords=point, point_labels=label, multimask_output=True
    )
    best_ft = masks_ft[np.argmax(scores_ft)]

    # 시각화
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 원본 이미지 + 포인트
    axes[0].imshow(image)
    axes[0].scatter(point[0, 0], point[0, 1], c='lime', s=200, marker='*')
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Ground Truth
    axes[1].imshow(image)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # 원본 SAM2
    iou_orig = calculate_iou(best_orig, gt_mask)
    axes[2].imshow(image)
    axes[2].imshow(best_orig, alpha=0.5, cmap='Blues')
    axes[2].set_title(f"Original SAM2\nIoU: {iou_orig:.2%}")
    axes[2].axis("off")

    # 파인튜닝 SAM2
    iou_ft = calculate_iou(best_ft, gt_mask)
    axes[3].imshow(image)
    axes[3].imshow(best_ft, alpha=0.5, cmap='Greens')
    axes[3].set_title(f"Fine-tuned SAM2\nIoU: {iou_ft:.2%}")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("\n" + "="*60)
    print("🔬 원본 vs 파인튜닝 SAM2 비교")
    print("="*60)

    # 설정
    config = {
        'model_cfg': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        'original_checkpoint': 'checkpoints/sam2.1_hiera_small.pt',
        'finetuned_checkpoint': 'output/sam2_best.pt',
        'test_image_dir': 'data/images/val',
        'test_mask_dir': 'data/masks/val',
        'output_dir': 'output/comparison',
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    # 디바이스
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"🖥️ 디바이스: {device}")

    # 테스트 데이터 로드
    test_image_dir = Path(config['test_image_dir'])
    test_mask_dir = Path(config['test_mask_dir'])

    image_paths = []
    mask_paths = []
    for img_path in sorted(test_image_dir.glob("*")):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            mask_path = test_mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                image_paths.append(img_path)
                mask_paths.append(mask_path)

    print(f"📁 테스트 이미지: {len(image_paths)}개")

    # 원본 모델 로드
    print("\n📦 원본 SAM2 로드 중...")
    model_original = build_sam2(
        config['model_cfg'],
        config['original_checkpoint'],
        device=device
    )
    predictor_original = SAM2ImagePredictor(model_original)

    # 파인튜닝 모델 로드
    print("📦 파인튜닝 SAM2 로드 중...")
    model_finetuned = build_sam2(
        config['model_cfg'],
        config['original_checkpoint'],  # 먼저 원본 구조 로드
        device=device
    )
    # 파인튜닝된 가중치 로드
    state_dict = torch.load(config['finetuned_checkpoint'], map_location=device)
    model_finetuned.load_state_dict(state_dict)
    predictor_finetuned = SAM2ImagePredictor(model_finetuned)

    # 테스트 (처음 50개만)
    test_size = min(50, len(image_paths))
    print(f"\n🧪 테스트 중... ({test_size}개 이미지)")

    iou_original, ious_orig = test_model(
        predictor_original,
        image_paths[:test_size],
        mask_paths[:test_size],
        "원본 SAM2"
    )

    iou_finetuned, ious_ft = test_model(
        predictor_finetuned,
        image_paths[:test_size],
        mask_paths[:test_size],
        "파인튜닝 SAM2"
    )

    # 결과 출력
    print("\n" + "="*60)
    print("📊 결과 비교")
    print("="*60)
    print(f"""
    ┌─────────────────────────────────────────┐
    │           평균 IoU 비교                  │
    ├─────────────────────────────────────────┤
    │  원본 SAM2:      {iou_original:.2%}               │
    │  파인튜닝 SAM2:  {iou_finetuned:.2%}               │
    ├─────────────────────────────────────────┤
    │  🚀 개선율:      +{(iou_finetuned - iou_original)*100:.1f}%p              │
    └─────────────────────────────────────────┘
    """)

    # 비교 시각화 (처음 5개)
    print("🖼️ 비교 이미지 생성 중...")
    for i in range(min(5, len(image_paths))):
        output_path = os.path.join(config['output_dir'], f"compare_{i+1}.png")
        visualize_comparison(
            image_paths[i],
            mask_paths[i],
            predictor_original,
            predictor_finetuned,
            output_path
        )
        print(f"  💾 {output_path}")

    print(f"\n✅ 완료! 비교 이미지: {config['output_dir']}/")


if __name__ == "__main__":
    main()
