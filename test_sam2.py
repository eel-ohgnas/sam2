"""
SAM2 파인튜닝 결과 테스트 스크립트
학습된 모델로 이미지 세분화를 테스트합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️ SAM2가 설치되지 않았습니다.")


def show_mask(mask, ax, color=None):
    """마스크를 시각화합니다."""
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """포인트 프롬프트를 시각화합니다."""
    pos_points = coords[labels == 1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='lime',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.5)


def test_on_image(image_path, predictor, output_path=None):
    """단일 이미지 테스트"""
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    h, w = image.shape[:2]
    center_point = np.array([[w // 2, h // 2]])
    point_labels = np.array([1])

    # 예측
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=point_labels,
        multimask_output=True
    )

    # 시각화
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 원본
    axes[0].imshow(image)
    show_points(center_point, point_labels, axes[0])
    axes[0].set_title("Input (center click)")
    axes[0].axis("off")

    # 마스크들
    for i, (mask, score) in enumerate(zip(masks[:3], scores[:3])):
        axes[i+1].imshow(image)
        show_mask(mask, axes[i+1])
        axes[i+1].set_title(f"Mask {i+1} (score: {score:.3f})")
        axes[i+1].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 저장: {output_path}")

    plt.close()
    return masks, scores


def main():
    if not SAM2_AVAILABLE:
        print("❌ SAM2가 설치되지 않았습니다.")
        return

    # 설정
    config = {
        'model_cfg': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        'original_checkpoint': 'checkpoints/sam2.1_hiera_small.pt',
        'finetuned_checkpoint': 'output/sam2_finetuned_best.pt',
        'test_image_dir': 'data/images/val',
        'output_dir': 'output/test_results',
    }

    # 출력 폴더
    os.makedirs(config['output_dir'], exist_ok=True)

    # 디바이스
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"🖥️ 디바이스: {device}")

    # 테스트 이미지 찾기
    test_dir = Path(config['test_image_dir'])
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

    if len(test_images) == 0:
        print(f"⚠️ 테스트 이미지가 없습니다: {config['test_image_dir']}")
        return

    print(f"📁 테스트 이미지: {len(test_images)}개")

    # 원본 모델 로드
    print("\n📦 원본 SAM2 모델 로드 중...")
    model_original = build_sam2(
        config['model_cfg'],
        config['original_checkpoint'],
        device=device
    )
    predictor_original = SAM2ImagePredictor(model_original)

    # 테스트 (처음 5개)
    print("\n🧪 원본 모델 테스트:")
    for img_path in test_images[:5]:
        output_path = os.path.join(config['output_dir'], f"original_{img_path.stem}.png")
        test_on_image(img_path, predictor_original, output_path)

    print(f"\n✅ 테스트 완료!")
    print(f"📂 결과 위치: {config['output_dir']}/")


if __name__ == "__main__":
    main()
