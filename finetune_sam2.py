"""
SAM2 파인튜닝 스크립트 (초보자용)
참고: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code

이 스크립트는 SAM2의 Mask Decoder만 파인튜닝합니다.
- 장점: 적은 GPU 메모리, 빠른 학습
- 필요 GPU 메모리: ~8GB
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️ SAM2가 설치되지 않았습니다. setup_guide.md를 참고하세요.")


def load_image_and_mask(image_path, mask_path, image_size=1024):
    """이미지와 마스크를 로드하고 전처리합니다."""
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image = np.array(image)

    # 마스크 로드
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    mask = np.array(mask) > 127  # 이진화

    return image, mask


def get_random_point_in_mask(mask):
    """마스크 내부에서 랜덤 포인트를 선택합니다."""
    coords = np.where(mask)
    if len(coords[0]) == 0:
        # 마스크가 비어있으면 중앙점 반환
        return np.array([[mask.shape[1] // 2, mask.shape[0] // 2]])

    idx = np.random.randint(len(coords[0]))
    return np.array([[coords[1][idx], coords[0][idx]]])  # (x, y)


def train_one_epoch(model, image_paths, mask_paths, optimizer, device, image_size=1024):
    """한 에폭 학습 - 모델 직접 사용"""
    model.train()

    # Image encoder는 freeze, mask decoder만 학습
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.sam_mask_decoder.parameters():
        param.requires_grad = True

    total_loss = 0
    indices = list(range(len(image_paths)))
    random.shuffle(indices)

    pbar = tqdm(indices, desc="Training")
    for idx in pbar:
        try:
            # 데이터 로드
            image, gt_mask = load_image_and_mask(
                image_paths[idx], mask_paths[idx], image_size
            )

            if not gt_mask.any():  # 빈 마스크 스킵
                continue

            # 랜덤 포인트 선택
            point_coords = get_random_point_in_mask(gt_mask)

            # 이미지를 텐서로 변환 (B, C, H, W)
            image_tensor = torch.tensor(image, dtype=torch.float32, device=device)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            image_tensor = image_tensor / 255.0  # 정규화

            # 타겟 마스크
            gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32, device=device)
            gt_mask_256 = F.interpolate(
                gt_mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # 포인트 좌표 텐서
            point_coords_tensor = torch.tensor(point_coords, dtype=torch.float32, device=device)
            point_coords_tensor = point_coords_tensor.unsqueeze(0)  # (1, N, 2)
            point_labels = torch.ones((1, point_coords_tensor.shape[1]), dtype=torch.int32, device=device)

            # Forward pass with gradient
            with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=False):
                # Image encoding (no grad)
                with torch.no_grad():
                    backbone_out = model.forward_image(image_tensor)
                    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)

                    # 수정: 마지막 feature 사용
                    if len(vision_feats) > 0:
                        feat = vision_feats[-1]
                        if feat.dim() == 3:  # (B, N, C) -> (B, C, H, W)
                            B, N, C = feat.shape
                            H = W = int(N ** 0.5)
                            feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
                        image_embed = feat
                    else:
                        continue

                # Point 임베딩 생성
                concat_points = (point_coords_tensor, point_labels)
                sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                    points=concat_points,
                    boxes=None,
                    masks=None,
                )

                # Mask decoder (with grad)
                low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=False,
                )

                # 손실 계산
                pred_mask = low_res_masks[:, 0, :, :]  # 첫 번째 마스크 사용
                pred_mask = F.interpolate(
                    pred_mask.unsqueeze(1),
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

                loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask_256)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        except Exception as e:
            print(f"\n⚠️ 오류 발생 (스킵): {e}")
            continue

    return total_loss / max(len(indices), 1)


def main():
    """메인 학습 함수"""

    if not SAM2_AVAILABLE:
        print("❌ SAM2가 설치되지 않았습니다.")
        print("📖 setup_guide.md를 참고하여 설치해주세요.")
        return

    # ============ 설정 ============
    config = {
        'checkpoint_path': 'checkpoints/sam2.1_hiera_small.pt',
        'model_cfg': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        'image_dir': 'data/images/train',
        'mask_dir': 'data/masks/train',
        'output_dir': 'output',
        'epochs': 5,  # 빠른 테스트를 위해 줄임
        'learning_rate': 1e-5,
        'image_size': 1024,
    }

    print("\n" + "="*50)
    print("🚀 SAM2 파인튜닝 시작")
    print("="*50)

    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon GPU (MPS) 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ NVIDIA GPU (CUDA) 사용")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU 사용 (느릴 수 있음)")

    # 체크포인트 확인
    if not os.path.exists(config['checkpoint_path']):
        print(f"\n❌ 체크포인트를 찾을 수 없습니다: {config['checkpoint_path']}")
        return

    # 데이터 경로 수집
    image_dir = Path(config['image_dir'])
    mask_dir = Path(config['mask_dir'])

    if not image_dir.exists():
        print(f"\n❌ 이미지 폴더가 없습니다: {image_dir}")
        print("📂 python download_dataset.py 를 먼저 실행하세요.")
        return

    # 이미지-마스크 쌍 찾기
    image_paths = []
    mask_paths = []

    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + ".jpg")
            if mask_path.exists():
                image_paths.append(img_path)
                mask_paths.append(mask_path)

    print(f"📁 데이터셋: {len(image_paths)}개 이미지-마스크 쌍")

    if len(image_paths) == 0:
        print("❌ 유효한 이미지-마스크 쌍이 없습니다.")
        return

    # 출력 폴더 생성
    os.makedirs(config['output_dir'], exist_ok=True)

    # 모델 로드
    print("\n📦 SAM2 모델 로드 중...")
    device_str = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")

    model = build_sam2(
        config['model_cfg'],
        config['checkpoint_path'],
        device=device_str,
        mode="train"  # 학습 모드
    )

    # 옵티마이저 설정 (Mask Decoder만 학습)
    optimizer = torch.optim.AdamW(
        model.sam_mask_decoder.parameters(),
        lr=config['learning_rate']
    )

    # 학습 루프
    print(f"\n🎯 학습 시작: {config['epochs']} 에폭")
    print("-" * 50)

    best_loss = float('inf')
    for epoch in range(config['epochs']):
        loss = train_one_epoch(
            model, image_paths, mask_paths,
            optimizer, device, config['image_size']
        )
        print(f"\nEpoch {epoch+1}/{config['epochs']} | Average Loss: {loss:.4f}")

        # 베스트 모델 저장
        if loss < best_loss:
            best_loss = loss
            save_path = os.path.join(config['output_dir'], 'sam2_finetuned_best.pt')
            torch.save({
                'model_state_dict': model.sam_mask_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }, save_path)
            print(f"💾 베스트 모델 저장: {save_path}")

    # 최종 모델 저장
    final_path = os.path.join(config['output_dir'], 'sam2_finetuned_final.pt')
    torch.save({
        'model_state_dict': model.sam_mask_decoder.state_dict(),
        'epoch': config['epochs'],
        'loss': loss,
    }, final_path)
    print(f"\n✅ 학습 완료! 최종 모델: {final_path}")


if __name__ == "__main__":
    main()
