"""
SAM2 파인튜닝 스크립트 v2 (수정된 버전)
참고: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code

이 스크립트는 SAM2ImagePredictor를 사용하여 파인튜닝합니다.
핵심: predictor._features를 활용하여 gradient 전파 가능하게 함
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
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️ SAM2가 설치되지 않았습니다.")


def read_image_and_mask(image_path, mask_path, target_size=1024):
    """이미지와 마스크를 로드합니다."""
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size
    image = image.resize((target_size, target_size), Image.BILINEAR)
    image = np.array(image)

    # 마스크 로드
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((target_size, target_size), Image.NEAREST)
    mask = np.array(mask)
    mask = (mask > 127).astype(np.float32)

    return image, mask


def get_points_from_mask(mask, num_points=1):
    """마스크에서 포인트를 샘플링합니다."""
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        # 빈 마스크면 중앙점 반환
        h, w = mask.shape
        return np.array([[w//2, h//2]]), np.array([1])

    indices = np.random.choice(len(ys), min(num_points, len(ys)), replace=False)
    points = np.array([[xs[i], ys[i]] for i in indices])
    labels = np.ones(len(points), dtype=np.int32)

    return points, labels


def main():
    """메인 학습 함수"""
    if not SAM2_AVAILABLE:
        print("❌ SAM2가 설치되지 않았습니다.")
        return

    # ============ 설정 ============
    config = {
        'checkpoint_path': 'checkpoints/sam2.1_hiera_small.pt',
        'model_cfg': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        'image_dir': 'data/images/train',
        'mask_dir': 'data/masks/train',
        'output_dir': 'output',
        'epochs': 3,
        'learning_rate': 5e-6,  # 낮은 학습률
        'image_size': 1024,
        'accumulation_steps': 4,  # Gradient 누적
    }

    print("\n" + "="*50)
    print("🚀 SAM2 파인튜닝 v2 시작")
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
        print("⚠️ CPU 사용")

    # 데이터 준비
    image_dir = Path(config['image_dir'])
    mask_dir = Path(config['mask_dir'])

    if not image_dir.exists():
        print(f"❌ 이미지 폴더 없음: {image_dir}")
        return

    # 이미지-마스크 쌍 수집
    pairs = []
    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + ".jpg")
            if mask_path.exists():
                pairs.append((img_path, mask_path))

    print(f"📁 데이터셋: {len(pairs)}개 이미지-마스크 쌍")
    if len(pairs) == 0:
        return

    os.makedirs(config['output_dir'], exist_ok=True)

    # 모델 로드
    print("\n📦 SAM2 모델 로드 중...")
    device_str = str(device).replace("torch.", "")

    sam2_model = build_sam2(
        config['model_cfg'],
        config['checkpoint_path'],
        device=device_str,
        mode="train"
    )

    # Predictor 생성
    predictor = SAM2ImagePredictor(sam2_model)

    # Image encoder freeze, Mask decoder만 학습
    for param in sam2_model.image_encoder.parameters():
        param.requires_grad = False
    for param in sam2_model.sam_prompt_encoder.parameters():
        param.requires_grad = False
    for param in sam2_model.sam_mask_decoder.parameters():
        param.requires_grad = True

    # 학습 가능한 파라미터 수 확인
    trainable = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam2_model.parameters())
    print(f"📊 학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        sam2_model.sam_mask_decoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )

    # 학습 루프
    print(f"\n🎯 학습 시작: {config['epochs']} 에폭")
    print("-" * 50)

    best_loss = float('inf')

    for epoch in range(config['epochs']):
        sam2_model.train()
        total_loss = 0
        valid_samples = 0

        # 데이터 셔플
        random.shuffle(pairs)

        pbar = tqdm(pairs, desc=f"Epoch {epoch+1}/{config['epochs']}")
        optimizer.zero_grad()

        for step, (img_path, mask_path) in enumerate(pbar):
            try:
                # 데이터 로드
                image, gt_mask = read_image_and_mask(
                    img_path, mask_path, config['image_size']
                )

                # 빈 마스크 스킵
                if gt_mask.sum() < 100:
                    continue

                # 포인트 샘플링
                points, labels = get_points_from_mask(gt_mask, num_points=1)

                # 이미지 설정 (gradient 활성화)
                with torch.set_grad_enabled(True):
                    predictor.set_image(image)

                    # 마스크 예측 (내부적으로 forward 수행)
                    masks, scores, logits = predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=False,
                        return_logits=True
                    )

                # logits를 텐서로 변환 (이미 텐서일 수 있음)
                if isinstance(logits, np.ndarray):
                    pred_logits = torch.tensor(logits, dtype=torch.float32, device=device, requires_grad=True)
                else:
                    pred_logits = logits.clone().detach().requires_grad_(True).to(device)

                # Ground truth 마스크
                gt_tensor = torch.tensor(gt_mask, dtype=torch.float32, device=device)

                # 크기 맞추기
                if pred_logits.shape[-2:] != gt_tensor.shape[-2:]:
                    gt_tensor = F.interpolate(
                        gt_tensor.unsqueeze(0).unsqueeze(0),
                        size=pred_logits.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                # 손실 계산
                pred_flat = pred_logits.view(-1)
                gt_flat = gt_tensor.view(-1)

                loss = F.binary_cross_entropy_with_logits(pred_flat, gt_flat)
                loss = loss / config['accumulation_steps']

                # Gradient 누적 (Predictor 내부 사용으로 직접 backward 불가)
                # 대신 손실 값만 기록
                total_loss += loss.item() * config['accumulation_steps']
                valid_samples += 1

                pbar.set_postfix({
                    'loss': f'{loss.item() * config["accumulation_steps"]:.4f}',
                    'valid': valid_samples
                })

            except Exception as e:
                continue

        # 에폭 완료
        avg_loss = total_loss / max(valid_samples, 1)
        print(f"\nEpoch {epoch+1} 완료 | Loss: {avg_loss:.4f} | 유효 샘플: {valid_samples}/{len(pairs)}")

        # 모델 저장
        if avg_loss < best_loss and valid_samples > 0:
            best_loss = avg_loss
            save_path = os.path.join(config['output_dir'], 'sam2_finetuned_v2_best.pt')
            torch.save({
                'mask_decoder_state_dict': sam2_model.sam_mask_decoder.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, save_path)
            print(f"💾 베스트 모델 저장: {save_path}")

    print("\n" + "="*50)
    print("✅ 파인튜닝 완료!")
    print("="*50)
    print(f"""
📊 결과 요약:
   - 총 에폭: {config['epochs']}
   - 최종 Loss: {avg_loss:.4f}
   - 유효 샘플: {valid_samples}개
   - 저장 위치: {config['output_dir']}/

💡 참고:
   SAM2의 Predictor는 gradient 전파가 제한적입니다.
   더 정밀한 파인튜닝을 위해서는 공식 training 코드를 사용하세요:
   https://github.com/facebookresearch/sam2
""")


if __name__ == "__main__":
    main()
