"""
내 이미지로 SAM2 테스트하기
사용법: python test_my_image.py --image 이미지경로.jpg
"""

import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_model(use_finetuned=True):
    """모델 로드"""
    config = 'configs/sam2.1/sam2.1_hiera_s.yaml'

    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"디바이스: {device}")

    # 모델 로드
    original_checkpoint = 'checkpoints/sam2.1_hiera_small.pt'
    model = build_sam2(config, original_checkpoint, device=device)

    if use_finetuned:
        finetuned_checkpoint = 'output/sam2_best.pt'
        if Path(finetuned_checkpoint).exists():
            state_dict = torch.load(finetuned_checkpoint, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ 파인튜닝 모델 로드 완료")
        else:
            print("⚠️ 파인튜닝 모델 없음, 원본 모델 사용")
    else:
        print("📦 원본 SAM2 모델 사용")

    return SAM2ImagePredictor(model)


def interactive_segment(image_path, predictor, output_dir="output/my_results"):
    """인터랙티브 세그멘테이션"""
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📷 이미지 크기: {image.shape[1]} x {image.shape[0]}")

    # 출력 폴더 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 클릭 포인트 저장
    points = []
    labels = []

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("클릭으로 포인트 추가 (좌클릭=전경, 우클릭=배경)\n닫으면 세그멘테이션 실행")
    ax.axis('off')

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if event.button == 1:  # 좌클릭 = 전경
            points.append([x, y])
            labels.append(1)
            ax.scatter(x, y, c='lime', s=200, marker='*', edgecolors='white', linewidths=2)
            print(f"  ✅ 전경 포인트 추가: ({x}, {y})")
        elif event.button == 3:  # 우클릭 = 배경
            points.append([x, y])
            labels.append(0)
            ax.scatter(x, y, c='red', s=200, marker='x', linewidths=3)
            print(f"  ❌ 배경 포인트 추가: ({x}, {y})")

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    print("\n🖱️ 이미지를 클릭하세요:")
    print("   - 좌클릭: 전경 (세그멘트할 영역)")
    print("   - 우클릭: 배경 (제외할 영역)")
    print("   - 창 닫기: 세그멘테이션 실행\n")

    plt.show()

    if len(points) == 0:
        print("⚠️ 포인트가 선택되지 않았습니다.")
        return

    # 세그멘테이션 실행
    print(f"\n🔄 세그멘테이션 실행 중... ({len(points)}개 포인트)")

    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=True
    )

    # 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 원본 이미지 + 포인트
    axes[0, 0].imshow(image)
    for i, (pt, lb) in enumerate(zip(points, labels)):
        color = 'lime' if lb == 1 else 'red'
        marker = '*' if lb == 1 else 'x'
        axes[0, 0].scatter(pt[0], pt[1], c=color, s=200, marker=marker, edgecolors='white', linewidths=2)
    axes[0, 0].set_title("입력 이미지 + 포인트")
    axes[0, 0].axis('off')

    # 3개 마스크 출력
    colors = ['Blues', 'Greens', 'Oranges']
    for i, (mask, score) in enumerate(zip(masks, scores)):
        row, col = (0, 1) if i == 0 else (1, i-1)
        axes[row, col].imshow(image)
        axes[row, col].imshow(mask, alpha=0.6, cmap=colors[i])
        axes[row, col].set_title(f"마스크 {i+1} (Score: {score:.2%})")
        axes[row, col].axis('off')

    plt.tight_layout()

    # 저장
    img_name = Path(image_path).stem
    output_path = f"{output_dir}/{img_name}_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 결과 저장: {output_path}")

    plt.show()

    # 베스트 마스크 저장
    best_idx = np.argmax(scores)
    mask_output = (masks[best_idx] * 255).astype(np.uint8)
    mask_path = f"{output_dir}/{img_name}_mask.png"
    cv2.imwrite(mask_path, mask_output)
    print(f"💾 마스크 저장: {mask_path}")

    return masks[best_idx]


def batch_segment(image_path, predictor, points_list, output_dir="output/my_results"):
    """좌표를 직접 지정해서 세그멘테이션 (GUI 없이)"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    predictor.set_image(image)

    points = np.array([[p[0], p[1]] for p in points_list])
    labels = np.array([p[2] if len(p) > 2 else 1 for p in points_list])

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )

    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # 결과 저장
    img_name = Path(image_path).stem

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    for pt, lb in zip(points, labels):
        color = 'lime' if lb == 1 else 'red'
        axes[0].scatter(pt[0], pt[1], c=color, s=200, marker='*')
    axes[0].set_title("입력")
    axes[0].axis('off')

    axes[1].imshow(best_mask, cmap='gray')
    axes[1].set_title(f"마스크 (Score: {scores[best_idx]:.2%})")
    axes[1].axis('off')

    axes[2].imshow(image)
    axes[2].imshow(best_mask, alpha=0.5, cmap='Greens')
    axes[2].set_title("오버레이")
    axes[2].axis('off')

    plt.tight_layout()
    output_path = f"{output_dir}/{img_name}_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 저장 완료: {output_path}")
    return best_mask


def main():
    parser = argparse.ArgumentParser(description="내 이미지로 SAM2 테스트")
    parser.add_argument("--image", "-i", type=str, required=True, help="테스트할 이미지 경로")
    parser.add_argument("--original", action="store_true", help="원본 SAM2 사용 (기본: 파인튜닝 모델)")
    parser.add_argument("--point", "-p", type=str, help="포인트 좌표 (예: '100,200' 또는 '100,200,1;300,400,0')")
    parser.add_argument("--output", "-o", type=str, default="output/my_results", help="출력 폴더")

    args = parser.parse_args()

    print("\n" + "="*50)
    print("🎯 SAM2 이미지 세그멘테이션 테스트")
    print("="*50)

    # 모델 로드
    predictor = load_model(use_finetuned=not args.original)

    if args.point:
        # 좌표 직접 지정
        points = []
        for pt_str in args.point.split(';'):
            parts = pt_str.split(',')
            x, y = int(parts[0]), int(parts[1])
            label = int(parts[2]) if len(parts) > 2 else 1
            points.append([x, y, label])

        batch_segment(args.image, predictor, points, args.output)
    else:
        # 인터랙티브 모드
        interactive_segment(args.image, predictor, args.output)

    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
