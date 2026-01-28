"""
SAM2 파인튜닝용 공개 데이터셋 다운로드 스크립트
초보자용 - 선택한 데이터셋을 자동으로 다운로드하고 정리합니다.
"""

import os
import ssl
import shutil
import zipfile
import urllib.request
import subprocess
from pathlib import Path
from tqdm import tqdm

# SSL 인증서 문제 해결 (Mac에서 흔히 발생)
ssl._create_default_https_context = ssl._create_unverified_context


class DownloadProgressBar(tqdm):
    """다운로드 진행률 표시"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """파일 다운로드 with 진행률 표시 (curl 사용)"""
    print(f"📥 다운로드: {url}")
    try:
        # curl 사용 (더 안정적)
        result = subprocess.run(
            ["curl", "-L", "-o", output_path, "--progress-bar", url],
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        # curl 실패 시 urllib 시도
        print("curl 실패, urllib로 재시도...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True


def setup_kvasir_seg():
    """
    Kvasir-SEG 데이터셋 다운로드 및 설정
    - 1,000개 폴립 이미지 + 마스크
    - 의료 영상 세분화 연습에 적합
    - 다운로드 크기: ~46MB
    """
    print("\n" + "="*60)
    print("📥 Kvasir-SEG 데이터셋 다운로드")
    print("="*60)
    print("• 이미지 수: 1,000장")
    print("• 용도: 대장 폴립 세분화")
    print("• 크기: ~46MB")
    print("="*60)

    # 다운로드 URL
    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    zip_path = "kvasir-seg.zip"
    extract_dir = "kvasir-seg"

    # 다운로드
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
        print("\n⬇️ 다운로드 중... (약 46MB)")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        download_file(url, zip_path)
        print("✅ 다운로드 완료!")
    else:
        print("✅ 이미 다운로드됨")

    # 압축 해제
    if not os.path.exists(extract_dir):
        print("\n📂 압축 해제 중...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ 압축 해제 완료!")

    # 데이터 폴더 구조로 정리
    print("\n🔧 데이터 정리 중...")

    # 폴더 생성
    for folder in ["data/images/train", "data/images/val", "data/masks/train", "data/masks/val"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # 이미지와 마스크 경로 찾기
    kvasir_images = Path("Kvasir-SEG/images") if Path("Kvasir-SEG/images").exists() else Path("kvasir-seg/images")
    kvasir_masks = Path("Kvasir-SEG/masks") if Path("Kvasir-SEG/masks").exists() else Path("kvasir-seg/masks")

    if not kvasir_images.exists():
        # 다른 가능한 경로 확인
        for possible_path in ["Kvasir-SEG", "kvasir-seg", "kvasir_seg"]:
            if Path(possible_path).exists():
                for subdir in Path(possible_path).iterdir():
                    if subdir.is_dir():
                        if "image" in subdir.name.lower():
                            kvasir_images = subdir
                        elif "mask" in subdir.name.lower():
                            kvasir_masks = subdir

    if not kvasir_images.exists() or not kvasir_masks.exists():
        print(f"❌ 데이터셋 구조를 찾을 수 없습니다.")
        print(f"   다운로드된 폴더 내용을 확인해주세요.")
        return False

    # 이미지 목록 가져오기
    images = sorted(list(kvasir_images.glob("*.jpg")) + list(kvasir_images.glob("*.png")))
    print(f"📁 발견된 이미지: {len(images)}개")

    # 80/20 분할 (학습/검증)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"   학습용: {len(train_images)}개")
    print(f"   검증용: {len(val_images)}개")

    # 파일 복사
    def copy_files(image_list, split_name):
        for img_path in tqdm(image_list, desc=f"{split_name} 복사"):
            # 이미지 복사
            dst_img = Path(f"data/images/{split_name}") / img_path.name
            shutil.copy2(img_path, dst_img)

            # 마스크 복사 (같은 이름 찾기)
            mask_name = img_path.stem + ".png"  # 마스크는 보통 PNG
            mask_path = kvasir_masks / mask_name
            if not mask_path.exists():
                mask_path = kvasir_masks / img_path.name  # 같은 확장자 시도
            if not mask_path.exists():
                mask_path = kvasir_masks / (img_path.stem + ".jpg")  # JPG 시도

            if mask_path.exists():
                dst_mask = Path(f"data/masks/{split_name}") / (img_path.stem + ".png")
                shutil.copy2(mask_path, dst_mask)

    copy_files(train_images, "train")
    copy_files(val_images, "val")

    print("\n✅ 데이터 정리 완료!")
    return True


def setup_oxford_pets():
    """
    Oxford-IIIT Pet Dataset 다운로드 및 설정
    - 37 종류의 개/고양이
    - ~3,700장 이미지
    """
    print("\n" + "="*60)
    print("📥 Oxford Pets 데이터셋 다운로드")
    print("="*60)
    print("• 이미지 수: ~3,700장")
    print("• 용도: 반려동물 세분화")
    print("• 크기: ~800MB (이미지) + ~50MB (마스크)")
    print("="*60)

    # 이미지 다운로드
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    masks_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    print("\n⬇️ 이미지 다운로드 중... (약 800MB)")
    if not os.path.exists("images.tar.gz"):
        download_file(images_url, "images.tar.gz")

    print("\n⬇️ 마스크 다운로드 중... (약 50MB)")
    if not os.path.exists("annotations.tar.gz"):
        download_file(masks_url, "annotations.tar.gz")

    # 압축 해제
    print("\n📂 압축 해제 중...")
    import tarfile

    with tarfile.open("images.tar.gz", "r:gz") as tar:
        tar.extractall(".")
    with tarfile.open("annotations.tar.gz", "r:gz") as tar:
        tar.extractall(".")

    # 데이터 정리
    print("\n🔧 데이터 정리 중...")
    for folder in ["data/images/train", "data/images/val", "data/masks/train", "data/masks/val"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # trimaps 폴더에서 마스크 가져오기
    images_dir = Path("images")
    masks_dir = Path("annotations/trimaps")

    images = sorted(list(images_dir.glob("*.jpg")))
    split_idx = int(len(images) * 0.8)

    for i, img_path in enumerate(tqdm(images, desc="파일 복사")):
        split = "train" if i < split_idx else "val"

        # 이미지 복사
        shutil.copy2(img_path, f"data/images/{split}/{img_path.name}")

        # 마스크 복사
        mask_path = masks_dir / (img_path.stem + ".png")
        if mask_path.exists():
            shutil.copy2(mask_path, f"data/masks/{split}/{img_path.stem}.png")

    print("\n✅ 데이터 정리 완료!")
    return True


def setup_simple_shapes():
    """
    간단한 도형 데이터셋 생성 (테스트용)
    - 원, 사각형, 삼각형
    - 100장 자동 생성
    """
    print("\n" + "="*60)
    print("🎨 간단한 도형 데이터셋 생성")
    print("="*60)
    print("• 이미지 수: 100장")
    print("• 용도: 빠른 테스트")
    print("• 생성 시간: ~10초")
    print("="*60)

    import numpy as np
    from PIL import Image, ImageDraw
    import random

    # 폴더 생성
    for folder in ["data/images/train", "data/images/val", "data/masks/train", "data/masks/val"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    def create_shape_image(idx, split):
        """도형 이미지와 마스크 생성"""
        # 배경 생성
        width, height = 512, 512
        bg_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        image = Image.new('RGB', (width, height), bg_color)
        mask = Image.new('L', (width, height), 0)

        draw_img = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)

        # 랜덤 도형 선택
        shape_type = random.choice(['circle', 'rectangle', 'ellipse'])

        # 랜덤 위치와 크기
        x1 = random.randint(50, width - 200)
        y1 = random.randint(50, height - 200)
        x2 = x1 + random.randint(100, 200)
        y2 = y1 + random.randint(100, 200)

        # 랜덤 색상
        shape_color = (random.randint(0, 100), random.randint(0, 100), random.randint(200, 255))

        if shape_type == 'circle':
            draw_img.ellipse([x1, y1, x2, y2], fill=shape_color)
            draw_mask.ellipse([x1, y1, x2, y2], fill=255)
        elif shape_type == 'rectangle':
            draw_img.rectangle([x1, y1, x2, y2], fill=shape_color)
            draw_mask.rectangle([x1, y1, x2, y2], fill=255)
        else:  # ellipse
            draw_img.ellipse([x1, y1, x2, y1 + (y2-y1)//2], fill=shape_color)
            draw_mask.ellipse([x1, y1, x2, y1 + (y2-y1)//2], fill=255)

        # 저장
        image.save(f"data/images/{split}/shape_{idx:04d}.png")
        mask.save(f"data/masks/{split}/shape_{idx:04d}.png")

    # 이미지 생성
    print("\n🖼️ 이미지 생성 중...")
    for i in tqdm(range(80), desc="학습용"):
        create_shape_image(i, "train")
    for i in tqdm(range(20), desc="검증용"):
        create_shape_image(80 + i, "val")

    print("\n✅ 도형 데이터셋 생성 완료!")
    return True


def print_menu():
    """메뉴 출력"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              📦 SAM2 파인튜닝용 데이터셋 다운로드               ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  1️⃣  Kvasir-SEG (추천 - 초보자용)                              ║
║      • 1,000장 의료 이미지 (폴립)                               ║
║      • 다운로드: ~46MB                                         ║
║      • 마스크가 깔끔하고 학습이 잘 됨                            ║
║                                                                ║
║  2️⃣  Oxford Pets                                               ║
║      • 3,700장 반려동물 이미지                                  ║
║      • 다운로드: ~850MB                                        ║
║      • 다양한 종류의 개/고양이                                   ║
║                                                                ║
║  3️⃣  Simple Shapes (빠른 테스트용)                             ║
║      • 100장 자동 생성 (원, 사각형)                             ║
║      • 다운로드 불필요                                          ║
║      • 파인튜닝 과정 테스트용                                    ║
║                                                                ║
║  0️⃣  종료                                                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")


def main():
    print_menu()

    choice = input("선택하세요 (1/2/3/0): ").strip()

    success = False
    if choice == "1":
        success = setup_kvasir_seg()
    elif choice == "2":
        success = setup_oxford_pets()
    elif choice == "3":
        success = setup_simple_shapes()
    elif choice == "0":
        print("👋 종료합니다.")
        return
    else:
        print("❌ 잘못된 선택입니다.")
        return

    if success:
        # 결과 확인
        train_images = len(list(Path("data/images/train").glob("*")))
        val_images = len(list(Path("data/images/val").glob("*")))
        train_masks = len(list(Path("data/masks/train").glob("*")))
        val_masks = len(list(Path("data/masks/val").glob("*")))

        print("\n" + "="*60)
        print("📊 데이터셋 준비 완료!")
        print("="*60)
        print(f"  학습 이미지: {train_images}개")
        print(f"  학습 마스크: {train_masks}개")
        print(f"  검증 이미지: {val_images}개")
        print(f"  검증 마스크: {val_masks}개")
        print("="*60)
        print("\n🚀 다음 명령으로 파인튜닝을 시작하세요:")
        print("   python finetune_sam2.py")


if __name__ == "__main__":
    main()
