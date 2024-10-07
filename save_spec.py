import os
import torch
from tqdm import tqdm
from dataset import FeatureExtractorDataset

# Preprocess the dataset and save the spectrograms
def save_spectrograms(dataset, save_dirs):
    assert len(dataset.path_dir_wb) == len(save_dirs), "각 path에 대해 저장할 디렉토리가 있어야 합니다."

    bar = tqdm(range(len(dataset)), desc="Processing files")
    for idx in bar:
        spec_e, masked_spec_e, filename = dataset[idx]
        
        # 파일이 어느 경로에서 왔는지 확인하여 save_dir을 결정
        for path_idx, path_dir in enumerate(dataset.path_dir_wb):
            if filename.startswith(path_dir):
                save_dir = save_dirs[path_idx]
                break

        # filename의 상위 디렉토리 구조를 유지하여 저장
        relative_path = os.path.relpath(filename, path_dir)  # 해당 path_dir을 기준으로 상대 경로 계산
        relative_dir = os.path.dirname(relative_path)  # 파일의 디렉토리만 추출

        output_dir = os.path.join(save_dir, relative_dir)  # 저장할 경로 구성
        os.makedirs(output_dir, exist_ok=True)  # 경로가 없으면 생성

        # 저장할 파일 경로 설정
        spec_e_path = os.path.join(output_dir, f"{os.path.basename(filename)}_spec_e.pt")
        masked_spec_e_path = os.path.join(output_dir, f"{os.path.basename(filename)}_masked_spec_e.pt")

        # Save spectrograms
        # print(spec_e_path)
        # print(save_dir)
        torch.save(spec_e, spec_e_path)
        torch.save(masked_spec_e, masked_spec_e_path)

# Dataset 객체 생성 후 사용
paths = [
        # "/mnt/hdd/Dataset_BESSL_p2/MUSDB_WB_SEGMENT_/", 
        #  "/mnt/hdd/Dataset_BESSL_p2/VCTK_WB_SEGMENT_/",
         "/mnt/hdd/Dataset_BESSL_p2/FSD50K_WB_SEGMENT"
         ]
save_dirs = [
            #  "/mnt/hdd/Dataset_BESSL_p2/MUSDB_spec", 
            #  "/mnt/hdd/Dataset_BESSL_p2/VCTK_spec",
             "/mnt/hdd/Dataset_BESSL_p2/FSD_spec"
             ]

dataset = FeatureExtractorDataset(paths)
save_spectrograms(dataset, save_dirs)
