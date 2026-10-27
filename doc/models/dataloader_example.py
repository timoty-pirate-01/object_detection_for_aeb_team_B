from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mmdet3d.apis import LidarDet3DInferencer


class CustomRawPointCloudDataset(Dataset):
    """
    Saját PyTorch Dataset, amely nyers LiDAR/Radar méréseket készít elő
    az MMDetection3D (nuScenes) modellek számára.
    """

    def __init__(self, frames: list[np.ndarray], target_dim: int = 5) -> None:
        self.frames = frames
        self.target_dim = target_dim

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> np.ndarray:
        points = self.frames[idx].astype(np.float32)
        num_points, current_dim = points.shape

        # Ha a saját mérésünk csak 3 (x,y,z) vagy 4 (x,y,z,i) oszlopos,
        # kiegészítjük nullákkal 5 oszloposra (x, y, z, intensity, timestamp=0.0),
        # hogy a nuScenes előtanított modell (pl. CenterPoint) elfogadja!
        if current_dim < self.target_dim:
            padding = np.zeros((num_points, self.target_dim - current_dim), dtype=np.float32)
            points = np.hstack((points, padding))
        elif current_dim > self.target_dim:
            points = points[:, :self.target_dim]

        return points


def custom_collate_fn(batch: list[np.ndarray]) -> list[np.ndarray]:
    """
    Mivel minden képkockában (frame-ben) eltérő számú LiDAR pont van (N1 != N2),
    nem lehet őket sima torch.stack-kel összefűzni. Listaként adjuk tovább!
    """
    return batch


def run_prediction_pipeline() -> None:
    # 1. Szimuláljuk a 'raw_measurements' betöltését (3 db képkocka eltérő pontszámmal)
    # A valóságban ide jön a te fájlod beolvasása (lásd a 4. pontot lentebb)
    dummy_frames = [
        np.random.uniform(-20, 20, size=(12000, 4)).astype(np.float32),
        np.random.uniform(-20, 20, size=(14500, 4)).astype(np.float32),
    ]

    # 2. Saját Dataset és DataLoader példányosítása
    dataset = CustomRawPointCloudDataset(frames=dummy_frames, target_dim=5)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # 3. nuScenes előtanított CenterPoint (PointPillars backbone) betöltése
    # Első futáskor az mmdet3d automatikusan letölti a konfigot és a .pth súlyokat
    inferencer = LidarDet3DInferencer(
        model="centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # 4. Predikció futtatása a DataLoader-en keresztül (In-Memory NumPy tömbökkel!)
    for frame_idx, batch in enumerate(dataloader):
        point_cloud_np = batch[0]  # Alak: (N, 5), típus: np.float32

        # Közvetlenül a NumPy tömböt adjuk át (ROS 2 kompatibilis működés!)
        result = inferencer(
            inputs={"points": point_cloud_np},
            no_save_vis=True,
        )

        preds = result["predictions"][0]
        bboxes_3d = np.array(preds["bboxes_3d"])  # [x, y, z, dx, dy, dz, yaw, vx, vy]
        scores_3d = np.array(preds["scores_3d"])
        labels_3d = np.array(preds["labels_3d"])

        # Szűrés 50% feletti konfidenciára
        valid_mask = scores_3d > 0.5
        print(f"Frame #{frame_idx}: {np.sum(valid_mask)} db objektum detektálva.")


if __name__ == "__main__":
    run_prediction_pipeline()