# Előre betanított 3D Pontfelhő (LiDAR / Radar) Neurális Hálózati Modellek

Az alábbi táblázat összefoglalja a személygépjárművek valós idejű objektumdetektálási és automata vészfékezési (AEB) feladataihoz legjobban használható, előre betanított neurális hálózati architektúrákat, azok Hugging Face-es és egyéb iparági elérhetőségeivel együtt.

| Modell architektúra | Elérhetőség a Hugging Face-en | Legjobb külső forrás (Model Zoo) | Támogatott szenzorok | Sebesség és főbb jellegzetességek |
| :--- | :--- | :--- | :--- | :--- |
| **PointPillars** | `torch-pointcloud/pointpillars-multihead.nuscenes.openpcdet`<br>`AMD-PAVS-AI/pointpillars` | **OpenPCDet**, **MMDetection3D**, **NVIDIA Isaac ROS** | LiDAR *(kiegészíthető Radarral)* | **A valós idejű bajnok (50–100+ FPS).** Függőleges oszlopokba (*pillars*) rendezi a pontokat, és gyors 2D konvolúciót használ. Vészfékezéshez a legjobb belépő modell. |
| **CenterPoint** | `qualcomm/CenterPoint`<br>`OpenExplorer/centerpoint_pointpillar` | **OpenPCDet**, **MMDetection3D** (`centerpoint` konfigok) | LiDAR / Radar BEV | **Középpont-alapú (30–40 FPS).** Nem fix befoglaló dobozokat tippel, hanem hőtérképen keresi az objektumok középpontját, és kiválóan becsüli a **relatív sebességet**. |
| **BEVFusion** | `OpenExplorer/bevfusion_pointpillar_henet_multisensor_multitask` | **MIT-HAN-Lab / BEVFusion (GitHub)**, **MMDetection3D** | LiDAR + Radar + Kamera | **Multiszenzoros fúzióra tervezve (15–30 FPS).** Közös felülnézeti (BEV – *Bird's-Eye View*) térbe vetíti a LiDAR geometriai pontosságát és a Radar/Kamera adatait. |
| **CenterFusion** | *Nincs hivatalos HF súly (GitHubon elérhető)* | **GitHub:** `HengWeiBin/CenterFusionDetect3D`<br>`nabati/CenterFusion` | Radar + Kamera / LiDAR | **Radar-specialista.** Kifejezetten arra épít, hogy a ritka radar-pontfelhő Doppler-sebességét pontosan hozzárendelje a detektált 3D objektumokhoz. |
| **VoxelNeXt** | `torch-pointcloud` kollekciók | **OpenPCDet** (`VoxelNeXt` nuScenes és Waymo súlyok) | LiDAR | **Modern, teljesen ritka (sparse) CNN.** Közvetlenül a 3D voxelekből prediktál felesleges és lassító utófeldolgozási lépések (pl. NMS) nélkül. |

---

## Hasznos linkek és keretrendszerek a letöltéshez

* **OpenMMLab – MMDetection3D:** [https://github.com/open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
* **OpenPCDet:** [https://github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* **MIT-HAN-Lab BEVFusion:** [https://github.com/mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion)
* **NVIDIA NGC / Isaac ROS (TensorRT optimalizált modellek):** [https://catalog.ngc.nvidia.com/](https://catalog.ngc.nvidia.com/)