# MMDetection3D összefoglaló: modellválasztás és DataLoader

## 1. Modellválasztás: nuScenes vs. KITTI

**Névtisztázás:** a sima MMDetection a 2D-s képfeldolgozó könyvtár. A projekthez az OpenMMLab 3D-s testvére, az **MMDetection3D (mmdet3d)** kell, a modern, MMEngine alapú **v1.1+ / v1.4+** verzióban.

Ha a cél a predikció (inferencia) saját adatokon és a későbbi ROS 2 vészfékezés, számít, hogy nuScenes vagy KITTI előtanított modellt választunk, mert **eltérő DataLoader struktúrát** várnak.

| Jellemző | CenterPoint / PointPillars (nuScenes súlyok) | PointPillars (KITTI súlyok) |
|---|---|---|
| MMDetection3D konfig név | `centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d` | `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class` |
| Elvárt bemeneti dimenzió (`load_dim`) | 5 érték/pont: `[x, y, z, intensity, timestamp_lag]` | 4 érték/pont: `[x, y, z, intensity]` |
| Látószög (FOV) | 360° (teljes környezet az autó körül) | Csak előre néző (kb. 90°, `x > 0`) |
| Sebességbecslés | **Igen**, prediktálja a sebességvektort (`vx`, `vy`) | **Nem**, csak pozíciót és méretet ad |
| Mikor válaszd? | **Ajánlott a projekthez.** A sebességbecslés és a Radar-kompatibilitás miatt vészfékezéshez (AEB) sokkal jobb. | Gyors tesztelésre egyszerű, 4 oszlopos pontfelhőn. |

* **A nuScenes és a KITTI két különböző önvezetéses adathalmaz (dataset). Mindkettő LiDAR-felvételeket és 3D-s annotációkat tartalmaz, a modellek ezeken tanultak. Hogy az előtanított súlyok melyiken készültek, az határozza meg, mit várnak bemenetként és mit tudnak kimenetként. Összességében a nuScenes egy több fajta adaton betanított "okosabb" és robosztusabb adathalmaz, a KITTI pedig egy egyszerűbb és kevesebb fajta adatot tartlamazó kisebb adathalmaz.**

---

## 2. A DataLoader működése és a ROS 2 „memória-trükk”

Predikciókor a háttérben egy **Adatcsővezeték (Data Pipeline)** fut le. A nuScenes modellek konfigurációjában:

- `LoadPointsFromFile(coord_type='LIDAR', load_dim=5, use_dim=5)`: beolvassa a nyers bináris fájlt, és 5 oszlopos (`float32`) mátrixot készít belőle.
- `Pack3DDetInputs(keys=['points'])`: a PyTorch tenzort becsomagolja egy `Det3DDataSample` objektumba, amit a neurális háló vár.

### Két gyakorlati szabály a saját adatokhoz (`raw_measurements`)

**1. Dimenzió-párnázás (padding)**

Ha a `raw_measurements` fájlban pontonként csak 3 (`x, y, z`) vagy 4 (`x, y, z, intensity`) adat van, de nuScenes modellt használunk, a modell `IndexError`-t dob, mert keresi az 5. oszlopot (a nuScenes-ben az időbélyeg-különbség, `dt`).

> **Megoldás:** a saját DataLoaderben ki kell egészíteni a mátrixot egy 5., nullákból álló (`0.0`) oszloppal.

**2. ROS 2 memória-trükk (nincs fájlból olvasás)**

- Pythonban, meglévő méréseken tesztelve használható PyTorch `DataLoader`, amely fájlokból vagy a `raw_measurements`-ből tölti be a képkockákat.
- ROS 2-ben **soha ne írjuk ki a pontfelhőt a merevlemezre**, hogy a modell onnan olvassa be.
- Az mmdet3d modern inferencere közvetlenül **memóriából**, egy `np.ndarray` (NumPy tömb) átadásával is tud prediktálni.
