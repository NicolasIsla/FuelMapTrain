import json
import os
from datetime import datetime
from typing import Union, Dict 
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from einops import rearrange

from shapeft.datasets.base import RawGeoFMDataset, temporal_subsampling


def prepare_dates(date_dict, reference_date):
    if type(date_dict) is str:
        date_dict = json.loads(date_dict)
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return torch.tensor(d.values)


class FuelMap(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        reference_date: str = "2020-09-01", 
        cover=0,
        obj="class"
    ):
        super(FuelMap, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download
        )

        print("Reading patch metadata...")
        self.obj = obj
        self.meta_patch = gpd.read_file(os.path.join(root_path, "metadata.geojson"))
        if cover > 0:
            self.meta_patch = self.meta_patch[self.meta_patch["cover"] > cover].copy()

        if "Fold" not in self.meta_patch.columns:
            n = len(self.meta_patch)
            folds = np.tile(np.arange(1, 6), n // 5 + 1)[:n]
            self.meta_patch["Fold"] = folds

        assert split in ["train", "val", "test"], "Invalid split"
        if split == "train":
            folds = [1, 2, 3]
        elif split == "val":
            folds = [4]
        else:
            folds = [5]

        self.modalities = ["S2", "S1_asc", "S1_des", "elevation", "mTPI", "landforms"]
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.num_classes = 11

        self.meta_patch = pd.concat(
            [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
        )
        self.meta_patch.index = self.meta_patch["id"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        self.memory_dates = {}

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        self.date_tables = {s: None for s in self.modalities}

        for s in ["S2", "S1_asc", "S1_des"]:
            dates = self.meta_patch["dates_{}".format(s)]
            self.date_range = np.array(range(-200, 600))

            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_str in dates.items():
                if isinstance(date_str, str):
                    date_list = date_str.split(",")
                else:
                    continue
                d = pd.Series(date_list).apply(
                    lambda x: (datetime.strptime(x, "%Y-%m-%d") - self.reference_date).days
                )
                date_table.loc[pid, d.values] = 1

            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        indices = np.where(self.date_tables[sat][id_patch] == 1)[0]
        indices = indices[indices < len(self.date_range)]
        return torch.tensor(self.date_range[indices], dtype=torch.int32)

    
    def __getitem__(self, i: int) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]]:
        line = self.meta_patch.iloc[i]
        id_patch = self.id_patches[i]
        name = line["id"]

        target = torch.from_numpy(
            np.load(os.path.join(self.root_path, f"ANNOTATIONS_{self.obj}", f"{name}.npy"))
        )

        data = {}
        metadata = {}

        for modality in self.modalities:
            path = os.path.join(self.root_path, f"DATA_{modality}", f"{name}.npy")
            array = np.load(path)

            if array.ndim == 4:
                # Datos multitemporales: (T, C, H, W)
                indexes = torch.linspace(0, array.shape[0] - 1, self.multi_temporal, dtype=torch.long)
                tensor = torch.from_numpy(array).to(torch.float32)[indexes]
                tensor = rearrange(tensor, "t c h w -> c t h w")

            elif array.ndim == 2:
                # Estático monocanal: (H, W)
                tensor = torch.from_numpy(array).to(torch.float32).unsqueeze(0).unsqueeze(1)  # (1, 1, H, W)
                tensor = tensor.repeat(1, self.multi_temporal, 1, 1)  # → (1, T, H, W)

            else:
                raise ValueError(f"Unsupported array shape {array.shape} for modality {modality}")

            # Mantener cada modalidad por separado como antes
            data[modality] = tensor

            if modality in ["S2", "S1_asc", "S1_des"]:
                metadata[modality] = self.get_dates(id_patch, modality)

        if self.obj == "class":
            return {
                "image": {
                    "optical": data["S2"],
                    "sar_asc": data["S1_asc"],
                    "sar_desc": data["S1_des"],
                    "elevation": data["elevation"],
                    "mTPI": data["mTPI"],
                    "landforms": data["landforms"]

                },
                "target": target.to(torch.int64),
                "metadata": metadata
            }
        else:
            return {
                "image": data,
                "target": target.to(torch.float32),
                "metadata": metadata
            }
    @staticmethod
    def download():
        pass