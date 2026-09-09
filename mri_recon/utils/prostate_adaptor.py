import os
import glob
import h5py
import numpy as np
import torch


class FastMRIProstateDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path: str, num_samples: None | int = None, slice_index: str = "middle"
    ) -> None:
        self.data_path = data_path
        self.num_samples = num_samples
        self.slice_index = slice_index
        self.image_data, self.sample_names = self.get_image_data()

        if num_samples is not None:
            self.image_data = self.image_data[:num_samples]
        else:
            self.num_samples = len(self.image_data)

    def get_image_data(self) -> tuple[np.ndarray, list[str]]:
        image_result_list = []
        sample_name_list = []
        for sample_idx, filename in enumerate(
            sorted(glob.glob(os.path.join(self.data_path, "*.h5")))
        ):
            if (self.num_samples is not None) and (sample_idx >= self.num_samples):
                break
            try:
                with h5py.File(filename, "r") as hf:
                    image_recon = hf["reconstruction_rss"][:]

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

            if self.slice_index == "middle":
                image_result_list.append(image_recon[image_recon.shape[0] // 2])
            else:
                image_result_list.extend(
                    [image_recon[i, :, :] for i in range(image_recon.shape[0])]
                )
            sample_name = (
                os.path.basename(filename)
                .split(".")[0]
                .replace("file_prostate_", "")
                .replace("_", "-")
            )
            sample_name_list.append(sample_name)

        return image_result_list, sample_name_list

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # add batch dimension and convert to torch.Tensor
        return {
            "image": torch.from_numpy(self.image_data[idx]).unsqueeze(0),
            "sample_name": self.sample_names[idx],
        }
