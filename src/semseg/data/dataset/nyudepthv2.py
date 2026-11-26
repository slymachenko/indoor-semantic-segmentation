from pathlib import Path
import h5py

from semseg.data.dataset import SegmentationDataset

from typing import Any
from numpy.typing import NDArray


class NYUDepthV2Dataset(SegmentationDataset):
    """Dataset reader for NYU Depth V2."""

    def __init__(self, dataset_path: Path | str):
        self.dataset_path = Path(dataset_path)
        self.file = h5py.File(self.dataset_path, "r")

        self.images = self.file["images"]
        self.labels = self.file["labels"]

        self.n_images = len(self.images)
        self.class_names = self._read_names("names")
        self.scene_names = self._read_names("scenes")

    def _read_names(self, key: str) -> list[str]:
        """Decode MATLAB character arrays."""
        names = []

        for ref in self.file[key][0]:
            # chars = self.file[ref][()].ravel()
            # name = "".join(chr(int(char)) for char in chars if char != 0)
            # names.append(name)
            chars = self.file[ref][:]
            name = "".join(chr(int(char[0])) for char in chars if char != 0)
            names.append(name)

        return names

    def get_pair(self, index: int) -> tuple[NDArray[Any], NDArray[Any]]:
        if not 0 <= index < len(self):
            raise IndexError(f"Invalid dataset index: {index}")

        image = self.images[index].transpose(2, 1, 0)
        mask = self.labels[index].T

        return image, mask

    def get_class_id(self, class_name: str) -> int:
        try:
            return self.class_names.index(class_name) + 1
        except ValueError:
            raise ValueError(f"Unknown class '{class_name}'.") from None

    def __len__(self) -> int:
        return self.n_images

    def close(self) -> None:
        if self.file:
            self.file.close()
