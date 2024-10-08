"""
Image readers
"""

from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Optional, Tuple

import pims
import tifffile
import zarr

from .._shared.types import ArrayLike, PathLike


class ImageReader(ABC):
    """
    Abstract class to create image readers
    classes
    """

    def __init__(self, data_path: PathLike) -> None:
        """
        Class constructor of image reader.

        Parameters
        ------------------------
        data_path: PathLike
            Path where the image is located

        """

        self.__data_path = Path(data_path)
        super().__init__()

    @abstractmethod
    def as_numpy_array(self) -> ArrayLike:
        """
        Abstract method to return the image as a numpy array.

        Returns
        ------------------------
        np.ndarray
            Numpy array with the image

        """
        pass

    @abstractproperty
    def shape(self) -> Tuple:
        """
        Abstract method to return the shape of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the shape of the image

        """
        pass

    @property
    def data_path(self) -> PathLike:
        """
        Getter to return the path where the image is located.

        Returns
        ------------------------
        PathLike
            Path of the image

        """
        return self.__data_path

    @data_path.setter
    def data_path(self, new_data_path: PathLike) -> None:
        """
        Setter of the path attribute where the image is located.

        Parameters
        ------------------------
        new_data_path: PathLike
            New path of the image

        """
        self.__data_path = Path(new_data_path)


class ZarrReader(ImageReader):
    """
    OMEZarr reader class
    """

    def __init__(
        self, data_path: PathLike, multiscale: Optional[int] = 0
    ) -> None:
        """
        Class constructor of image OMEZarr reader.

        Parameters
        ------------------------
        data_path: PathLike
            Path where the image is located

        multiscale: Optional[int]
            Desired multiscale to read from the image. Default: 0 which is
            supposed to be the highest resolution

        """
        super().__init__(Path(data_path).joinpath(str(multiscale)))

    @property
    def shape(self):
        """
        Method to return the shape of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the shape of the image

        """
        return zarr.open(self.data_path, "r").shape

    def as_numpy_array(self):
        """
        Method to return the image as a numpy array.

        Returns
        ------------------------
        np.ndarray
            Numpy array with the image

        """
        return zarr.open(self.data_path, "r")[:]


class N5Reader(ImageReader):
    """
    N5 Reader class
    """

    def __init__(self, data_path: PathLike):
        """
        Class constructor of image N5 reader.

        Parameters
        ------------------------
        data_path: PathLike
            Path where the image is located

        """
        super().__init__(data_path)

    @property
    def shape(self) -> Tuple:
        """
        Return the shape of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the shape of the image

        """
        return zarr.open(self.data_path, "r").shape

    def as_numpy_array(self, dataset: str) -> ArrayLike:
        """
        Return the image as a numpy array.

        Parameters
        ------------------------
        dataset: str
            String representing the dataset
            inside of the n5 data

        Returns
        ------------------------
        np.ndarray
            Numpy array with the image

        """
        store = zarr.N5Store(self.data_path)
        root = zarr.open(store)

        # Accessing the multiscale
        data = root[dataset]

        return data[:]


class TiffReader(ImageReader):
    """
    TiffReader class
    """

    def __init__(self, data_path: PathLike) -> None:
        """
        Class constructor of image Tiff reader.

        Parameters
        ------------------------
        data_path: PathLike
            Path where the image is located

        """
        super().__init__(Path(data_path))
        self.tiff = tifffile.TiffFile(self.data_path)

    def as_numpy_array(self) -> ArrayLike:
        """
        Abstract method to return the image as a numpy array.

        Returns
        ------------------------
        np.ndarray
            Numpy array with the image

        """
        return self.tiff.asarray()

    @property
    def shape(self) -> Tuple:
        """
        Abstract method to return the shape of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the shape of the image

        """
        with pims.open(str(self.data_path)) as imgs:
            shape = (len(imgs),) + imgs.frame_shape

        return shape

    def close_handler(self) -> None:
        """
        Closes image handler
        """
        if self.tiff is not None:
            self.tiff.close()
            self.tiff = None

    def __del__(self) -> None:
        """Overriding destructor to safely close image"""
        self.close_handler()