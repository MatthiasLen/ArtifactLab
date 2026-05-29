import os
import glob
from typing import Dict, Tuple, Sequence
import h5py
import numpy as np
from skimage.util import view_as_windows
from tempfile import NamedTemporaryFile as NTF
from tifffile import imwrite
import torch
import xml.etree.ElementTree as etree


class Grappa:
    def __init__(
        self, kspace: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), coil_axis: int = -1
    ) -> None:
        self.kspace = kspace
        self.kernel_size = kernel_size
        self.coil_axis = coil_axis
        self.lamda = 0.01

        self.kernel_var_dict = self.get_kernel_geometries()

    def get_kernel_geometries(self):
        """
        Extract unique kernel geometries based on a slice of kspace data

        Returns
        -------
        geometries : dict
            A dictionary containing the following keys:
            - 'patches': an array of overlapping patches from the k-space data.
            - 'patch_indices': an array of unique patch indices.
            - 'holes_x': a dictionary of x-coordinates for holes in each patch.
            - 'holes_y': a dictionary of y-coordinates for holes in each patch.

        Notes
        -----
        This function extracts unique kernel geometries from a slice of k-space data.
        The geometries correspond to overlapping patches that contain at least one hole.
        A hole is defined as a region of k-space data where the absolute value of the
        complex signal is equal to zero. The function returns a dictionary containing
        information about the patches and holes, which can be used to compute weights
        for each geometry using the GRAPPA algorithm.

        """
        self.kspace = np.moveaxis(self.kspace, self.coil_axis, -1)

        # Quit early if there are no holes
        if np.sum((np.abs(self.kspace[..., 0]) == 0).flatten()) == 0:
            return np.moveaxis(self.kspace, -1, self.coil_axis)

        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx / 2), int(ky / 2)
        nc = self.kspace.shape[-1]

        self.kspace = np.pad(self.kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")

        mask = np.ascontiguousarray(np.abs(self.kspace[..., 0]) > 0)

        with NTF() as fP:
            # Get all overlapping patches from the mask
            P = np.memmap(
                fP,
                dtype=mask.dtype,
                mode="w+",
                shape=(mask.shape[0] - 2 * kx2, mask.shape[1] - 2 * ky2, 1, kx, ky),
            )
            P = view_as_windows(mask, (kx, ky))
            Psh = P.shape[:]  # save shape for unflattening indices later
            P = P.reshape((-1, kx, ky))

            # Find the unique patches and associate them with indices
            P, iidx = np.unique(P, return_inverse=True, axis=0)

            # Filter out geometries that don't have a hole at the center.
            # These are all the kernel geometries we actually need to
            # compute weights for.
            validP = np.argwhere(~P[:, kx2, ky2]).squeeze()

            # ignore empty patches
            invalidP = np.argwhere(np.all(P == 0, axis=(1, 2)))
            validP = np.setdiff1d(validP, invalidP, assume_unique=True)

            validP = np.atleast_1d(validP)

            # Give P back its coil dimension
            P = np.tile(P[..., None], (1, 1, 1, nc))

            holes_x = {}
            holes_y = {}
            for ii in validP:
                # x, y define where top left corner is, so move to ctr,
                # also make sure they are iterable by enforcing atleast_1d
                idx = np.unravel_index(np.argwhere(iidx == ii), Psh[:2])
                x, y = idx[0] + kx2, idx[1] + ky2
                x = np.atleast_1d(x.squeeze())
                y = np.atleast_1d(y.squeeze())

                holes_x[ii] = x
                holes_y[ii] = y

        return {"patches": P, "patch_indices": validP, "holes_x": holes_x, "holes_y": holes_y}

    def compute_weights(self, calib: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute the GRAPPA weights for each slice in the input calibration data.

        Parameters:
        ----------
        calib : numpy.ndarray
            Calibration data with shape (Nx, Nc, Ny) where Nx, Ny are the size of the image in the x and y dimensions,
            respectively, and Nc is the number of coils.

        Returns:
        -------
        weights : dict
            A dictionary of GRAPPA weights for each patch index.

        Notes:
        -----
        The GRAPPA algorithm is used to estimate the missing k-space data in undersampled MRI acquisitions.
        The algorithm used to compute the GRAPPA weights involves first extracting patches from the calibration data,
        and then solving a linear system to estimate the weights. The resulting weights are stored in a dictionary
        where the key is the patch index. The equation to solve for the weights involves taking the product of the
        sources and the targets in the patch domain, and then regularizing the matrix using Tikhonov regularization.
        The function uses numpy's `memmap` to store temporary files to avoid overwhelming memory usage.
        """

        calib = np.moveaxis(calib, self.coil_axis, -1)
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx / 2), int(ky / 2)
        nc = calib.shape[-1]

        calib = np.pad(calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")

        # Store windows in temporary files so we don't overwhelm memory
        with NTF() as fA:
            # Get all overlapping patches of ACS
            try:
                A = np.memmap(
                    fA,
                    dtype=calib.dtype,
                    mode="w+",
                    shape=(calib.shape[0] - 2 * kx, calib.shape[1] - 2 * ky, 1, kx, ky, nc),
                )
                A[:] = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
            except ValueError:
                A = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))

            weights = {}

            for ii in self.kernel_var_dict["patch_indices"]:
                # Get the sources by masking all patches of the ACS and
                # get targets by taking the center of each patch. Source
                # and targets will have the following sizes:
                #     S : (# samples, N possible patches in ACS)
                #     T : (# coils, N possible patches in ACS)
                # Solve the equation for the weights: using numpy.linalg.solve,
                # and Tikhonov regularization for better conditioning:
                #     SW = T
                #     S^HSW = S^HT
                #     W = (S^HS)^-1 S^HT
                #  -> W = (S^HS + lamda I)^-1 S^HT

                S = A[:, self.kernel_var_dict["patches"][ii, ...]]
                T = A[:, kx2, ky2, :]
                ShS = S.conj().T @ S
                ShT = S.conj().T @ T
                lamda0 = self.lamda * np.linalg.norm(ShS) / ShS.shape[0]
                weights[ii] = np.linalg.solve(ShS + lamda0 * np.eye(ShS.shape[0]), ShT).T

        return weights

    def apply_weights(self, kspace: np.ndarray, weights: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Applies the computed GRAPPA weights to the k-space data.

        Parameters:
        ----------
            kspace : numpy.ndarray
                The k-space data to apply the weights to.

            weights : dict
                A dictionary containing the GRAPPA weights to apply.

        Returns:
        -------
            numpy.ndarray: The reconstructed data after applying the weights.
        """

        # fin_shape = kspace.shape[:]

        # Put the coil dimension at the end
        kspace = np.moveaxis(kspace, self.coil_axis, -1)

        # Get shape of kernel
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx / 2), int(ky / 2)

        # adjustment factor for odd kernel size
        adjx = np.mod(kx, 2)
        adjy = np.mod(ky, 2)

        # Pad kspace data
        kspace = np.pad(kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")

        with NTF() as frecon:
            # Initialize recon array
            recon = np.memmap(frecon, dtype=kspace.dtype, mode="w+", shape=kspace.shape)

            for ii in self.kernel_var_dict["patch_indices"]:
                for xx, yy in zip(
                    self.kernel_var_dict["holes_x"][ii], self.kernel_var_dict["holes_y"][ii]
                ):
                    # Collect sources for this hole and apply weights
                    S = kspace[xx - kx2 : xx + kx2 + adjx, yy - ky2 : yy + ky2 + adjy, :]
                    S = S[self.kernel_var_dict["patches"][ii, ...]]
                    recon[xx, yy, :] = (weights[ii] @ S[:, None]).squeeze()

            return np.moveaxis((recon[:] + kspace)[kx2:-kx2, ky2:-ky2, :], -1, self.coil_axis)


def et_query(
    root: etree.Element, qlist: Sequence[str], namespace: str = "http://www.ismrm.org/ISMRMRD"
) -> str:
    """
    ElementTree query function.

    This function queries an XML document using ElementTree.

    Parameters:
    -----------
    root : Element
        Root of the XML document to search through.
    qlist : Sequence of str
        A sequence of strings for nested searches, e.g., ["Encoding", "matrixSize"].
    namespace : str, optional
        XML namespace to prepend query.

    Returns:
    --------
    str
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def get_padding(hdr: str) -> float:
    """
    Extract the padding value from an XML header string.

    Parameters:
    -----------
    hdr : str
        The XML header string.

    Returns:
    --------
    float
        The padding value calculated as (x - max_enc)/2, where x is the readout dimension and
        max_enc is the maximum phase-encoding dimension.
    """
    et_root = etree.fromstring(hdr)
    lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
    enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1
    enc = ["encoding", "encodedSpace", "matrixSize"]
    enc_x = int(et_query(et_root, enc + ["x"]))
    padding = (enc_x - enc_limits_max) / 2

    print(f"Padding from header: {padding}")
    return padding


def get_padding_from_image_recon(k_space_shape: np.ndarray, image_recon_shape: np.ndarray) -> float:
    """
    Calculate the padding value based on the shapes of the k-space data and the reconstructed image.

    Parameters:
    -----------
    k_space_shape : np.ndarray
        The shape of the k-space data, typically in the format (num_avg, num_slices, num_coils, num_ro, num_pe).
    image_recon_shape : np.ndarray
        The shape of the reconstructed image, typically in the format (num_avg, num_slices, num_coils, num_x, num_y).

    Returns:
    --------
    float
        The padding value calculated as (x - max_enc)/2, where x is the readout dimension from the k-space shape and
        max_enc is the maximum phase-encoding dimension from the reconstructed image shape.
    """
    enc_limits_max = image_recon_shape[
        -1
    ]  # Assuming last dimension corresponds to phase-encoding direction
    enc_x = k_space_shape[-2]  # Assuming second to last dimension corresponds to readout direction
    padding = (enc_x - enc_limits_max) / 2

    print(f"Padding from image recon shapes: {padding}")
    return padding


def zero_pad_kspace_slice_hdr(
    hdr: str, unpadded_kspace: np.ndarray, image_recon_shape: np.ndarray
) -> np.ndarray:
    """
    Perform zero-padding on k-space data to have the same number of
    points in the x- and y-directions.

    Parameters
    ----------
    hdr : str
        The XML header string.
    unpadded_kspace : array-like of shape (ro , coils, pe)
        The k-space data to be padded.

    Returns
    -------
    padded_kspace : ndarray of shape (ro_padded, coils, pe_padded)
        The zero-padded k-space data, where ro_padded and pe_padded are
        the dimensions of the readout and phase-encoding directions after
        padding.

    Notes
    -----
    The padding value is calculated using the `get_padding` function, which
    extracts the padding value from the XML header string. If the difference
    between the readout dimension and the maximum phase-encoding dimension
    is not divisible by 2, the padding is applied asymmetrically, with one
    side having an additional zero-padding.

    """
    padding = get_padding(hdr)
    print(f"Calculated padding: {padding}")
    padding2 = get_padding_from_image_recon(unpadded_kspace.shape, image_recon_shape)
    print(f"Calculated padding from image recon shapes ({image_recon_shape}): {padding2}")
    if padding % 2 != 0:
        padding_left = int(np.floor(padding))
        padding_right = int(np.ceil(padding))
    else:
        padding_left = int(padding)
        padding_right = int(padding)
    padded_kspace = np.pad(unpadded_kspace, ((0, 0), (0, 0), (padding_left, padding_right)))

    return padded_kspace


class FastMRIProstateDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, num_samples: None | int = None) -> None:
        self.data_path = data_path
        self.num_samples = num_samples
        self.kspace_data, self.image_data = self.pre_calc_kspace_with_grappa()

        if num_samples is not None:
            self.kspace_data = self.kspace_data[:num_samples]
            self.image_data = self.image_data[:num_samples]
        else:
            self.num_samples = len(self.kspace_data)

    def pre_calc_kspace_with_grappa(self) -> np.ndarray:
        kspace_result_list = []
        image_result_list = []
        for sample_idx, filename in enumerate(glob.glob(os.path.join(self.data_path, "*.h5"))):
            if (self.num_samples is not None) and (sample_idx >= self.num_samples):
                break
            try:
                with h5py.File(filename, "r") as hf:
                    kspace_data = hf["kspace"][:]
                    calibration_data = hf["calibration_data"][:]
                    hdr = hf["ismrmrd_header"][()]
                    image_recon = hf["reconstruction_rss"][:]

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

            # kspace data: (num_avg, num_slices, num_coils, num_ro, num_pe)
            # Calib_data: (num_slices, num_coils, num_pe_cal)

            # middle slice:
            kspace_middle_slice = kspace_data[:, kspace_data.shape[1] // 2]
            imwrite(
                "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_middle_slice.tiff",
                kspace_middle_slice,
            )
            cal_middle_slice = calibration_data[calibration_data.shape[0] // 2]
            print(
                f"Processing file {filename} with kspace shape {kspace_middle_slice.shape} and calib shape {cal_middle_slice.shape}"
            )
            imwrite(
                "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/calibration_middle_slice.tiff",
                cal_middle_slice,
            )

            #######

            kspace_slice_regridded = kspace_data[0, 0]
            grappa_obj = Grappa(
                np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1
            )

            kspace_slice_regridded_2 = kspace_data[1, 0]
            grappa_obj_2 = Grappa(
                np.transpose(kspace_slice_regridded_2, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1
            )

            # calculate GRAPPA weights for middle slice:

            calibration_regridded = calibration_data[calibration_data.shape[0] // 2, ...]
            grappa_weight = grappa_obj.compute_weights(
                np.transpose(calibration_regridded, (2, 0, 1))
            )
            grappa_weight2 = grappa_obj_2.compute_weights(
                np.transpose(calibration_regridded, (2, 0, 1))
            )

            ####
            kspace_post_grappa_slice = np.zeros(shape=kspace_middle_slice.shape, dtype=complex)
            kspace_post_grappa_slice_padded: Dict[int, np.ndarray] = {}
            for average, grappa_obj, grappa_weight_dict in zip(
                [0, 1, 2],
                [grappa_obj, grappa_obj_2, grappa_obj],
                [grappa_weight, grappa_weight2, grappa_weight],
            ):
                kspace_slice_regridded = kspace_middle_slice[average, ...]
                kspace_post_grappa = grappa_obj.apply_weights(
                    np.transpose(kspace_slice_regridded, (2, 0, 1)), grappa_weight_dict
                )
                kspace_post_grappa_slice[average] = np.moveaxis(
                    np.moveaxis(kspace_post_grappa, 0, 1), 1, 2
                )

                # pad:
                kspace_post_grappa_slice_padded[average] = zero_pad_kspace_slice_hdr(
                    hdr, kspace_post_grappa_slice[average], image_recon.shape
                )

            # stack k-space data for all averages:
            kspace_result_list.append(
                np.stack(list(kspace_post_grappa_slice_padded.values()), axis=0)
            )
            image_result_list.append(image_recon)
        return kspace_result_list, image_result_list

    def __len__(self) -> int:
        return len(self.kspace_data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        kspace = self.kspace_data[idx]
        return torch.from_numpy(kspace), torch.from_numpy(self.image_data[idx])
