import os
from tempfile import NamedTemporaryFile as NTF
from typing import Dict, Tuple, Optional, Sequence
import xml.etree.ElementTree as etree

import h5py
import numpy as np
from numpy.fft import fftshift, ifftshift, ifftn
from tifffile import imwrite
from skimage.util import view_as_windows


def center_crop_im(im_3d: np.ndarray, crop_to_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop an image to a given size.

    Parameters:
    -----------
    im_3d : numpy.ndarray
        Input image of shape (slices, x, y).
    crop_to_size : tuple
        Tuple containing the target size for x and y dimensions.

    Returns:
    --------
    numpy.ndarray
        Center cropped image of size {slices, x_cropped, y_cropped}.
    """
    x_crop = im_3d.shape[-1] / 2 - crop_to_size[0] / 2
    y_crop = im_3d.shape[-2] / 2 - crop_to_size[1] / 2

    return im_3d[
        :, int(y_crop) : int(crop_to_size[1] + y_crop), int(x_crop) : int(crop_to_size[0] + x_crop)
    ]


def ifftnd(kspace: np.ndarray, axes: Optional[Sequence[int]] = [-1]) -> np.ndarray:
    """
    Compute the n-dimensional inverse Fourier transform of the k-space data along the specified axes.

    Parameters:
    -----------
    kspace: np.ndarray
        The input k-space data.
    axes: list or tuple, optional
        The list of axes along which to compute the inverse Fourier transform. Default is [-1].

    Returns:
    --------
    img: ndarray
        The output image after inverse Fourier transform.
    """

    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))

    return img


def create_coil_combined_im(multicoil_multislice_kspace: np.ndarray) -> np.ndarray:
    """
    Create a coil combined image from a multicoil-multislice k-space array.

    Parameters:
    -----------
    multicoil_multislice_kspace : array-like
        Input k-space data with shape (slices, coils, readout, phase encode).

    Returns:
    --------
    image_mat : array-like
        Coil combined image data with shape (slices, x, y).
    """

    k = multicoil_multislice_kspace
    image_mat = np.zeros((k.shape[0], k.shape[2], k.shape[3]))
    for i in range(image_mat.shape[0]):
        data_sl = k[i, :, :, :]
        image = ifftnd(data_sl, [1, 2])
        image_rss = rss(image, axis=0)
        image_mat[i, :, :] = np.flipud(image_rss)
        if i == 15:
            print("data_sl.shape:", data_sl.shape)
            print("image.shape:", image.shape)
            print("image flipped: ", np.flipud(image).shape)
            imwrite(
                "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/image_slice_15_kspace.tiff",
                np.abs(data_sl[0, :, :]),
            )
            imwrite(
                "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/image_slice_15_ifft.tiff",
                np.abs(image),
            )
            imwrite(
                "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/image_slice_15_rss.tiff",
                image_rss,
            )
            imwrite(
                "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/image_slice_15.tiff",
                image_mat[i, :, :],
            )
    return image_mat


def rss(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the Root Sum-of-Squares (RSS) value of a complex signal along a specified axis.

    Parameters
    ----------
    sig : np.ndarray
        The complex signal to compute the RMS value of.
    axis : int, optional
        The axis along which to compute the RMS value. Default is -1.

    Returns
    -------
    rss : np.ndarray
        The RSS value of the complex signal along the specified axis.
    """
    return np.sqrt(np.sum(abs(sig) ** 2, axis))


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

    return padding


def zero_pad_kspace_hdr(hdr: str, unpadded_kspace: np.ndarray) -> np.ndarray:
    """
    Perform zero-padding on k-space data to have the same number of
    points in the x- and y-directions.

    Parameters
    ----------
    hdr : str
        The XML header string.
    unpadded_kspace : array-like of shape (sl, ro , coils, pe)
        The k-space data to be padded.

    Returns
    -------
    padded_kspace : ndarray of shape (sl, ro_padded, coils, pe_padded)
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
    if padding % 2 != 0:
        padding_left = int(np.floor(padding))
        padding_right = int(np.ceil(padding))
    else:
        padding_left = int(padding)
        padding_right = int(padding)
    padded_kspace = np.pad(unpadded_kspace, ((0, 0), (0, 0), (0, 0), (padding_left, padding_right)))

    return padded_kspace


def zero_pad_kspace_slice_hdr(hdr: str, unpadded_kspace: np.ndarray) -> np.ndarray:
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
    if padding % 2 != 0:
        padding_left = int(np.floor(padding))
        padding_right = int(np.ceil(padding))
    else:
        padding_left = int(padding)
        padding_right = int(padding)
    padded_kspace = np.pad(unpadded_kspace, ((0, 0), (0, 0), (padding_left, padding_right)))

    return padded_kspace


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
            map_of_holes = np.zeros(shape=kspace.shape[:2], dtype=bool)
            for patch_index, ii in enumerate(self.kernel_var_dict["patch_indices"]):
                imwrite(
                    f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/patch_{ii}.tiff",
                    self.kernel_var_dict["patches"][ii, ...],
                )
                map_of_holes = np.zeros(shape=kspace.shape, dtype=bool)
                for hole_idx, (xx, yy) in enumerate(
                    zip(self.kernel_var_dict["holes_x"][ii], self.kernel_var_dict["holes_y"][ii])
                ):
                    # Collect sources for this hole and apply weights

                    map_of_holes[xx - kx2 : xx + kx2 + adjx, yy - ky2 : yy + ky2 + adjy, :] = (
                        hole_idx
                    )
                    S = kspace[xx - kx2 : xx + kx2 + adjx, yy - ky2 : yy + ky2 + adjy, :]
                    # if patch_index < 10:
                    #     print(f"Kernel-patch {ii} from k-space: {S.shape}")
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/kernel_kspace_{ii}_hole{xx}_hole{yy}.tiff", S)
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/abs_kernel_kspace_{ii}_hole{xx}_hole{yy}.tiff", np.abs(S))
                    S = S[self.kernel_var_dict["patches"][ii, ...]]
                    # if patch_index >10:
                    #     print(f"Sources for hole x/y in kspace for patch {ii}: {S.shape}")
                    #     print(f"Weights for hole x/y in kspace for patch {ii}: {weights[ii].shape}")
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/patch_mask_{ii}.tiff", self.kernel_var_dict['patches'][ii, ...])
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/kernel_patch_kspace_{ii}_hole{xx}_hole{yy}.tiff", S)
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/abs_kernel_patch_kspace_{ii}_hole{xx}_hole{yy}.tiff", np.abs(S))
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/weights_{ii}_hole{xx}_hole{yy}.tiff", weights[ii])
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/abs_weights_{ii}_hole{xx}_hole{yy}.tiff", np.abs(weights[ii]))

                    recon[xx, yy, :] = (weights[ii] @ S[:, None]).squeeze()
                    # if patch_index > 10:
                    #     print(f"Reconstructed hole x/y in kspace{ii}: {recon[xx, yy, :].shape}")
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/recon_{ii}_hole{xx}_hole{yy}.tiff", np.moveaxis(recon[xx, yy, :], -1, 0))
                    #     imwrite(f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/abs_recon_{ii}_hole{xx}_hole{yy}.tiff", np.moveaxis(np.abs(recon[xx, yy, :]), -1,0))

                imwrite(
                    f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/map_of_holes_patch_{ii}.tiff",
                    map_of_holes,
                )
            print(f"recon shape before adding to kspace: {recon.shape}")
            print(f"(padded) kspace shape before adding recon: {kspace.shape}")
            return np.moveaxis((recon[:] + kspace)[kx2:-kx2, ky2:-ky2, :], -1, self.coil_axis)


def _kspace_to_log_magnitude(kspace: np.ndarray) -> np.ndarray:
    """Convert k-space tensor to a log-magnitude image for visualization."""

    magnitude = np.log1p(np.abs(kspace))

    lower = np.quantile(magnitude, 0.05)
    upper = np.quantile(magnitude, 0.995)
    if float(upper) > float(lower):
        magnitude = np.clip(magnitude, lower, upper)
        magnitude = (magnitude - lower) / (upper - lower)
    else:
        mag_max = float(magnitude.max())
        if mag_max > 0.0:
            magnitude = magnitude / mag_max

    return np.sqrt(magnitude)


if __name__ == "__main__":
    filename = "/home/melanie.dohmen/mri_recon/data/fastmri/fastMRI_prostate_T2_IDS_001_020/file_prostate_AXT2_001.h5"

    os.makedirs("/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/", exist_ok=True)
    os.makedirs(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/grappa/", exist_ok=True
    )
    with h5py.File(filename, "r") as hf:
        kspace_data = hf["kspace"][:]
        calibration_data = hf["calibration_data"][:]
        hdr = hf["ismrmrd_header"][()]
        im_recon = hf["reconstruction_rss"][:]
        atts = dict()
        atts["max"] = hf.attrs["max"]
        atts["norm"] = hf.attrs["norm"]
        atts["patient_id"] = hf.attrs["patient_id"]
        atts["acquisition"] = hf.attrs["acquisition"]

    # (A, S, C, RO, PE)
    num_avg, num_slices, num_coils, num_ro, num_pe = kspace_data.shape

    # Calib_data shape: num_slices, num_coils, num_pe_cal
    grappa_weight_dict = {}
    grappa_weight_dict_2 = {}

    # (A, S, C, RO, PE) -> take first average and slice to get (C, RO, PE) for GRAPPA weight calculation
    kspace_slice_regridded = kspace_data[0, 0, ...]

    print("kspace_slice_regridded shape: (A, S, C, RO, PE)")
    print(kspace_slice_regridded.shape)

    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_av0_slice0.tiff",
        _kspace_to_log_magnitude(kspace_data[0, 0, :, :, :]),
    )
    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_av1_slice0.tiff",
        _kspace_to_log_magnitude(kspace_data[1, 0, :, :, :]),
    )
    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_av2_slice0.tiff",
        _kspace_to_log_magnitude(kspace_data[2, 0, :, :, :]),
    )

    print("kspace_slice_regridded transposed shape for GRAPPA: (PE, C, RO)")
    print(np.transpose(kspace_slice_regridded, (2, 0, 1)).shape)

    grappa_obj = Grappa(
        np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1
    )

    kspace_slice_regridded_2 = kspace_data[1, 0, ...]
    grappa_obj_2 = Grappa(
        np.transpose(kspace_slice_regridded_2, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1
    )

    # calculate GRAPPA weights
    for slice_num in range(num_slices):
        # (S, C, PE, cal) -> (C, PE, cal)
        calibration_regridded = calibration_data[slice_num, ...]
        # (C, PE, cal) -> (cal, C, PE) for GRAPPA weight calculation#
        if slice_num == 0:
            print(f"calibration_data shape (S, C, PE, cal): {calibration_data.shape}")
            print(f"calibration_regridded shape (C, PE, cal): {calibration_regridded.shape}")
            print(
                f"calibration_regridded transposed shape for GRAPPA: (cal, C, PE)?: {np.transpose(calibration_regridded, (2, 0, 1)).shape}"
            )
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0, 1))
        )
        grappa_weight_dict_2[slice_num] = grappa_obj_2.compute_weights(
            np.transpose(calibration_regridded, (2, 0, 1))
        )

    # apply GRAPPA weights
    kspace_post_grappa_all = np.zeros(shape=kspace_data.shape, dtype=complex)

    for average, grappa_obj, grappa_weight_dict in zip(
        [0, 1, 2],
        [grappa_obj, grappa_obj_2, grappa_obj],
        [grappa_weight_dict, grappa_weight_dict_2, grappa_weight_dict],
    ):
        for slice_num in range(num_slices):
            # (A, S, C, RO, PE) -> (C, RO, PE) for GRAPPA application
            kspace_slice_regridded = kspace_data[average, slice_num, ...]

            # apply weights to transposed k-space slice (PE, C, RO)
            kspace_post_grappa = grappa_obj.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)), grappa_weight_dict[slice_num]
            )
            # and move axes back to (C, RO, PE) after GRAPPA application
            kspace_post_grappa_all[average, slice_num, ...] = np.moveaxis(
                np.moveaxis(kspace_post_grappa, 0, 1), 1, 2
            )
            if average == 0 and slice_num == 0:
                print(
                    f"k-space transposed shape: {np.transpose(kspace_slice_regridded, (2, 0, 1)).shape}"
                )
                print(f"k-space post GRAPPA shape: {kspace_post_grappa.shape}")
                print(
                    f"k-space post GRAPPA moved axes shape: {np.moveaxis(np.moveaxis(kspace_post_grappa, 0, 1), 1, 2).shape}"
                )
                imwrite(
                    "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_pre_grappa_av0_slice0.tiff",
                    _kspace_to_log_magnitude(kspace_slice_regridded),
                )
                imwrite(
                    "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_post_grappa_av0_slice0.tiff",
                    _kspace_to_log_magnitude(kspace_post_grappa_all[0, 0, :, :, :]),
                )

    # recon image for each average
    im = np.zeros((num_avg, num_slices, num_ro, num_ro))
    for average in range(num_avg):
        kspace_grappa = kspace_post_grappa_all[average, ...]
        kspace_grappa_padded = zero_pad_kspace_hdr(hdr, kspace_grappa)
        im[average] = create_coil_combined_im(kspace_grappa_padded)
        imwrite(
            f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/im_average_{average}.tiff",
            im[average],
        )

    im_3d = np.mean(im, axis=0)
    # center crop image to 320 x 320
    img_dict = {}
    img_dict["reconstruction_rss"] = center_crop_im(im_3d, [320, 320])

    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/reconstruction_rss.tiff",
        img_dict["reconstruction_rss"],
    )
    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/given_reconstruction_rss.tiff",
        im_recon,
    )
    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/calibratin_data.tiff",
        calibration_data[15, 10, :, :],
    )
    print("num_avg, num_slices, num_coils, num_ro, num_pe")
    print("kspace_data.shape:", kspace_data.shape)
    print("kspace_grappa.shape:", kspace_grappa.shape)
    print("kspace_grappa_padded.shape:", kspace_grappa_padded.shape)
    print("kspace_post_grappa_all.shape:", kspace_post_grappa_all.shape)
    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_slice_15_coil10.tiff",
        _kspace_to_log_magnitude(kspace_data[0, 15, 10, :, :]),
    )
    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_grappa_slice_15_coil10.tiff",
        _kspace_to_log_magnitude(kspace_grappa[15, 10, :, :]),
    )
    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_grappa_padded_slice_15_coil10.tiff",
        _kspace_to_log_magnitude(kspace_grappa_padded[15, 10, :, :]),
    )
    imwrite(
        "/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/kspace_post_grappa_slice_15_coil10.tiff",
        _kspace_to_log_magnitude(kspace_post_grappa_all[0, 15, 10, :, :]),
    )

    for average in range(num_avg):
        print("average =", average, ": coil_combined_im(kspace_grappe_padded): ", im[average].shape)
        imwrite(
            f"/home/melanie.dohmen/mri_recon/reports/test_prostate_T2_recon/im_slice_15_average_{average}.tiff",
            im[average][15, :, :],
        )

    print("im_3d.shape:", im_3d.shape)
    print("num_slices, num_coils, num_pe_cal")
    print("calibration_data.shape:", calibration_data.shape)
    print("done")
