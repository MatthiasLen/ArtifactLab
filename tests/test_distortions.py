"""Tests for MRI distortion interfaces."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from mri_recon.distortions import (
    AliasingWrapAroundDistortion,
    AnisotropicResolutionChange,
    Apodization,
    BaseDistortion,
    CoordinateScaling,
    DirectionalSharpnessControl,
    EPINHalfGhostDistortion,
    GibbsRingingDistortion,
    HighFrequencyBoost,
    IsotropicResolutionReduction,
    LineByLineMotionGhostDistortion,
    OffResonanceDistortion,
    PhaseEncodeDecimation,
    RegularizedInverseBlur,
    UnsharpMaskKspace,
    VariableDensityBandwidthReduction,
    ZeroFillDistortion,
)


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class DistortionBaseTests(unittest.TestCase):
    def test_base_distortion_is_abstract(self) -> None:
        self.assertIn("apply", BaseDistortion.__abstractmethods__)


@unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
class SpectralBehaviorValidationTests(unittest.TestCase):
    """Compact numerical regression checks for spectral math."""

    def setUp(self) -> None:
        n = 32
        x = np.linspace(-1.0, 1.0, n)
        yy, xx = np.meshgrid(x, x, indexing="ij")
        self.kspace_flat = np.ones((n, n), dtype=np.complex64)
        self.kspace_gaussian = np.exp(
            -((xx * xx + yy * yy) / (0.35**2))
        ).astype(np.complex64)

    def test_isotropic_reduction_masks_high_frequencies(
        self,
    ) -> None:
        distorted = IsotropicResolutionReduction(
            radius_fraction=0.5,
        ).apply(self.kspace_flat)
        self.assertEqual(float(np.abs(distorted[16, 16])), 1.0)
        self.assertEqual(float(np.abs(distorted[0, 0])), 0.0)

    def test_anisotropic_resolution_change_blurs_selected_axis(self) -> None:
        distorted = AnisotropicResolutionChange(
            kx_fraction=0.9,
            ky_fraction=0.3,
        ).apply(self.kspace_flat)
        self.assertGreater(float(np.abs(distorted[16, 24])), 0.0)
        self.assertEqual(float(np.abs(distorted[24, 16])), 0.0)

    def test_zero_fill_preserves_samples_and_adds_outer_zeros(self) -> None:
        distorted = ZeroFillDistortion(pad_factor=2.0).apply(self.kspace_flat)
        self.assertTrue(np.allclose(distorted[16:48, 16:48], self.kspace_flat))
        self.assertEqual(float(np.abs(distorted[0, 0])), 0.0)

    def test_phase_encode_decimation_keeps_every_rth_line(self) -> None:
        distorted = PhaseEncodeDecimation(factor=4).apply(self.kspace_flat)
        kept_lines = np.any(np.abs(distorted) > 0, axis=-1)
        self.assertEqual(int(np.count_nonzero(kept_lines)), 8)
        self.assertTrue(bool(kept_lines[16]))

    def test_variable_density_reduction_is_radially_smooth(self) -> None:
        distorted = VariableDensityBandwidthReduction(
            kappa=0.5,
        ).apply(self.kspace_flat)
        self.assertGreater(float(np.abs(distorted[16, 16])), 0.99)
        self.assertLess(float(np.abs(distorted[0, 0])), 0.2)

    def test_coordinate_scaling_identity_and_bandwidth_change(self) -> None:
        identity = CoordinateScaling(alpha_x=1.0, alpha_y=1.0).apply(
            self.kspace_gaussian
        )
        shrunk = CoordinateScaling(alpha_x=1.2, alpha_y=1.2).apply(
            self.kspace_gaussian
        )
        self.assertTrue(np.allclose(identity, self.kspace_gaussian))
        self.assertLess(
            float(np.abs(shrunk[16, 24])),
            float(np.abs(self.kspace_gaussian[16, 24])),
        )

    def test_apodization_suppresses_outer_band(self) -> None:
        distorted = Apodization(
            window="gaussian",
            kappa_x=0.6,
            kappa_y=0.6,
        ).apply(self.kspace_flat)
        self.assertGreater(float(np.abs(distorted[16, 16])), 0.99)
        self.assertLess(float(np.abs(distorted[0, 0])), 0.2)

    def test_directional_sharpness_control_is_anisotropic(self) -> None:
        distorted = DirectionalSharpnessControl(
            kappa_x=0.9,
            kappa_y=0.4,
        ).apply(self.kspace_flat)
        self.assertLess(
            float(np.abs(distorted[24, 16])),
            float(np.abs(distorted[16, 24])),
        )

    def test_high_frequency_boost_increases_outer_band(self) -> None:
        distorted = HighFrequencyBoost(beta=0.3, power=2.0).apply(
            self.kspace_flat
        )
        self.assertAlmostEqual(float(np.abs(distorted[16, 16])), 1.0, places=6)
        self.assertGreater(float(np.abs(distorted[0, 0])), 1.0)

    def test_unsharp_mask_kspace_preserves_low_and_boosts_high(self) -> None:
        distorted = UnsharpMaskKspace(beta=0.3, lowpass_kappa=0.5).apply(
            self.kspace_flat
        )
        self.assertAlmostEqual(float(np.abs(distorted[16, 16])), 1.0, places=6)
        self.assertGreater(float(np.abs(distorted[0, 0])), 1.0)

    def test_regularized_inverse_blur_improves_blurred_spectrum(self) -> None:
        blur = np.abs(self.kspace_gaussian)
        blurred = self.kspace_flat * blur
        restored = RegularizedInverseBlur(
            l2_weight=1e-6,
            blur_window=blur,
        ).apply(blurred)
        blurred_mae = float(np.mean(np.abs(blurred - self.kspace_flat)))
        restored_mae = float(np.mean(np.abs(restored - self.kspace_flat)))
        self.assertLess(restored_mae, blurred_mae * 0.5)


@unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
class ArtifactDistortionTests(unittest.TestCase):
    """Regression checks for artifact-specific distortion math."""

    def setUp(self) -> None:
        self.kspace = np.ones((16, 16), dtype=np.complex64)

    def test_gibbs_ringing_uses_rectangular_hard_crop(self) -> None:
        distorted = GibbsRingingDistortion(
            kx_fraction=0.5,
            ky_fraction=0.5,
        ).apply(self.kspace)
        self.assertEqual(float(np.abs(distorted[8, 8])), 1.0)
        self.assertEqual(float(np.abs(distorted[0, 0])), 0.0)

    def test_aliasing_wraparound_keeps_every_rth_ky_line(self) -> None:
        distorted = AliasingWrapAroundDistortion(factor=2).apply(self.kspace)
        kept_lines = np.any(np.abs(distorted) > 0, axis=-1)
        self.assertEqual(int(np.count_nonzero(kept_lines)), 8)

    def test_epi_nhalf_ghost_phase_offset_affects_only_odd_lines(self) -> None:
        phase_offset = np.pi / 6.0
        distorted = EPINHalfGhostDistortion(
            phase_offset_rad=phase_offset,
            delta_x_pixels=0.0,
        ).apply(self.kspace)

        self.assertTrue(np.allclose(distorted[0, :], 1.0 + 0.0j))
        self.assertTrue(
            np.allclose(distorted[1, :], np.exp(1j * phase_offset))
        )

    def test_line_by_line_motion_translation_is_pure_phase(self) -> None:
        distorted = LineByLineMotionGhostDistortion(
            max_shift_x_pixels=2.0,
            max_shift_y_pixels=1.5,
            pattern="ramp",
        ).apply(self.kspace)
        self.assertTrue(np.allclose(np.abs(distorted), np.abs(self.kspace)))

    def test_off_resonance_is_pure_phase_and_varies_along_readout(
        self,
    ) -> None:
        distorted = OffResonanceDistortion(
            omega_max=1.0,
            omega_pattern="linear",
            readout_time_scale=1.0,
        ).apply(self.kspace)

        self.assertTrue(np.allclose(np.abs(distorted), np.abs(self.kspace)))
        self.assertNotAlmostEqual(
            float(np.angle(distorted[0, 0])),
            float(np.angle(distorted[0, -1])),
            places=6,
        )


@unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
class ResolutionDistortionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kspace = np.ones((8, 8), dtype=np.complex64)

    def test_isotropic_and_anisotropic_reduction_mask_expected(
        self,
    ) -> None:
        isotropic = IsotropicResolutionReduction(
            radius_fraction=0.5,
        ).apply(self.kspace)
        anisotropic = AnisotropicResolutionChange(
            kx_fraction=0.8,
            ky_fraction=0.4,
        ).apply(self.kspace)

        self.assertLess(int(np.count_nonzero(isotropic)), self.kspace.size)
        self.assertLess(
            int(np.count_nonzero(anisotropic)),
            int(np.count_nonzero(isotropic)),
        )

    def test_zero_fill_changes_shape_and_keeps_center(self) -> None:
        distorted = ZeroFillDistortion(pad_factor=2.0).apply(self.kspace)
        self.assertEqual(distorted.shape, (16, 16))
        self.assertTrue(np.allclose(distorted[4:12, 4:12], self.kspace))

    def test_phase_encode_decimation_keeps_every_rth_line(self) -> None:
        distorted = PhaseEncodeDecimation(factor=2).apply(self.kspace)
        kept_lines = np.any(np.abs(distorted) > 0, axis=-1)
        self.assertEqual(int(np.count_nonzero(kept_lines)), 4)

    def test_variable_density_taper_and_coordinate_scaling_run(self) -> None:
        tapered = VariableDensityBandwidthReduction(
            kappa=0.5,
        ).apply(self.kspace)
        self.assertLess(
            float(np.abs(tapered[0, 0])),
            float(np.abs(tapered[4, 4])),
        )

        scaled = CoordinateScaling(
            alpha_x=0.95,
            alpha_y=1.05,
        ).apply(self.kspace)
        self.assertEqual(scaled.shape, self.kspace.shape)


@unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
class SharpnessDistortionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kspace = np.ones((16, 16), dtype=np.complex64)

    def test_apodization_and_directional_reduce_outer_band(
        self,
    ) -> None:
        apodized = Apodization(
            window="gaussian",
            kappa_x=0.6,
            kappa_y=0.6,
        ).apply(self.kspace)
        directional = DirectionalSharpnessControl(
            kappa_x=0.9,
            kappa_y=0.4,
        ).apply(self.kspace)

        self.assertLess(float(np.abs(apodized[0, 0])), 1.0)
        self.assertLess(
            float(np.abs(directional[0, 8])),
            float(np.abs(directional[8, 0])),
        )

    def test_high_frequency_boost_and_unsharp_increase_high_band(self) -> None:
        boosted = HighFrequencyBoost(beta=0.4, power=2.0).apply(self.kspace)
        unsharp = UnsharpMaskKspace(
            beta=0.3,
            lowpass_kappa=0.5,
        ).apply(self.kspace)

        self.assertGreater(float(np.abs(boosted[0, 0])), 1.0)
        self.assertGreater(float(np.abs(unsharp[0, 0])), 1.0)

    def test_regularized_inverse_blur_recovers_blurred_signal(self) -> None:
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, 16),
            np.linspace(-1.0, 1.0, 16),
            indexing="ij",
        )
        blur = np.exp(-(xx * xx + yy * yy) / (0.5**2))
        blurred = self.kspace * blur
        restored = RegularizedInverseBlur(
            l2_weight=1e-6,
            blur_window=blur,
        ).apply(blurred)

        blurred_mae = float(np.mean(np.abs(blurred - self.kspace)))
        restored_mae = float(np.mean(np.abs(restored - self.kspace)))
        self.assertLess(restored_mae, blurred_mae * 0.2)


if __name__ == "__main__":
    unittest.main()
