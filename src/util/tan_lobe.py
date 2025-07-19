import torch
import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect


# def tan_ggx_lobe(alpha: torch.Tensor, k: float = 0.5) -> torch.Tensor:
#     """
#     Tangent of GGX lobe with roughness alpha and threshold k
#     """
#     sqrt_k = np.sqrt(k)
#     sq_a = alpha ** 2

#     assert (alpha >= 0).all(), "Roughness must be positive"
#     assert (sq_a < sqrt_k).all(), "Roughness must be less than threshold"

#     # Numerator
#     numer = torch.sqrt((1 - sqrt_k) * sq_a * (sqrt_k - sq_a))
#     # Denominator
#     denom = sqrt_k + (sqrt_k - 2) * sq_a

#     return numer / denom


"""
Calculate the tangent value of a level-set of an NDF function,
such that the integral within the level-set is k.
"""

def GGX(theta, alpha):
    """
    GGX NDF function.
    """
    cos_theta = np.cos(theta)
    alpha_sq = alpha ** 2
    denom = (cos_theta ** 2 * (alpha_sq - 1) + 1) ** 2
    return alpha_sq / (np.pi * denom)

def energy_ratio(theta_k, alpha, integrand=GGX):
    """
    Calculate the energy ratio of the GGX NDF function within the level-set defined by theta_k.
    """
    num, _ = quad(integrand, 0, theta_k, args=(alpha,))
    denom, _ = quad(integrand, 0, np.pi / 2, args=(alpha,))
    return num / denom

def find_theta_k(alpha, target_ratio=0.95, integrand=GGX):
    """
    Find the theta_k value for a given alpha and target energy ratio.
    """
    return bisect(lambda t: energy_ratio(t, alpha, integrand=integrand) - target_ratio, 1e-12, np.pi / 2 - 1e-12)


class LobeLUT:
    """
    1D Lookup Table for Tangent Values of BSDF Lobes.
    """
    def __init__(self,
        alpha: list[float],
        cone_threshold: float = 0.95,
        integrand_type="GGX",
        device: str = "cuda"
    ):
        assert len(alpha) > 0, "Alpha list must not be empty."
        self.alpha = np.array(alpha, dtype=np.float32)

        integrand_map = {
            "GGX": GGX,
        }
        if integrand_type not in integrand_map:
            raise ValueError(f"Unsupported integrand type: {integrand_type}. Supported types: {list(integrand_map.keys())}")
        
        # Build the lookup table
        self.lut = np.zeros_like(self.alpha, dtype=np.float32)
        last_alpha = alpha[0]
        for i, a in enumerate(alpha):
            if a < last_alpha:
                raise ValueError("Alpha values must be in non-decreasing order.")
            self.lut[i] = find_theta_k(a, cone_threshold, integrand=integrand_map[integrand_type])
            last_alpha = a

        self.alpha = torch.tensor(self.alpha, dtype=torch.float32, device=device)
        self.lut = torch.tensor(self.lut, dtype=torch.float32, device=device)
        
    def __call__(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Interpolate the tangent values for the given alpha values.
        """
        if not isinstance(alpha, torch.Tensor):
            raise TypeError("Alpha must be a torch.Tensor.")
        
        assert alpha.dim() == 2 and alpha.shape[1] == 1, "Alpha tensor must be of shape (N, 1)."

        alpha = torch.clamp(alpha, min=self.alpha[0], max=self.alpha[-1])

        # Lookup the indices for interpolation
        idx = torch.searchsorted(self.lut, alpha, right=True) - 1
        idx = torch.clamp(idx, min=0, max=self.lut.numel() - 2)

        # Linear interpolation
        alpha0 = self.alpha[idx]
        alpha1 = self.alpha[idx + 1]
        lut0 = self.lut[idx]
        lut1 = self.lut[idx + 1]

        t = (alpha - alpha0) / (alpha1 - alpha0)
        tan_value = lut0 + t * (lut1 - lut0)

        return tan_value


if __name__ == "__main__":
    # # Example usage
    # alpha = 0.005  # Example roughness parameter
    # target_ratio = 0.95  # Target energy ratio

    # theta_k = find_theta_k(alpha, target_ratio)
    # print(f"Theta_k for alpha={alpha} and target ratio={target_ratio}: {theta_k:.6f}")

    # # Calculate the tangent value
    # tan_value = np.tan(theta_k)
    # print(f"Tangent value: {tan_value:.6f}")

    # Example usage of LobeLUT
    alpha_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    lut = LobeLUT(alpha_values, cone_threshold=0.99, integrand_type="GGX")

    print("LUT values:", lut.lut)

    test_alpha = torch.tensor([[0.001], [0.005], [0.01], [0.02], [0.1]], dtype=torch.float32, device="cuda")
    tan_values = lut(test_alpha)

    print("Tangent values:", tan_values)