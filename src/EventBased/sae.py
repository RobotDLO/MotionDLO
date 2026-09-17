import numpy as np

class sae:

    def compute_recentness(T_on, T_off, t_now_us, tau_us):
        """
        Compute recentness map R from polarity-separated SAE.

        Parameters
        ----------
        T_on : (H,W) int64
            Last ON event timestamp per pixel (microseconds).
        T_off : (H,W) int64
            Last OFF event timestamp per pixel (microseconds).
        t_now_us : int
            Current timestamp in microseconds.
        tau_us : int
            Decay constant in microseconds (e.g. 10000 for 10 ms).

        Returns
        -------
        R : (H,W) float32
            Recentness map in [0,1].
        """

        if tau_us <= 0:
            raise ValueError("tau_us must be > 0")

        t_now_us = np.int64(t_now_us)

        # Time difference (clamped)
        dt_on = np.maximum(t_now_us - T_on, 0)
        dt_off = np.maximum(t_now_us - T_off, 0)

        # Convert to float for exponential
        dt_on_f = dt_on.astype(np.float32)
        dt_off_f = dt_off.astype(np.float32)
        tau_f = np.float32(tau_us)

        # Exponential decay
        R_on = np.exp(-dt_on_f / tau_f)
        R_off = np.exp(-dt_off_f / tau_f)

        # Combine polarities
        R = np.maximum(R_on, R_off)
        
        return R.astype(np.float32)