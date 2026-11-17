import numpy as np
import pandas as pd

from contagion.data.calendar import align_to_master, lead_lag_profile, sync_covariance
from contagion.data.events import bear_episodes


def test_alignment_conserves_cumulative_return():
    master = pd.bdate_range("2024-01-01", "2024-03-29")
    # a market that is closed on two master days and prints once on a non-master day (Saturday)
    own = master.drop([master[10], master[11]]).append(pd.DatetimeIndex(["2024-02-10"])).sort_values()
    rng = np.random.default_rng(0)
    lvl = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(own)))), index=own)
    a, st = align_to_master(lvl, master, max_stale=3)
    r = np.log(a).diff()
    assert np.isclose(r.sum(), np.log(lvl.iloc[-1] / lvl.iloc[0]))
    assert st.iloc[10] == 1 and st.iloc[11] == 2 and st.iloc[12] == 0
    assert r.iloc[10] == 0.0 and r.iloc[11] == 0.0


def test_max_stale_masks_long_gaps():
    master = pd.bdate_range("2024-01-01", periods=30)
    own = master[:10].append(master[20:])
    a, st = align_to_master(pd.Series(1.0, index=own), master, max_stale=3)
    assert a.iloc[10:13].notna().all() and a.iloc[13:20].isna().all()


def test_asynchrony_detected_and_corrected():
    """A market that closes before the reference absorbs part of each day's news one day later."""
    rng = np.random.default_rng(1)
    T = 20000
    early = rng.normal(size=T)  # news before the early close
    late = rng.normal(size=T)  # news between the early close and the reference close
    ref = early + late
    x = early + np.concatenate([[0.0], late[:-1]])  # late news shows up the next day
    df = pd.DataFrame({"ref": ref, "x": x})
    ll = lead_lag_profile(df, "ref")
    assert ll.loc["x", "lag+1"] > 0.3 and ll.loc["x", "timing_signal"] > 0.3
    S = sync_covariance(df.to_numpy(), lags=1, method="dimson")
    c_sync = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
    c_naive = np.corrcoef(ref, x)[0, 1]
    assert c_naive < 0.55 and c_sync > 0.9  # true synchronous correlation is 1


def test_bear_episodes_state_machine():
    path = np.concatenate([np.linspace(0, 0.3, 50), np.linspace(0.3, 0.05, 30), np.linspace(0.05, 0.5, 60)])
    r = pd.Series(np.diff(np.concatenate([[0], path])), index=pd.bdate_range("2020-01-01", periods=len(path)))
    ep = bear_episodes(r, down=0.15, up=0.15)
    assert len(ep) == 1
    assert ep.loc[0, "peak"] == r.index[49] and ep.loc[0, "trough"] == r.index[79]
