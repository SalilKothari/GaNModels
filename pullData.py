# %%
import os
from pathlib import Path
from collections import defaultdict
from transistordatabase.database_manager import DatabaseManager

TDB_DIR = Path("tdb_local").resolve()
TDB_DIR.mkdir(parents=True, exist_ok=True)

INDEX_URL = "https://raw.githubusercontent.com/upb-lea/transistordatabase_File_Exchange/main/index.txt"
MODULE_MFR_URL = "https://raw.githubusercontent.com/upb-lea/transistordatabase_File_Exchange/main/module_manufacturers.txt"
HOUSING_TYPES_URL = "https://raw.githubusercontent.com/upb-lea/transistordatabase_File_Exchange/main/housing_types.txt"

db = DatabaseManager()
db.set_operation_mode_json("tdb_local")

type_map = defaultdict(list)

for name in db.get_transistor_names_list():
    t = db.load_transistor(name)
    device_type = getattr(t, "type", "Unknown")
    type_map[device_type].append(name)

# %%
import numpy as np

def calc_v_kn(V, C, vmin_ignore=0.0):
    """
      C(V) ≈ C_H for V < V_kn,  C_L for V >= V_kn
    Returns: (V_kn, C_H, C_L)
    """
    V = np.asarray(V, dtype=float)
    C = np.asarray(C, dtype=float)

    m = np.isfinite(V) & np.isfinite(C) & (V >= vmin_ignore)
    V, C = V[m], C[m]

    idx = np.argsort(V)
    V, C = V[idx], C[idx]

    n = len(V)
    if n < 5:
        raise ValueError("Need at least 5 points for a stable knee estimate.")

    best_sse = np.inf
    best_k = None
    best_CH = None
    best_CL = None

    for k in range(2, n - 2):
        CH = C[:k].mean()
        CL = C[k:].mean()
        sse = np.sum((C[:k] - CH) ** 2) + np.sum((C[k:] - CL) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_k = k
            best_CH, best_CL = CH, CL

    V_kn = float(V[best_k])
    return V_kn, float(best_CH), float(best_CL)

def transferCharacteristics(transistor, Vds_fit=10.0, T_j=25.0):
    """
    Extract g_fs and V_th from switch.channel entries by:
      - fitting Id = g_fs*(Vgs - V_th)
    """
    d = transistor.convert_to_dict()
    ch_list = d["switch"]["channel"]

    Vgs_list = []
    Id_list  = []

    for e in ch_list:
        if float(e.get("t_j", -1)) != float(T_j):
            continue

        vg = e.get("v_g", None)         
        gvi = e.get("graph_v_i", None)  

        if vg is None or gvi is None:
            continue

        Vds_raw = np.array(gvi[0], dtype=float)
        Id_raw  = np.array(gvi[1], dtype=float)

        if Id_raw.ndim != 1:
            continue

        idx = np.argsort(Vds_raw)
        Vds_raw = Vds_raw[idx]
        Id_raw  = Id_raw[idx]

        Id_at = float(np.interp(Vds_fit, Vds_raw, Id_raw))

        if np.isfinite(Id_at) and Id_at > 0:
            Vgs_list.append(float(vg))
            Id_list.append(Id_at)

    Vgs = np.array(Vgs_list, dtype=float)
    Id  = np.array(Id_list, dtype=float)

    if Vgs.size < 3:
        raise ValueError("Not enough (Vgs, Id) points to fit g_fs and V_th.")

    s = np.argsort(Vgs)
    Vgs, Id = Vgs[s], Id[s]

    Id_max = np.max(Id)
    mask = (Id > 0.05 * Id_max) & (Id < 0.9 * Id_max)
    if np.sum(mask) < 3:
        mask = (Id > 0.02 * Id_max)  # fallback

    Vgs_fit = Vgs[mask]
    Id_fit  = Id[mask]

    if Vgs_fit.size < 3:
        raise ValueError("Not enough points in fit region. Try changing Vds_fit or mask thresholds.")

    # linear fit: Id = a*Vgs + b => g_fs=a, V_th=-b/a
    a, b = np.polyfit(Vgs_fit, Id_fit, 1)
    g_fs = float(a)
    V_th = float(-b / a)

    return g_fs, V_th

import numpy as np

def rds_on_at_T(transistor, Tj_C, vgs_target=None, pick="closest", allow_extrap=False):
    """
    Get Rds(on) at a specified junction temperature from transistor.switch.r_channel_th.

    Args:
        transistor: loaded db transistor object
        Tj_C: target temperature in °C
        vgs_target: desired Vgs (V). If None, picks the first available curve.
        pick: "closest" or "exact" (how to select curve when multiple Vgs curves exist)
        allow_extrap: if False, returns None when Tj_C outside curve range; if True, clamps to endpoints.

    Returns:
        (r_ohm, vgs_used, t_min, t_max) or (None, None, None, None) if unavailable.
    """
    rcurves = getattr(getattr(transistor, "switch", None), "r_channel_th", None)
    if not rcurves:
        return None, None, None, None

    # Build list of candidate curves with their Vgs
    candidates = []
    for entry in rcurves:
        vgs = getattr(entry, "v_g", None)

        # graph_t_r = [T_list, R_list]
        if not hasattr(entry, "graph_t_r") or entry.graph_t_r is None:
            continue

        T = np.array(entry.graph_t_r[0], dtype=float)
        R = np.array(entry.graph_t_r[1], dtype=float)

        m = np.isfinite(T) & np.isfinite(R)
        T, R = T[m], R[m]
        if T.size < 2:
            continue

        idx = np.argsort(T)
        T, R = T[idx], R[idx]

        candidates.append((vgs, T, R))

    if not candidates:
        return None, None, None, None

    # Choose which curve to use
    if vgs_target is None:
        vgs_used, T, R = candidates[0]
    else:
        # Filter curves that have numeric vgs if possible
        numeric = [(vgs, T, R) for (vgs, T, R) in candidates if vgs is not None]
        pool = numeric if numeric else candidates

        if pick == "exact":
            exact = [(vgs, T, R) for (vgs, T, R) in pool if float(vgs) == float(vgs_target)]
            if not exact:
                return None, None, None, None
            vgs_used, T, R = exact[0]
        else:  # "closest"
            vgs_used, T, R = min(pool, key=lambda x: abs(float(x[0]) - float(vgs_target)) if x[0] is not None else 1e9)

    t_min, t_max = float(T[0]), float(T[-1])

    # Handle out-of-range temperature
    if (Tj_C < t_min) or (Tj_C > t_max):
        if not allow_extrap:
            return None, (None if vgs_used is None else float(vgs_used)), t_min, t_max
        Tq = float(np.clip(Tj_C, t_min, t_max))
    else:
        Tq = float(Tj_C)

    r_ohm = float(np.interp(Tq, T, R))
    return r_ohm, (None if vgs_used is None else float(vgs_used)), t_min, t_max


def data(part, v_op, Tj_C=25.0, vgs_rds_target=None):
    t = db.load_transistor(part)
    print("\nLoaded:", t.name)

    v_max  = getattr(t, "v_abs_max", None)
    i_max  = getattr(t, "i_abs_max", None)
    i_cont = getattr(t, "i_cont", None)

    capacitances = ["c_oss", "c_iss", "c_rss"]
    cap_params = {}

    for cap in capacitances:
        cap_obj = getattr(t, cap)[0]
        V_cap = cap_obj.graph_v_c[0]
        C_cap = cap_obj.graph_v_c[1]

        V_kn, C_H, C_L = calc_v_kn(V_cap, C_cap, vmin_ignore=10.0)

        print(f"\n{cap.upper()}:")
        print(f"  V_kn  = {V_kn:.2f} V")
        print(f"  {cap}H = {C_H:.3e} F")
        print(f"  {cap}L = {C_L:.3e} F")

        cap_params[cap] = {"V_kn": V_kn, "H": C_H, "L": C_L}

    g_fs, V_th = transferCharacteristics(t)
    print(f"\ng_fs = {g_fs:.3e} S")
    print(f"V_th = {V_th:.2f} V")

    # At V_op
    crss = np.interp(v_op, t.c_rss[0].graph_v_c[0], t.c_rss[0].graph_v_c[1])
    ciss = np.interp(v_op, t.c_iss[0].graph_v_c[0], t.c_iss[0].graph_v_c[1])
    c_gd = crss
    c_gs = ciss - crss

    print(f"\nAt V_op = {v_op} V:")
    print(f"  C_gd = {c_gd:.3e} F")
    print(f"  C_gs = {c_gs:.3e} F")

    # Rds(on) at specified temperature
    rds_T, vgs_used, tmin, tmax = rds_on_at_T(
        t, Tj_C=Tj_C,
        vgs_target=vgs_rds_target,
        pick="closest",
        allow_extrap=False
    )

    print(f"\nR_ds(on) from r_channel_th at Tj = {Tj_C} °C:")
    if rds_T is None:
        if tmin is None:
            print("  No r_channel_th data available.")
        else:
            print(f"  No value (Tj out of curve range {tmin:.1f}–{tmax:.1f} °C or Vgs not found).")
    else:
        vgs_msg = f"{vgs_used:g} V" if vgs_used is not None else "unknown Vgs"
        print(f"  R_ds(on) = {rds_T:.4e} Ω  (using Vgs ≈ {vgs_msg}, curve range {tmin:.1f}–{tmax:.1f} °C)")

    return cap_params, g_fs, V_th, c_gd, c_gs, rds_T, vgs_used, v_max, i_max, i_cont
