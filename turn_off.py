import numpy as np
import pandas as pd
from pathlib import Path

def calc_turn_off_loss(part, v_op, IL, fsw, params_csv, Tj_C=25.0):
    """
    Calculate turn-off switching loss for GaN transistor.
    
    Args:
        part: transistor part name
        v_op: drain-source switching voltage (VdsSW) in V
        IL: load current in A
        fsw: switching frequency in Hz
        params_csv: path to CSV file with turn-off parameters
        Tj_C: junction temperature in °C
    
    Returns:
        dict with turn-off loss and timing breakdown
    """
    
    # Load params from CSV
    params_df = pd.read_csv(params_csv)
    
    # Find row for this part number (or use default)
    part_row = params_df[params_df['part_number'] == part]
    
    params = part_row.iloc[0]
    
    # Extract parameters from CSV - csv is specific format, can be seen in sample formatt
    Vdron = float(params['Vdron'])
    Vdroff = float(params['Vdroff'])
    Rgoff = float(params['Rgoff'])
    depth = float(params['depth'])
    vgsmiller_method = params['vgsmiller_method']
    vgsmiller_value = params.get('vgsmiller_value', None)
    voltage_threshold = params.get('voltage_threshold', 100.0)  # V, for eq 18 vs 19
    
    # Get transistor data from database
    cap_params, g_fs, V_th, c_gd_op, c_gs_op, rds_T, vgs_used, v_max, i_max, i_cont = \
        data(part, v_op, Tj_C=Tj_C)
    

    # TODO: data function needs to be linked to pullData.py
    
    # Extract capacitances at low and high voltage
    # "H" = high capacitance (low voltage region)
    # "L" = low capacitance (high voltage region)
    Cgdlv = cap_params["c_rss"]["H"]  # C_gd at low voltage
    Cgslv = cap_params["c_iss"]["H"] - cap_params["c_rss"]["H"]  # C_gs at low V
    Cgdhv = cap_params["c_rss"]["L"]  # C_gd at high voltage
    Cgshv = cap_params["c_iss"]["L"] - cap_params["c_rss"]["L"]  # C_gs at high V
    
    Vgsth = V_th
    
    # Calculate Vgsmiller based on method specified in CSV
    if vgsmiller_method == 'threshold_plus_delta':
        # Method 1: Vth + specified delta
        if pd.notna(vgsmiller_value):
            Vgsmiller = Vgsth + float(vgsmiller_value)
        else:
            raise ValueError("vgsmiller_value must be specified for 'threshold_plus_delta' method")
    
    elif vgsmiller_method == 'current_based':
        # Method 2: Calculate from transconductance (Vgs where gfs*(Vgs-Vth) = IL)
        Vgsmiller = Vgsth + IL / g_fs
    
    elif vgsmiller_method == 'fixed_voltage':
        # Method 3: Use fixed voltage value
        if pd.notna(vgsmiller_value):
            Vgsmiller = float(vgsmiller_value)
        else:
            raise ValueError("vgsmiller_value must be specified for 'fixed_voltage' method")
    
    elif vgsmiller_method == 'fraction_of_vdron':
        # Method 4: Fraction of drive voltage
        if pd.notna(vgsmiller_value):
            Vgsmiller = Vdron * float(vgsmiller_value)
        else:
            raise ValueError("vgsmiller_value must be specified for 'fraction_of_vdron' method")
    
    else:
        raise ValueError(f"Unknown vgsmiller_method: {vgsmiller_method}")
    
    VdsSW = v_op
    
    # -------------------
    # Stage t0 to t1: Equation (17)
    # -------------------
    arg = (Vdron - Vgsmiller) / (Vdron - Vdroff)
    if arg > 0:
        t_off1 = -Rgoff * (Cgdlv + Cgslv) * np.log(arg)
    else:
        t_off1 = 0.0
    
    # -------------------
    # Stage t1 to t2: Equation (18) for high voltage, (19) for low voltage
    # -------------------
    denominator = 0.5 * (Vgsmiller + Vgsth) - Vdroff
    
    if abs(denominator) < 1e-6:
        print("Warning: Denominator near zero in t_off2 calculation, using small offset")
        denominator = 1e-3
    
    # Choose equation based on voltage threshold
    if VdsSW >= voltage_threshold:
        # High voltage: use equation (18) - only Cgd matters
        t_off2 = (Cgdlv * VdsSW * Rgoff) / denominator
        eq_used = "18 (high voltage)"
    else:
        # Low voltage: use equation (19) - both Cgd and Cgs matter
        t_off2 = ((Cgdlv + Cgslv) * VdsSW * Rgoff) / denominator
        eq_used = "19 (low voltage)"
    
    # -------------------
    # Stage t2 to t3: Equation (20) 
    # -------------------
    numerator = Vgsth - Vdroff
    denominator_final = depth * Vgsth - Vdroff
    
    if abs(denominator_final) < 1e-6:
        print("Warning: Denominator near zero in t_off3 calculation")
        t_off3 = 0.0
    else:
        arg_final = numerator / denominator_final
        if arg_final > 0:
            t_off3 = -Rgoff * (Cgdhv + Cgshv) * np.log(arg_final)
        else:
            t_off3 = 0.0
    
    # -------------------
    # Turn-off loss calculation: Equation (21) 
    # -------------------
    # Energy per switching event (only t_off2 contributes to loss)
    E_off = (VdsSW * IL / 6.0) * t_off2
    
    # Power loss at given frequency
    P_off = E_off * fsw
    
    # -------------------
    # Return results
    # -------------------
    results = {
        "P_off_W": P_off,
        "E_off_J": E_off,
        "t_off1_s": t_off1,
        "t_off2_s": t_off2,
        "t_off3_s": t_off3,
        "t_total_s": t_off1 + t_off2 + t_off3,
        "Vgsmiller_V": Vgsmiller,
        "Vgsth_V": Vgsth,
        "Vdron_V": Vdron,
        "Vdroff_V": Vdroff,
        "Cgdlv_F": Cgdlv,
        "Cgslv_F": Cgslv,
        "Cgdhv_F": Cgdhv,
        "Cgshv_F": Cgshv,
        "Rgoff_ohm": Rgoff,
        "depth": depth,
        "equation_used": eq_used,
        "vgsmiller_method": vgsmiller_method
    }
    
    return results

# ts from calude
def print_turn_off_results(part_name, v_op, IL, fsw, results):
    """Pretty print turn-off loss analysis results"""
    
    print(f"\n{'='*60}")
    print(f"Turn-Off Loss Analysis for {part_name}")
    print(f"{'='*60}")
    print(f"Operating Voltage (VdsSW): {v_op} V")
    print(f"Load Current (IL): {IL} A")
    print(f"Switching Frequency: {fsw/1e3:.1f} kHz")
    print(f"\nGate Drive Parameters:")
    print(f"  Vdron: {results['Vdron_V']:.2f} V")
    print(f"  Vdroff: {results['Vdroff_V']:.2f} V")
    print(f"  Rg(off): {results['Rgoff_ohm']:.2f} Ω")
    print(f"  Depth: {results['depth']:.2f}")
    
    print(f"\nThreshold Voltages:")
    print(f"  Vgs(th): {results['Vgsth_V']:.3f} V")
    print(f"  Vgs(miller): {results['Vgsmiller_V']:.3f} V  [{results['vgsmiller_method']}]")
    
    print(f"\nCapacitances:")
    print(f"  Cgd (low V): {results['Cgdlv_F']*1e12:.2f} pF")
    print(f"  Cgs (low V): {results['Cgslv_F']*1e12:.2f} pF")
    print(f"  Cgd (high V): {results['Cgdhv_F']*1e12:.2f} pF")
    print(f"  Cgs (high V): {results['Cgshv_F']*1e12:.2f} pF")
    
    print(f"\nTiming Breakdown:")
    print(f"  t_off1 (Vdron → Vgsmiller): {results['t_off1_s']*1e9:.2f} ns")
    print(f"  t_off2 (Vds↑, Id↓): {results['t_off2_s']*1e9:.2f} ns  [Eq. {results['equation_used']}]")
    print(f"  t_off3 (Final discharge): {results['t_off3_s']*1e9:.2f} ns")
    print(f"  Total turn-off time: {results['t_total_s']*1e9:.2f} ns")
    
    print(f"\nLoss Results:")
    print(f"  Energy per turn-off: {results['E_off_J']*1e6:.3f} µJ")
    print(f"  Turn-off power loss: {results['P_off_W']:.3f} W")
    print(f"{'='*60}\n")


# main
if __name__ == "__main__":
    # possibly import from transistor_database_code import data, db, DatabaseManager
    
    # Example calculation
    part_name = "GS66516T"  # Replace with actual part from database
    v_operating = 400  # V
    load_current = 30  # A
    switch_freq = 100e3  # 100 kHz
    params_file = "turn_off_params.csv"
    
    try:
        result = calc_turn_off_loss(
            part=part_name,
            v_op=v_operating,
            IL=load_current,
            fsw=switch_freq,
            params_csv=params_file,
            Tj_C=25.0
        )
        
        print_turn_off_results(part_name, v_operating, load_current, switch_freq, result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()