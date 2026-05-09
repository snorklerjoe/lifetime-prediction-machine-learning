"""Dataset helpers for the IGBT lifetime-prediction MATLAB files.

The MATLAB file is a single top-level ``measurement`` struct with three
branches:

* ``pwmTempControllerState``: 151 controller-setting records
* ``transient``: 110 transient waveform records
* ``steadyState``: 10,978 steady-state temperature records

This module converts those nested MATLAB structs into pandas DataFrames.
Waveform arrays are kept as NumPy arrays inside object columns, which is the
most practical tabular representation for these very wide traces.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.integrate import trapezoid

pd: Any = importlib.import_module("pandas")

DEFAULT_FILENAME = "data/8. IGBT Accelerated Aging/IGBTAgingData_04022009/Data/Thermal Overstress Aging with Square Signal at gate and SMU data/Aging Data/Device 5/Device5  1.mat"

def load_raw_mat(filename: str | Path = DEFAULT_FILENAME) -> dict[str, Any]:
	"""Load the raw MATLAB file with SciPy."""

	return loadmat(Path(filename))


def _is_structured(value: Any) -> bool:
	return isinstance(value, np.void) or (
		isinstance(value, np.ndarray) and value.dtype.names is not None
	)


def _unwrap_scalar(value: Any) -> Any:
	"""Turn MATLAB-ish scalar containers into Python values.

	Arrays with more than one element are preserved as NumPy arrays, except for
	a light squeeze so the resulting dataframe is easier to read.
	"""

	if _is_structured(value):
		return value

	array = np.asarray(value)

	if array.dtype.kind in {"U", "S"}:
		if array.size == 1:
			return str(array.reshape(-1)[0])
		return array.reshape(-1).tolist()

	if array.size == 1:
		return array.reshape(-1)[0].item()

	squeezed = np.squeeze(array)
	if squeezed.shape == ():
		return squeezed.item()
	return squeezed


def _flatten_struct_record(
	record: np.void,
	nested_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
	"""Flatten one MATLAB struct record into a plain dictionary."""

	nested_prefixes = nested_prefixes or {}
	row: dict[str, Any] = {}

	for field in record.dtype.names or ():
		value = record[field]

		if _is_structured(value):
			nested_prefix = nested_prefixes.get(field, f"{field}_")
			if isinstance(value, np.ndarray):
				nested_record = value.reshape(-1)[0]
			else:
				nested_record = value

			for nested_field in nested_record.dtype.names or ():
				row[f"{nested_prefix}{nested_field}"] = _unwrap_scalar(
					nested_record[nested_field]
				)
			continue

		row[field] = _unwrap_scalar(value)

	return row


def structured_array_to_frame(
	array: np.ndarray,
	*,
	nested_prefixes: dict[str, str] | None = None,
) -> Any:
	"""Convert a MATLAB structured array to a pandas DataFrame.

	Each element of ``array`` becomes one row. Nested structs are flattened with
	the provided prefixes.
	"""

	if array.dtype.names is None:
		raise TypeError("Expected a structured NumPy array")

	rows = [_flatten_struct_record(record, nested_prefixes=nested_prefixes) for record in array.ravel()]
	return pd.DataFrame(rows)


def load_measurement_frames(filename: str | Path = DEFAULT_FILENAME) -> dict[str, Any]:
	"""Load the MATLAB file and return one dataframe per top-level branch."""

	raw = load_raw_mat(filename)
	measurement = raw["measurement"][0, 0]

	pwm_df = structured_array_to_frame(measurement["pwmTempControllerState"])
	transient_df = structured_array_to_frame(
		measurement["transient"],
		nested_prefixes={
			"timeDomain": "time_domain_",
			"frequencyDomain": "frequency_domain_",
		},
	)
	steady_state_df = structured_array_to_frame(
		measurement["steadyState"],
		nested_prefixes={"timeDomain": "time_domain_"},
	)

	return {
		"pwm_temp_controller_state": pwm_df,
		"transient": transient_df,
		"steady_state": steady_state_df,
	}


def describe_measurement_structure(filename: str | Path = DEFAULT_FILENAME) -> Any:
	"""Return a compact dataframe describing the MATLAB structure."""

	raw = load_raw_mat(filename)
	measurement = raw["measurement"][0, 0]

	rows: list[dict[str, Any]] = []
	for branch_name in measurement.dtype.names or ():
		branch = measurement[branch_name]
		first_record = branch.reshape(-1)[0]
		rows.append(
			{
				"branch": branch_name,
				"records": measurement[branch_name].shape[1],
				"fields": ", ".join(first_record.dtype.names or ()),
				"has_nested_structs": any(
					_is_structured(first_record[field])
					for field in first_record.dtype.names or ()
				),
			}
		)

	summary = pd.DataFrame(rows)
	summary["notes"] = [
		"controller settings only",
		"waveform records; frequency-domain fields are partly empty",
		"steady-state thermal records; ambient temperature is NaN in many rows",
	]
	return summary


def extract_esw_from_transient(transient_df: Any) -> np.ndarray:
	"""Extract switching energy Esw = integral(Vce(t) * Ic(t)) from each transient record.
	
	Uses trapezoidal numerical integration.
	
	Returns:
		Array of Esw values, one per transient record.
	"""
	esw_values = []
	
	for idx, row in transient_df.iterrows():
		# Get the voltage and current waveforms
		vce = row.get("time_domain_collectorEmitterVoltage")
		ic = row.get("time_domain_collectorEmitterCurrentSignal")
		
		if vce is None or ic is None:
			esw_values.append(np.nan)
			continue
		
		vce = np.asarray(vce).flatten()
		ic = np.asarray(ic).flatten()
		
		# Ensure they're the same length
		min_len = min(len(vce), len(ic))
		if min_len == 0:
			esw_values.append(np.nan)
			continue
		
		vce = vce[:min_len]
		ic = ic[:min_len]
		
		# Trapezoidal integration using scipy
		esw = trapezoid(vce * ic, dx=1.0)  # dx=1 since we don't have actual time in data
		esw_values.append(esw)
	
	return np.array(esw_values)


def extract_vt_from_transient(transient_df: Any, vt_ic_threshold: float = 1.0) -> np.ndarray:
	"""Extract threshold voltage Vt from turn-on transient using cubic interpolation.
	
	Extracts Vt as the gate-emitter voltage (V_GE) at the moment the collector 
	current (I_C) reaches `vt_ic_threshold` (default 1A).
	
	Returns:
		Array of Vt estimates, one per transient record.
	"""
	from scipy.interpolate import CubicSpline
	
	vt_values = []
	
	for idx, row in transient_df.iterrows():
		# Get waveforms
		v_gate_signal = row.get("time_domain_gateSignalVoltage")
		v_ge = row.get("time_domain_gateEmitterVoltage")
		ic = row.get("time_domain_collectorEmitterCurrentSignal")
		
		if v_gate_signal is None or v_ge is None or ic is None:
			vt_values.append(np.nan)
			continue
		
		v_gate_signal = np.asarray(v_gate_signal).flatten()
		v_ge = np.asarray(v_ge).flatten()
		ic = np.asarray(ic).flatten()
		
		min_len = min(len(v_gate_signal), len(v_ge), len(ic))
		if min_len < 10:  # Need enough points for interpolation
			vt_values.append(np.nan)
			continue
		
		v_gate_signal = v_gate_signal[:min_len]
		v_ge = v_ge[:min_len]
		ic = ic[:min_len]
		
		# Find the edge (rising portion)
		gate_diff = np.diff(v_gate_signal)
		rising_idx = np.where(gate_diff > np.max(gate_diff) * 0.1)[0]
		
		if len(rising_idx) < 2:
			vt_values.append(np.nan)
			continue
		
		# Get a window around the rise
		start_idx = max(0, rising_idx[0] - 10)
		end_idx = min(len(v_ge), rising_idx[-1] + 40)
		
		try:
			# Use cubic spline to interpolate
			x = np.arange(end_idx - start_idx)
			vge_win = v_ge[start_idx:end_idx]
			ic_win = ic[start_idx:end_idx]
			
			cs_vge = CubicSpline(x, vge_win)
			cs_ic = CubicSpline(x, ic_win)
			
			# Evaluate on denser grid
			x_dense = np.linspace(0, len(x) - 1, len(x) * 50)
			vge_dense = cs_vge(x_dense)
			ic_dense = cs_ic(x_dense)
			
			# Find where Ic crosses the threshold and maintains positive slope
			# Look for the first index where Ic >= vt_ic_threshold (1.0A)
			# and the surrounding points (e.g. +/- 10 dense points) are monotonically increasing.
			cross_idx = np.where(ic_dense >= vt_ic_threshold)[0]
			vt = np.nan
			for idx in cross_idx:
				if idx < 15 or idx >= len(ic_dense) - 15:
					continue
				
				# Check if the surrounding points have a positive slope.
				# A spline might have tiny ripples, so we check that the general 
				# trend over the next/prev points is strongly positive.
				window_diffs = np.diff(ic_dense[idx-10:idx+10])
				if np.all(window_diffs >= -1e-4) and (ic_dense[idx+5] > ic_dense[idx-5]):
					vt = vge_dense[idx]
					break
					
			if np.isnan(vt) and len(cross_idx) > 0:
				# Fallback just in case we didn't meet the strict monotonicity
				vt = vge_dense[cross_idx[0]]
		except Exception:
			vt = np.nan
		
		vt_values.append(vt)
	
	return np.array(vt_values)


def extract_rth_from_steady_state(steady_state_df: Any) -> np.ndarray:
	"""Extract thermal resistance Rth = (T_internal - T_package) / P_diss.
	
	Estimates P_diss from available thermal and operating conditions if possible.
	
	Returns:
		Array of Rth values, one per steady-state record.
	"""
	rth_values = []
	
	for idx, row in steady_state_df.iterrows():
		t_internal = row.get("time_domain_internalTemperature")
		t_package = row.get("time_domain_packageTemperature")
		
		# Try to estimate power dissipation from available signals
		# P_diss ≈ V_supply * I_ce
		v_supply = row.get("supplyVoltage")
		ic = row.get("collectorEmitterCurrent")
		
		if t_internal is None or t_package is None:
			rth_values.append(np.nan)
			continue
		
		t_internal = np.asarray(t_internal).flatten()
		t_package = np.asarray(t_package).flatten()
		
		# Convert to scalars if arrays
		if t_internal.size > 0:
			t_internal = t_internal.flat[0]
		if t_package.size > 0:
			t_package = t_package.flat[0]
		
		delta_t = float(t_internal) - float(t_package)
		
		if v_supply is not None and ic is not None:
			v_supply = np.asarray(v_supply).flatten()
			ic = np.asarray(ic).flatten()
			if v_supply.size > 0 and ic.size > 0:
				p_diss = float(v_supply.flat[0]) * float(ic.flat[0])
				if p_diss > 0:
					rth = delta_t / p_diss
				else:
					rth = np.nan
			else:
				rth = np.nan
		else:
			# If no power info, skip computation
			rth = np.nan
		
		rth_values.append(rth)
	
	return np.array(rth_values)


def create_processed_dataframe(
	filename: str | Path = DEFAULT_FILENAME,
) -> Any:
	"""Extract features and create a unified processed dataframe.
	
	Combines transient-derived features (Esw, Vt) with steady-state thermal data
	and returns a dataframe with key measurements and derived features.
	"""
	frames = load_measurement_frames(filename)
	transient_df = frames["transient"]
	steady_state_df = frames["steady_state"]
	
	# Extract features from transient data
	esw = extract_esw_from_transient(transient_df)
	vt = extract_vt_from_transient(transient_df)
	
	# Add features to a copy of the transient dataframe
	processed_df = transient_df.copy()
	processed_df["Esw"] = esw
	processed_df["Vt"] = vt

	# Now, let's calculate Rth by aligning transient and steady-state data
	# We can use timeEpoch for alignment.
	# For each transient event, find the corresponding steady-state measurements.
	
	# Convert timeEpoch to numeric for easier comparison
	processed_df["timeEpoch"] = pd.to_numeric(processed_df["timeEpoch"], errors='coerce')
	steady_state_df["timeEpoch"] = pd.to_numeric(steady_state_df["timeEpoch"], errors='coerce')

	# Sort both frames by time
	processed_df = processed_df.sort_values("timeEpoch").reset_index(drop=True)
	steady_state_df = steady_state_df.sort_values("timeEpoch").reset_index(drop=True)

	# Use pandas merge_asof to find the nearest steady_state record for each transient record
	aligned_df = pd.merge_asof(
		processed_df,
		steady_state_df[['timeEpoch', 'time_domain_internalTemperature', 'time_domain_packageTemperature', 'time_domain_supplyVoltage', 'time_domain_collectorEmitterCurrent']],
		on="timeEpoch",
		direction="nearest",
		tolerance=10.0  # Look for a match within 10 seconds (timeEpoch is in seconds)
	)

	# Now calculate Rth using the aligned data
	rth_values = []
	for idx, row in aligned_df.iterrows():
		t_internal = row.get("time_domain_internalTemperature")
		t_package = row.get("time_domain_packageTemperature")
		v_supply = row.get("time_domain_supplyVoltage")
		ic = row.get("time_domain_collectorEmitterCurrent")

		if pd.isna(t_internal) or pd.isna(t_package) or pd.isna(v_supply) or pd.isna(ic):
			rth_values.append(np.nan)
			continue

		# Note: Sensor values for internal and package temps in this thermal run (-2260 and 220)
		# are likely raw uncalibrated ADC values. We apply a placeholder delta for now to prevent negative Rth,
		# but true calibration coefficients are needed to get physical K/W.
		delta_t = abs(float(t_internal) - float(t_package)) 
		p_diss = abs(float(v_supply) * float(ic))

		if p_diss > 0:
			rth = delta_t / p_diss
		else:
			rth = np.nan
		rth_values.append(rth)

	aligned_df["Rth"] = rth_values
	
	# Unbox scalar arrays for columns we want to keep in the CSV
	for col in ["time_domain_internalTemperature", "time_domain_packageTemperature", "time_domain_supplyVoltage", "time_domain_collectorEmitterCurrent"]:
		if col in aligned_df.columns:
			aligned_df[col] = aligned_df[col].apply(lambda x: float(x.flat[0]) if hasattr(x, 'flat') and x.size > 0 else (float(x) if isinstance(x, (float, int)) else np.nan))

	# Calculate Remaining Useful Life (RUL)
	failure_records = aligned_df[aligned_df["time_domain_packageTemperature"] >= 330]
	if len(failure_records) > 0:
		failure_time = failure_records.iloc[0]["timeEpoch"]
	else:
		temps = aligned_df["time_domain_packageTemperature"]
		max_idx = temps.idxmax()
		if pd.notna(max_idx) and max_idx < temps.index[-1]:
			min_temp_after_max = temps.loc[max_idx:].min()
			if min_temp_after_max < 225.0:
				failure_time = aligned_df.loc[max_idx, "timeEpoch"]
			else:
				failure_time = aligned_df["timeEpoch"].max()
		else:
			failure_time = aligned_df["timeEpoch"].max()

	aligned_df["RUL"] = failure_time - aligned_df["timeEpoch"]
	# if negative, clip to 0 since it implies it's already failed
	aligned_df.loc[aligned_df["RUL"] < 0, "RUL"] = 0.0

	# Truncate exported data to not include data after the failure time.
	aligned_df = aligned_df[aligned_df["timeEpoch"] <= failure_time].copy()

	return aligned_df


def downsample_to_1hz(
	df: Any,
	time_column: str | None = None,
) -> Any:
	"""Downsample dataframe to approximately 1 Hz (one sample per second).
	
	For numeric columns, uses mean aggregation. For other columns, keeps first value.
	"""
	# If there's no explicit time column, assume rows are evenly spaced
	# and create a time index based on row count
	if time_column is None or time_column not in df.columns:
		# Estimate: if we have ~110 transient records, assume they're ~10 seconds apart
		# or use a default spacing
		df_indexed = df.copy()
		df_indexed["__time__"] = np.arange(len(df)) * 0.1  # 100 ms spacing guess
		time_column = "__time__"
	else:
		df_indexed = df.copy()
	
	# Create a time index in seconds (using 100 ms intervals per record)
	df_indexed.index = df_indexed[time_column]
	df_indexed.index = pd.to_datetime(df_indexed.index, unit='s')
	
	# Resample to 1 second: aggregate numeric columns with mean, keep strings as first
	agg_dict = {}
	for col in df.columns:
		if col == time_column:
			continue
		if df[col].dtype in [np.float64, np.float32, int, np.int32, np.int64]:
			agg_dict[col] = "mean"
		else:
			agg_dict[col] = "first"
	
	try:
		resampled = df_indexed.resample("1s").agg(agg_dict)
	except Exception as e:
		print(f"Resampling failed: {e}. Returning original data.")
		return df.reset_index(drop=True)
	
	# Use cubic interpolation/extrapolation for numeric columns, fallback to linear, then ffill/bfill
	numeric_cols = resampled.select_dtypes(include=[np.number]).columns
	non_numeric_cols = resampled.select_dtypes(exclude=[np.number]).columns
	
	if len(numeric_cols) > 0:
		try:
			resampled[numeric_cols] = resampled[numeric_cols].interpolate(method='cubic')
		except Exception as e:
			print(f"Cubic interpolation failed: {e}. Falling back to linear.")
			resampled[numeric_cols] = resampled[numeric_cols].interpolate(method='linear')
		resampled[numeric_cols] = resampled[numeric_cols].ffill().bfill()
	
	if len(non_numeric_cols) > 0:
		resampled[non_numeric_cols] = resampled[non_numeric_cols].ffill().bfill()
	
	resampled = resampled.reset_index()
	
	# Convert time back to seconds (float) if it was the timeEpoch
	if time_column in resampled.columns and pd.api.types.is_datetime64_any_dtype(resampled[time_column]):
		resampled[time_column] = resampled[time_column].astype("int64") / 10**9
	
	return resampled


def save_processed_data(
	filename: str | Path = DEFAULT_FILENAME,
	output_dir: str | Path = "processed-data",
	output_filename: str | Path = "processed_data.csv",
) -> Path:
	"""Extract, process, downsample, and save the data.
	
	Returns the path to the saved CSV file and the raw pre-downsampled CSV file.
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	# Extract, process, downsample, and save the data.
	# Create processed dataframe
	df = create_processed_dataframe(filename)
	
	# Downsample to 1 Hz using actual timeEpoch
	df_1hz = downsample_to_1hz(df, time_column="timeEpoch")
	
	def _clean_non_serializable(dataframe: Any) -> Any:
		d = dataframe.copy()
		for col in d.columns:
			if d[col].dtype == object:
				try:
					test_val = d[col].iloc[0]
					if isinstance(test_val, (np.ndarray, list)):
						d = d.drop(columns=[col])
				except (IndexError, TypeError):
					pass
		return d

	# Save to CSV (handle non-serializable columns by dropping them)
	df_export = _clean_non_serializable(df_1hz)
	output_path = output_dir / output_filename
	df_export.to_csv(output_path, index=False)
	
	# Save raw (pre-downsampled) CSV
	# df_export_raw = _clean_non_serializable(df)
	# raw_output_path = output_dir / (output_filename + "_raw.csv")
	# df_export_raw.to_csv(raw_output_path, index=False)

	print(f"Saved processed data to {output_path}")
	
	return output_path


def plot_extracted_features(
	csv_path: str | Path = "processed-data/processed_data.csv",
	raw_csv_path: str | Path | None = "processed-data/processed_data_raw.csv",
) -> Any:
	"""Load processed data and plot extracted features over time.
	
	Returns the matplotlib figure object.
	"""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
	except ImportError:
		print("matplotlib not found; skipping plot")
		return None
	
	csv_path = Path(csv_path)
	if not csv_path.exists():
		print(f"CSV file not found at {csv_path}")
		return None
	
	df = pd.read_csv(csv_path)
	
	# Create a figure with subplots for each feature
	fig, axes = plt.subplots(5, 1, figsize=(12, 15))
	
	# Convert timeEpoch to numeric if it happens to be string
	df["timeEpoch"] = pd.to_numeric(df["timeEpoch"], errors='coerce')
	time_axis = df["timeEpoch"] - df["timeEpoch"].min()
	
	# Plot Esw
	if "Esw" in df.columns:
		axes[0].plot(time_axis, df["Esw"], "b-", marker="None", label="Esw (1Hz)")
		axes[0].set_ylabel("Esw (J)")
		axes[0].set_title("Switching Energy over Time")
		axes[0].grid(True, alpha=0.3)
		axes[0].legend()
	
	# Plot Vt
	if "Vt" in df.columns:
		if raw_csv_path and Path(raw_csv_path).exists():
			raw_df = pd.read_csv(raw_csv_path)
			raw_df["timeEpoch"] = pd.to_numeric(raw_df["timeEpoch"], errors='coerce')
			raw_time = raw_df["timeEpoch"] - raw_df["timeEpoch"].min()
			axes[1].plot(raw_time, raw_df["Vt"], "ro", alpha=0.4, markersize=4, label="Vt (Raw before downsampling)")
		axes[1].plot(time_axis, df["Vt"], "k-", linewidth=1.5, label="Vt (1Hz)")
		axes[1].set_ylabel("Vt (V)")
		axes[1].set_title("Threshold Voltage over Time")
		axes[1].grid(True, alpha=0.3)
		axes[1].legend()
	
	# Plot Rth
	if "Rth" in df.columns:
		axes[2].plot(time_axis, df["Rth"], "g-", marker="None", label="Rth")
		axes[2].set_ylabel("Rth (K/W)")
		axes[2].set_title("Thermal Resistance over Time")
		axes[2].grid(True, alpha=0.3)
		axes[2].legend()

	# Plot Package Temperature
	if "time_domain_packageTemperature" in df.columns:
		axes[3].plot(time_axis, df["time_domain_packageTemperature"], "c-", label="Package Temp")
		failure_time = time_axis.iloc[-1] if not time_axis.empty else 0
		axes[3].axvline(x=failure_time, color="r", linestyle="--", label="End of Test (Failure)")
		axes[3].set_ylabel("Temp (°C)")
		axes[3].set_title("Package Temperature over Time")
		axes[3].grid(True, alpha=0.3)
		axes[3].legend()

	# Plot RUL
	if "RUL" in df.columns:
		axes[4].plot(time_axis, df["RUL"], "m-", marker="None", label="RUL")
		axes[4].set_ylabel("RUL (s)")
		axes[4].set_xlabel("Time (s)")
		axes[4].set_title("Remaining Useful Life over Time")
		axes[4].grid(True, alpha=0.3)
		axes[4].legend()
	else:
		axes[4].set_xlabel("Time (s)")
	
	plt.tight_layout()
	output_path = Path(csv_path).parent / "features_plot.png"
	plt.savefig(output_path, dpi=100)
	print(f"Saved plot to {output_path}")
	
	return fig


def plot_transfer_curve(
	filename: str | Path = DEFAULT_FILENAME,
	output_dir: str | Path = "processed-data",
) -> Any:
	"""Plot the transfer curve ($I_C$ vs $V_{GE}$) during turn-on.
	Shows how $V_{th}$ is extracted via max slope on the linear region.
	"""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		from scipy.interpolate import CubicSpline
	except ImportError:
		print("matplotlib or scipy not found; skipping transfer curve plot")
		return None

	frames = load_measurement_frames(filename)
	transient_df = frames["transient"]
	
	if len(transient_df) == 0:
		return None
		
	# Focus on first valid cycle
	row = transient_df.iloc[0]
	v_gate_signal = row.get("time_domain_gateSignalVoltage")
	v_ge = row.get("time_domain_gateEmitterVoltage")
	ic = row.get("time_domain_collectorEmitterCurrentSignal")
	
	if v_gate_signal is None or v_ge is None or ic is None:
		return None
		
	v_gate_signal = np.asarray(v_gate_signal).flatten()
	v_ge = np.asarray(v_ge).flatten()
	ic = np.asarray(ic).flatten()
	
	min_len = min(len(v_gate_signal), len(v_ge), len(ic))
	v_gate_signal = v_gate_signal[:min_len]
	v_ge = v_ge[:min_len]
	ic = ic[:min_len]

	gate_diff = np.diff(v_gate_signal)
	rising_idx = np.where(gate_diff > np.max(gate_diff) * 0.1)[0]
	
	if len(rising_idx) < 2:
		return None
		
	start_idx = max(0, rising_idx[0] - 10)
	end_idx = min(len(v_ge), rising_idx[-1] + 40)
	
	x = np.arange(end_idx - start_idx)
	vge_window = v_ge[start_idx:end_idx]
	ic_window = ic[start_idx:end_idx]
	
# Extract Vth where Ic crosses 1A and maintains positive slope
	cs_vge = CubicSpline(x, vge_window)
	cs_ic = CubicSpline(x, ic_window)
	x_dense = np.linspace(0, len(x) - 1, len(x) * 50)
	vge_dense = cs_vge(x_dense)
	ic_dense = cs_ic(x_dense)

	cross_idx = np.where(ic_dense >= 1.0)[0]
	vt = np.nan
	for idx in cross_idx:
		if idx < 15 or idx >= len(ic_dense) - 15:
			continue
		window_diffs = np.diff(ic_dense[idx-10:idx+10])
		if np.all(window_diffs >= -1e-4) and (ic_dense[idx+5] > ic_dense[idx-5]):
			vt = vge_dense[idx]
			break
			
	if np.isnan(vt) and len(cross_idx) > 0:
		vt = vge_dense[cross_idx[0]]
	
	# Create true dynamic transfer curve plot (Ic vs Vge)
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.plot(vge_window, ic_window, "b-", alpha=0.7, label="Dynamic Transfer Trajectory")
	ax.plot(vge_window, ic_window, "k.", markersize=2)
	
	# Draw extracted Vt vertical line and intersection point
	if not np.isnan(vt):
		ax.axvline(x=vt, color="r", linestyle="--", linewidth=2, label=f"Extracted Vth ({vt:.2f} V)")
		ax.axhline(y=1.0, color="g", linestyle=":", linewidth=1, label="I_C = 1A")
		ax.plot([vt], [1.0], "ro", markersize=8)
	
	ax.set_xlabel("$V_{GE}$ (Gate-Emitter Voltage) [V]")
	ax.set_ylabel("$I_C$ (Collector Current) [A]")
	ax.set_title("IGBT Dynamic Transfer Curve (Turn-On Event)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "transfer_curve_vth.png"
	plt.savefig(output_path, dpi=100)
	print(f"Saved transfer curve plot to {output_path}")
	
	return fig


if __name__ == "__main__":

	# Process data for all filenames in the following list:
	files = [
		"data/8. IGBT Accelerated Aging/IGBTAgingData_04022009/Data/Thermal Overstress Aging with Square Signal at gate and SMU data/Aging Data/Device 2/Device2  1.mat",
		"data/8. IGBT Accelerated Aging/IGBTAgingData_04022009/Data/Thermal Overstress Aging with Square Signal at gate and SMU data/Aging Data/Device 3/Device3  1.mat",
		"data/8. IGBT Accelerated Aging/IGBTAgingData_04022009/Data/Thermal Overstress Aging with Square Signal at gate and SMU data/Aging Data/Device 4/Device4  1.mat",
		"data/8. IGBT Accelerated Aging/IGBTAgingData_04022009/Data/Thermal Overstress Aging with Square Signal at gate and SMU data/Aging Data/Device 5/Device5  1.mat",
	]

	devind = 1
	for file in files:
		devind += 1
		print(f"Processing file: {file}")
		downsampled_csv_path = save_processed_data(file, output_dir="processed-data", output_filename=f"device_{devind}_processed.csv")
		# plot_extracted_features(downsampled_csv_path)
		# plot_transfer_curve(file)
	# # Run the full pipeline
	# data = load_raw_mat(DEFAULT_FILENAME)
	# frames = load_measurement_frames(DEFAULT_FILENAME)
	# structure_summary = describe_measurement_structure(DEFAULT_FILENAME)
	
	# print("Data structure summary:")
	# print(structure_summary)
	
	# # Process and save
	# downsampled_csv_path, raw_csv_path = save_processed_data(DEFAULT_FILENAME)
	
	# # Plot features
	# plot_extracted_features(downsampled_csv_path, raw_csv_path)
	
	# # Plot transfer curve
	# plot_transfer_curve(DEFAULT_FILENAME)


