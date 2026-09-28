"""Compare the uploaded ProstateX-0001 candidates without extracting archives.

Run from the project root. This is stored-pixel/header QC, not clinical review.
"""

import hashlib
import io
import json
import platform
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import scipy

import prostatex_geometry_check as reader
from prostatex_patch_pilot import physical_grid, sample_grid

BUNDLE = Path("prism-uploads/ProstateX-0001_review_20260915_135244_057184.zip")
EXTRA_FIELDS = {
    (0x0008, 0x0008): "image_type",
    (0x0008, 0x0031): "series_time",
    (0x0008, 0x0032): "acquisition_time",
    (0x0018, 0x0024): "sequence_name",
    (0x0018, 0x0080): "repetition_time",
    (0x0018, 0x0081): "echo_time",
    (0x0018, 0x0083): "averages",
    (0x0018, 0x0091): "echo_train_length",
    (0x0018, 0x1030): "protocol_name",
    (0x0018, 0x1314): "flip_angle",
}
EXPECTED = {
    "6": "1.3.6.1.4.1.14519.5.2.1.7311.5101.143550427570936841761028976610",
    "10": "1.3.6.1.4.1.14519.5.2.1.7311.5101.296510477295449267377313817541",
    "8": "1.3.6.1.4.1.14519.5.2.1.7311.5101.158774312014703328155251833954",
}


class MemoryZip(io.BytesIO):
    def read_bytes(self):
        return self.getvalue()


def compare_uploaded():
    reports, samples, volumes = {}, {}, {}
    original_fields = reader.FIELDS.copy()
    reader.FIELDS.update(EXTRA_FIELDS)
    try:
        with ZipFile(BUNDLE) as outer:
            if outer.testzip() is not None:
                raise ValueError("Outer ZIP CRC failure")
            if len(outer.namelist()) != len(set(outer.namelist())):
                raise ValueError("Repeated outer member names")
            download = json.loads(outer.read("download_report.json"))
            snapshot_bytes = outer.read("inventory.json")
            if hashlib.sha256(snapshot_bytes).hexdigest() != download["inventory_sha256"]:
                raise ValueError("Inventory snapshot hash mismatch")
            inventory = json.loads(snapshot_bytes)
            if len(download["series"]) != 3:
                raise ValueError("Expected three series")
            for record in download["series"]:
                number = str(record["series_number"])
                if number in reports or record["series_uid"] != EXPECTED[number]:
                    raise ValueError("Unexpected or duplicate series")
                source = outer.read(record["file"])
                if len(source) != record["zip_bytes"] or hashlib.sha256(source).hexdigest() != record["zip_sha256"]:
                    raise ValueError("Inner archive provenance mismatch")
                matching = [item for item in inventory["candidates"] if item["series_uid"] == record["series_uid"]]
                if len(matching) != 1:
                    raise ValueError("Ambiguous inventory entry")
                report, series = reader.audit(MemoryZip(source), EXPECTED[number])
                if report["dicom_files"] != record["verified_dicom_files"] or report["dicom_files"] != matching[0]["image_count"]:
                    raise ValueError("DICOM count mismatch")
                for field in ("patient", "study_uid", "description"):
                    if report[field] != record[field] or report[field] != matching[0][field]:
                        raise ValueError(f"Identity mismatch: {field}")
                if report["patient"] != "ProstateX-0001" or int(report["series_number"]) != int(number):
                    raise ValueError("Unexpected patient/series number")
                report["extra_header_values"] = {
                    field: sorted({str(sample.get(field)) for sample in series})
                    for field in EXTRA_FIELDS.values()
                }
                pixel_arrays = []
                for sample in series:
                    if sample["photometric"] != "MONOCHROME2":
                        raise ValueError("Unexpected photometric interpretation")
                    array = np.frombuffer(sample["pixels"], dtype="<u2").reshape(sample["rows"], sample["columns"])
                    if int(array.max()) >= 2 ** sample["bits_stored"]:
                        raise ValueError("Pixels exceed stored bit range")
                    pixel_arrays.append(array)
                reports[number], samples[number] = report, series
                volumes[number] = np.stack(pixel_arrays)
                report["sorted_stored_pixels_sha256"] = hashlib.sha256(volumes[number].tobytes()).hexdigest()
    finally:
        reader.FIELDS.clear()
        reader.FIELDS.update(original_fields)
    for field in ("patient", "study_uid", "frame_uid"):
        if len({report[field] for report in reports.values()}) != 1:
            raise ValueError(f"Cross-series mismatch: {field}")
    geometry_differences = {}
    for field, tolerance in (("position", 1e-4), ("orientation", 1e-6), ("spacing", 1e-6)):
        first = np.asarray([sample[field] for sample in samples["6"]])
        second = np.asarray([sample[field] for sample in samples["10"]])
        geometry_differences[field] = {"max_absolute_component_difference": float(np.abs(first - second).max()), "absolute_tolerance": tolerance}
        np.testing.assert_allclose(first, second, rtol=0, atol=tolerance)
    np.testing.assert_array_equal(reports["6"]["dicom_affine_ijk_to_patient"], reports["10"]["dicom_affine_ijk_to_patient"])
    delta = volumes["6"].astype(np.float64) - volumes["10"].astype(np.float64)
    point = reports["6"]["findings"][0]["pos_mm"]
    for report in reports.values():
        if len(report["findings"]) != 1 or report["findings"][0]["pos_mm"] != point or report["findings"][0]["ClinSig"] != "FALSE":
            raise ValueError("Unexpected finding mapping")
    orientation = samples["6"][0]["orientation"]
    patches, bounds = {}, {}
    for number in ("6", "10", "8"):
        world = physical_grid(point, orientation)
        patches[number], bounds[number] = sample_grid(volumes[number], np.asarray(reports[number]["dicom_affine_ijk_to_patient"]), world)
    result = {
        "bundle_sha256": hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
        "scope": "One patient, native stored pixels and selected top-level headers; no registration or clinical quality ranking",
        "series": reports,
        "t2_comparison": {
            "geometry_matches_within_tolerance": True,
            "geometry_component_differences": geometry_differences,
            "affines_exactly_equal": True,
            "array_equal": bool(np.array_equal(volumes["6"], volumes["10"])),
            "voxel_count": int(delta.size), "different_voxel_count": int(np.count_nonzero(delta)),
            "different_voxel_fraction": float(np.count_nonzero(delta) / delta.size),
            "max_absolute_stored_value_difference": float(np.abs(delta).max()),
            "mean_absolute_stored_value_difference": float(np.abs(delta).mean()),
            "whole_volume_pearson_correlation": float(np.corrcoef(volumes["6"].ravel(), volumes["10"].ravel())[0, 1]),
            "shared_sop_uids": len({sample["sop_uid"] for sample in samples["6"]} & {sample["sop_uid"] for sample in samples["10"]}),
            "interpretation": "Not exact duplicates. Matching geometry/protocol and distinct recorded acquisition times are consistent with repeated acquisition; reason for repetition and preferred clinical series are unknown.",
        },
        "preview_patch_bounds": bounds,
        "selection": {"status": "not_finalized", "reason": "A consistent QC/series-selection policy is needed; acquisition order alone is not evidence of superior image quality."},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path("analysis/prostatex_t2_repeat_check.py"), Path("analysis/prostatex_geometry_check.py"), Path("analysis/prostatex_patch_pilot.py"))},
    }
    return result, volumes, patches


def render(result, volumes, patches):
    figure, axes = plt.subplots(2, 3, figsize=(11, 8))
    t2_high = max(float(np.percentile(volumes[number], 99.5)) for number in ("6", "10"))
    for column, number in enumerate(("6", "10", "8")):
        finding = result["series"][number]["findings"][0]
        high = t2_high if number != "8" else float(np.percentile(volumes[number], 99.5))
        axes[0, column].imshow(volumes[number][finding["nearest_slice_zero_based"]], cmap="gray", vmin=0, vmax=high)
        axes[0, column].scatter(*finding["dicom_continuous_ijk"][:2], marker="+", color="gold")
        label = "ADC" if number == "8" else "T2"
        axes[0, column].set_title(f"{label}, series {number}: native slice 9")
        axes[1, column].imshow(patches[number], cmap="gray", vmin=0, vmax=high, extent=(-25,25,25,-25), interpolation="nearest")
        axes[1, column].scatter(0, 0, marker="+", color="gold")
        axes[1, column].set_xlabel("mm from recorded point")
        axes[1, column].set_title("50 x 50 mm physical patch")
    figure.suptitle("ProstateX-0001: two distinct T2 acquisitions and ADC\n"
                     "Shared T2 display window; recorded finding point, not lesion outline")
    figure.text(.5,.01,"Stored-value QC only; no registration or clinical quality ranking; ADC units unverified.",ha="center",fontsize=9)
    figure.tight_layout(rect=(0,.035,1,.91),h_pad=2)
    figure.savefig("analysis/prostatex_0001_comparison.pdf")
    directory = Path("../../tmp/prism-pdf-previews/prostatex-0001")
    directory.mkdir(parents=True,exist_ok=True)
    figure.savefig(directory / "comparison.png",dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    result, volumes, patches = compare_uploaded()
    Path("analysis/prostatex_0001_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    render(result, volumes, patches)
    print(json.dumps(result["t2_comparison"],indent=2))