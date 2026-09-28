"""Local geometric QC of two uploaded series; no registration or training.

Run from the project root: python analysis/prostatex_pair_check.py
Previews show stored pixel values with independent display windows; ADC units
have not been established. The imported reader is intentionally format-limited.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prostatex_geometry_check import audit

SERIES = {
    "T2_axial": (Path("prism-uploads/T2_axial.zip"),
                 "1.3.6.1.4.1.14519.5.2.1.7311.5101.479812804428819225709948044920"),
    "ADC": (Path("prism-uploads/ADC.zip"),
            "1.3.6.1.4.1.14519.5.2.1.7311.5101.227755318954832018272622878067"),
}


def compare():
    reports = {}
    volumes = {}
    for name, (path, uid) in SERIES.items():
        reports[name], volumes[name] = audit(path, uid)
    for field in ("patient", "study_uid", "frame_uid"):
        if reports["T2_axial"][field] != reports["ADC"][field]:
            raise ValueError(f"Cross-series {field} mismatch")
    if reports["T2_axial"]["patient"] != "ProstateX-0002":
        raise ValueError("Unexpected patient")
    normal = np.cross(volumes["T2_axial"][0]["orientation"][:3],
                      volumes["T2_axial"][0]["orientation"][3:])
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(volumes["T2_axial"][0]["orientation"],
                               volumes["ADC"][0]["orientation"], atol=1e-6)
    if len(volumes["T2_axial"]) != len(volumes["ADC"]):
        raise ValueError("Different plane counts: this sample comparison requires review")
    t2_origins = np.asarray([sample["position"] for sample in volumes["T2_axial"]])
    adc_origins = np.asarray([sample["position"] for sample in volumes["ADC"]])
    plane_differences = (adc_origins - t2_origins) @ normal
    t2_affine = np.asarray(reports["T2_axial"]["dicom_affine_ijk_to_patient"])
    adc_affine = np.asarray(reports["ADC"]["dicom_affine_ijk_to_patient"])
    t2_to_adc = np.linalg.solve(adc_affine, t2_affine)
    key = lambda finding: (finding["fid"], tuple(finding["pos_mm"]))
    findings = {name: {key(finding): finding for finding in report["findings"]}
                for name, report in reports.items()}
    if set(findings["T2_axial"]) != set(findings["ADC"]):
        raise ValueError("Finding position mismatch between series")
    errors = []
    for finding_key, t2_finding in findings["T2_axial"].items():
        adc_finding = findings["ADC"][finding_key]
        if t2_finding["ClinSig"] != adc_finding["ClinSig"]:
            raise ValueError("Cross-series label mismatch")
        transformed = (t2_to_adc @ np.r_[t2_finding["dicom_continuous_ijk"], 1.0])[:3]
        errors.append(float(np.abs(transformed - adc_finding["dicom_continuous_ijk"]).max()))
    comparison = {
        "same_patient_study_frame": True,
        "same_orientation_within_tolerance": True,
        "max_corresponding_plane_separation_mm": float(np.abs(plane_differences).max()),
        "t2_ijk_to_adc_ijk_from_headers": t2_to_adc.tolist(),
        "max_finding_transform_error_in_voxels": max(errors),
        "meaning": "Header-defined geometry only; no anatomical registration accuracy established",
    }
    return {"series": reports, "comparison": comparison}, volumes


def render(result, volumes):
    labels = [finding["fid"] for finding in result["series"]["T2_axial"]["findings"]]
    if len(set(labels)) != len(labels):
        raise ValueError("This preview requires unique finding IDs within the patient")
    labels.sort(key=int)
    fig, axes = plt.subplots(len(labels), 2, figsize=(8.5, 4.1 * len(labels)), squeeze=False)
    for row_index, finding_id in enumerate(labels):
        for column_index, name in enumerate(SERIES):
            report = result["series"][name]
            finding = next(item for item in report["findings"] if item["fid"] == finding_id)
            sample = volumes[name][finding["nearest_slice_zero_based"]]
            pixels = np.frombuffer(sample["pixels"], dtype="<u2").reshape(sample["rows"], sample["columns"])
            if int(pixels.max()) >= 2**sample["bits_stored"]:
                raise ValueError("Pixel data exceeds the declared stored bit range")
            image_column, image_row = finding["dicom_continuous_ijk"][:2]
            row_spacing, column_spacing = sample["spacing"]
            bounds = ((-image_column - .5) * column_spacing,
                      (sample["columns"] - image_column - .5) * column_spacing,
                      (sample["rows"] - image_row - .5) * row_spacing,
                      (-image_row - .5) * row_spacing)
            if not (bounds[0] <= -25 and bounds[1] >= 25 and bounds[2] >= 25 and bounds[3] <= -25):
                raise ValueError("Requested physical view extends outside the image")
            high = float(np.percentile(pixels, 99.5))
            if high <= 0:
                raise ValueError("Insufficient intensity range for preview")
            cmap = "gray_r" if sample["photometric"] == "MONOCHROME1" else "gray"
            axis = axes[row_index, column_index]
            axis.imshow(pixels, cmap=cmap, origin="upper", extent=bounds,
                        interpolation="nearest", vmin=0, vmax=high)
            axis.scatter(0, 0, marker="+", s=150, color="#ffdd33")
            axis.set_xlim(-25, 25)
            axis.set_ylim(25, -25)
            axis.set_aspect("equal")
            axis.set_title(f"{name} | Finding {finding_id} | ClinSig={finding['ClinSig']}\n"
                           f"{column_spacing:g} mm/pixel; slice index {finding['nearest_slice_zero_based']}", fontsize=10)
            axis.set_xlabel("Image-column direction from recorded point (mm)")
            axis.set_ylabel("Image-row direction from recorded point (mm)")
    fig.suptitle("PROSTATEx-0002: same 50 x 50 mm viewing window\n"
                 "Recorded points, not lesion outlines; native pixels, no registration", fontsize=12)
    fig.text(.5, .012, "Independent display windows on stored values. ADC physical units not verified. QC only, not model results.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .93))
    fig.savefig("analysis/prostatex_pair_preview.pdf")
    preview = Path("../../tmp/prism-pdf-previews/prostatex-pair")
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / "comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    result, volumes = compare()
    render(result, volumes)
    Path("analysis/prostatex_pair_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparison"], indent=2))
    for name, report in result["series"].items():
        print(name, report["rows"], report["columns"], report["pixel_spacing_mm"],
              "rescale tags", report["rescale_tags_by_slice"][0])
        for finding in report["findings"]:
            print(finding["fid"], finding["ClinSig"], finding["dicom_continuous_ijk"], finding["nearest_filename"])