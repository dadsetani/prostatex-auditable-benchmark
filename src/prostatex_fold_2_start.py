from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import base64
import glob
import hashlib
import runpy


script_matches = glob.glob(
    "/kaggle/input/**/fold_retrieval_v3.py.base64.txt",
    recursive=True,
)
input_matches = glob.glob(
    "/kaggle/input/**/prostatex_fold_retrieval_kaggle_inputs.zip.base64.txt",
    recursive=True,
)
assert len(script_matches) == 1, script_matches
assert len(input_matches) == 1, input_matches

script_bytes = base64.b64decode(Path(script_matches[0]).read_text())
assert hashlib.sha256(script_bytes).hexdigest() == (
    "5b202b3541189c9a10de1b710b04648b309f7d6be99b442536d822cb70171f65"
)
script_path = Path("/kaggle/working/prostatex_fold_retrieval_kaggle_v3.py")
script_path.write_bytes(script_bytes)

input_zip_bytes = base64.b64decode(Path(input_matches[0]).read_text())
input_dir = Path("/kaggle/working/prostatex_fold_retrieval_inputs")
input_dir.mkdir(parents=True, exist_ok=True)
with ZipFile(BytesIO(input_zip_bytes)) as archive:
    archive.extractall(input_dir)

plan = input_dir / "prostatex_download_plan_v0_3.csv"
labels = input_dir / "ProstateX-TrainingLesionInformationv2.zip"
protocol = input_dir / "prostatex_selection_protocol_v0_3.json"
assert plan.is_file() and labels.is_file() and protocol.is_file()

runner = runpy.run_path(str(script_path))
print("Initialization complete")

result = runner["retrieve_fold"](2, plan, labels, protocol)
print("Fold 2 bundle:", result)