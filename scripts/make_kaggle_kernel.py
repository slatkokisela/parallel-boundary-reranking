"""Bundle this repo into a self-contained Kaggle GPU script kernel for CE-score precompute.

    python scripts/make_kaggle_kernel.py OUT_DIR --kernel-id user/slug --split dev \
        --gpu0 cross-encoder/ms-marco-MiniLM-L-12-v2 BAAI/bge-reranker-base \
        --gpu1 BAAI/bge-reranker-v2-m3 [--fp16] [--pool-size 1010]
    kaggle kernels push -p OUT_DIR
    kaggle kernels output user/slug -p DIR      # -> cache.zip (unzip into the repo root)

Embeddings are computed once. The models listed for each GPU then run sequentially on that
GPU, and the two GPUs run in parallel. The pool size defaults to 1010 (> local N=1000), so
near-tie rank flips at the cut caused by GPU/CPU float differences are still covered by the
cache.
"""
import argparse
import base64
import io
import json
import pathlib
import zipfile

RUNNER = r'''
import base64, io, json, os, shutil, subprocess, sys, time, zipfile
P = "/kaggle/temp_pbar"; W = "/kaggle/working"
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(P)
os.chdir(P)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers==6.3.0"])
import torch, sentence_transformers, transformers
info = dict(torch=torch.__version__, st=sentence_transformers.__version__, transformers=transformers.__version__,
            gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], spec=SPEC)
print(info, flush=True)
env = dict(os.environ, PYTHONPATH="src")
t0 = time.time()
# 1) embeddings + query embeddings once (CPU-free GPU step), via a no-CE prepare
subprocess.run([sys.executable, "-c", (
    "import sys; sys.path.insert(0,'src');"
    "from pbar.experiment import load_config; from pbar import data;"
    "from pbar.retrieval import DenseRetriever; from pbar.experiment import _query_embeddings, _slug;"
    "from pathlib import Path; cfg=load_config('configs/baselines.yaml'); rc=cfg['retriever'];"
    "ds=data.load('nfcorpus','data/raw'); r=DenseRetriever(rc['model'],rc['query_prefix'],rc['doc_prefix'],128);"
    "r.encode_corpus(ds.doc_texts, Path('cache')/f\"emb_nfcorpus_{_slug(rc['model'])}.npy\");"
    f"[_query_embeddings(lambda: r, ds, ds.split_qids(s), Path('cache')/f\"qemb_nfcorpus_{{_slug(rc['model'])}}.npz\") for s in {SPEC['splits']!r}]"
)], check=True, env=env)
# 2) one process per GPU, its models sequentially
def gpu_cmd(models):
    cmds = []
    for m in models:
        c = ["scripts/precompute.py", "configs/baselines.yaml", "--splits", *SPEC["splits"],
             "--pool-size", str(SPEC["pool_size"]), "--ce-model", m, "--ce-batch-size", str(SPEC["batch_size"])]
        if SPEC["fp16"]:
            c.append("--fp16")
        cmds.append(" ".join([sys.executable] + [repr(x) if " " in x else x for x in c]))
    return " && ".join(cmds)
procs = []
for gpu, models in enumerate([SPEC["gpu0"], SPEC["gpu1"]]):
    if models:
        procs.append(subprocess.Popen(gpu_cmd(models), shell=True, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu))))
codes = [p.wait() for p in procs]
info.update(codes=codes, seconds=round(time.time() - t0, 1))
json.dump(info, open("cache/kaggle_env_ce_pilot.json", "w"), indent=2)
for f in os.listdir("cache"):          # ship only CE caches + env (embeddings already local)
    if not (f.startswith("ce_") or f.startswith("kaggle_env")):
        os.remove(os.path.join("cache", f))
shutil.make_archive(W + "/cache", "zip", P, "cache")
print("DONE", info, os.listdir("cache"), flush=True)
if any(codes):
    sys.exit(1)
'''

COMMAND_RUNNER = r'''
import base64, io, json, os, shutil, subprocess, sys, time, zipfile
P = "/kaggle/temp_pbar"; W = "/kaggle/working"
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(P)
os.chdir(P)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers==6.3.0"])
import torch
print({"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, flush=True)
env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="0")
code = subprocess.run(COMMAND, shell=True, env=env).returncode
os.makedirs("results", exist_ok=True)
shutil.make_archive(W + "/results", "zip", P, "results")
print("DONE exit", code, flush=True)
sys.exit(code)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--kernel-id", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev"])
    ap.add_argument("--gpu0", nargs="*", default=[])
    ap.add_argument("--gpu1", nargs="*", default=[])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--pool-size", type=int, default=1010)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--command", help="instead of precompute: run this shell command in the repo "
                    "on GPU 0 and ship results/ as results.zip")
    ap.add_argument("--extra-root", help="directory whose files are bundled at their relative paths "
                    "(e.g. a folder containing cache/...)")
    a = ap.parse_args()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in ["src/pbar/**/*.py", "configs/*.yaml", "scripts/*.py", "pyproject.toml"]:
            for p in pathlib.Path(".").glob(pat):
                if "__pycache__" not in p.parts:
                    z.write(p, p.as_posix())
        if a.extra_root:
            root = pathlib.Path(a.extra_root)
            for p in root.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(root).as_posix())
    spec = dict(splits=a.splits, gpu0=a.gpu0, gpu1=a.gpu1, fp16=a.fp16, pool_size=a.pool_size,
                batch_size=a.batch_size)
    b64 = base64.b64encode(buf.getvalue()).decode()
    if a.command:
        code = f"BUNDLE = {b64!r}\nCOMMAND = {a.command!r}\n" + COMMAND_RUNNER
    else:
        code = f"BUNDLE = {b64!r}\nSPEC = {spec!r}\n" + RUNNER
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = a.kernel_id.split("/")[1]
    (out / f"{slug}.py").write_text(code)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": a.kernel_id, "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
        "kernel_sources": []}, indent=2))
    print("wrote", out, "spec", spec)


if __name__ == "__main__":
    main()
