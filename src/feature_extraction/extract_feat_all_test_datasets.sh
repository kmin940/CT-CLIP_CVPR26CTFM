#!/bin/bash
# ============================================================================
# Extract CT-CLIP features for every test dataset, in parallel across 2 GPUs.
#
# The six per-dataset scripts are not equally sized -- AMOS_clf_tr_val alone is
# ~12.8k forward passes (11 ROI diseases over their own mask subsets, plus one
# whole-image pass shared by the 4 non-ROI diseases) against ~6.8k for the other
# six combined. Handing each script to a GPU whole would therefore leave one
# GPU idle for most of the run, so the AMOS-shaped scripts are split per disease
# (via their DISEASES override) and the resulting ~29 units go into one queue
# that both GPUs pull from. Whichever GPU frees up first takes the next unit, so
# neither sits idle while work remains.
#
# The 4 non-ROI AMOS diseases are queued as ONE unit: they share a single
# whole-image extraction, and the scripts' own non_roi_extracted guard collapses
# them into it. Splitting them would make 4 units race for the same output dir.
#
# Extraction is CPU/IO-bound, not GPU-bound: nibabel's gzip read and the trilinear
# resample dominate, and a single unit leaves its GPU at ~2-7% utilisation while
# using ~1.5 cores and ~2 GB of VRAM. JOBS_PER_GPU therefore runs several units
# per GPU (4 by default -> 8 in flight, ~12 GB VRAM and ~12 of 48 cores), which is
# where the wall-clock win actually comes from. Lower it if the volumes are being
# read off a slow disk.
#
# Usage:
#   ./extract_feat_all_test_datasets.sh
#   GPUS="0 1" RESULTS_ROOT=/path/to/results_relu ./extract_feat_all_test_datasets.sh
#   JOBS_PER_GPU=1 ./extract_feat_all_test_datasets.sh   # one unit per GPU
#   DRYRUN=1 ./extract_feat_all_test_datasets.sh        # print the queue, run nothing
#
# Env:
#   GPUS="0 1"       GPU indices to spread the queue over
#   JOBS_PER_GPU=4   concurrent units per GPU
#   CTRATE_SHARDS=4  split CT-RATE into this many units (its volumes average
#                    ~195 MB gzipped vs ~50 MB elsewhere, so one worker runs it
#                    at ~6.5 s/volume against ~1.5 s for the other datasets)
#   RESULTS_ROOT     output tree, passed through to every unit
#                    (default: the scripts' own ${REPO_ROOT}/results)
#   LOG_DIR          per-unit logs (default: ${REPO_ROOT}/logs/feature_extraction/<ts>)
#   DRYRUN=1         list the units and exit
#
# Note: extract_feat_LP.py asserts each output .h5 does not already exist, so a
# unit whose embeddings are already on disk FAILS rather than resuming. Units are
# independent here -- one failure is logged and the rest continue; the exit code
# and the closing summary report what failed.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

read -r -a GPU_LIST <<< "${GPUS:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
CTRATE_SHARDS="${CTRATE_SHARDS:-4}"
DRYRUN="${DRYRUN:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/feature_extraction/$(date +%Y%m%d_%H%M%S)}"

# AMOS ROI diseases get one unit each; the non-ROI four share one.
amos_roi=(splenomegaly adrenal_hyperplasia fatty_liver cholecystitis
          liver_calcifications hydronephrosis gallstone liver_lesion
          kidney_stone liver_cyst renal_cyst)
amos_nonroi="atherosclerosis colorectal_cancer ascites lymphadenopathy"

# Each unit is "<label>|<script>|<DISEASES or ->|<extra env or ->". Ordered
# longest-first so the
# big AMOS units start early and the small whole-script ones backfill the tail.
units=()
for disease in "${amos_roi[@]}"; do
    units+=("AMOS_clf_tr_val/${disease}|extract_feat_alldiseases.sh|${disease}|-")
done
units+=("AMOS_clf_tr_val/_nonROI|extract_feat_alldiseases.sh|${amos_nonroi}|-")
for disease in "${amos_roi[@]}"; do
    units+=("FLARE/${disease}|extract_feat_alldiseases_test.sh|${disease}|-")
done
units+=("FLARE/_nonROI|extract_feat_alldiseases_test.sh|${amos_nonroi}|-")
units+=("MSWAL_train|extract_feat_alldiseases_MSWAL_train.sh|-|-")
units+=("MSWAL_test|extract_feat_alldiseases_MSWAL.sh|-|-")
units+=("Gaozb_lung_part1|extract_feat_alldiseases_Gaozb_lung.sh|-|-")
units+=("autoPET|extract_feat_alldiseases_autoPET.sh|-|-")
# CT-RATE: 1616 whole-image volumes, one shared embeddings set for all 18
# targets, split into CTRATE_SHARDS units. The shards stride the file list, so
# together they write exactly the .h5 files a single worker would.
for (( sh = 0; sh < CTRATE_SHARDS; sh++ )); do
    units+=("CT-RATE/shard${sh}|extract_feat_alldiseases_CTRATE.sh|-|NUM_SHARDS=${CTRATE_SHARDS} SHARD=${sh}")
done

echo "============================================================"
echo "GPUs:         ${GPU_LIST[*]}  (jobs/gpu=${JOBS_PER_GPU}, max concurrent=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU )))"
echo "Results root: ${RESULTS_ROOT:-<per-script default: ${REPO_ROOT}/results>}"
echo "Logs:         ${LOG_DIR}"
echo "Units:        ${#units[@]}"
echo "============================================================"

if [[ "${DRYRUN}" == "1" ]]; then
    for unit in "${units[@]}"; do
        IFS='|' read -r label script diseases extra <<< "${unit}"
        echo "[dryrun] ${label}  ->  ${script}$([[ "${diseases}" != "-" ]] && echo "  DISEASES='${diseases}'")$([[ "${extra}" != "-" ]] && echo "  ${extra}")"
    done
    exit 0
fi

mkdir -p "${LOG_DIR}"
FAIL_LOG="${LOG_DIR}/_failed.txt"; : > "${FAIL_LOG}"

run_unit() {  # label script diseases extra_env gpu
    local label="$1" script="$2" diseases="$3" extra="$4" gpu="$5"
    local log="${LOG_DIR}/${label//\//__}.log"
    local start=${SECONDS}

    echo "[gpu ${gpu}] start  ${label}"
    local env_args=(CUDA_VISIBLE_DEVICES="${gpu}")
    [[ -n "${RESULTS_ROOT:-}" ]] && env_args+=(RESULTS_ROOT="${RESULTS_ROOT}")
    [[ "${diseases}" != "-" ]]  && env_args+=(DISEASES="${diseases}")
    # extra is a space-separated "K=V K=V" list, so it must word-split here.
    # shellcheck disable=SC2206
    [[ "${extra}" != "-" ]]     && env_args+=(${extra})

    if env "${env_args[@]}" "${SCRIPT_DIR}/${script}" > "${log}" 2>&1; then
        echo "[gpu ${gpu}] ok     ${label}  ($(( SECONDS - start ))s)"
    else
        echo "[gpu ${gpu}] FAILED ${label}  ($(( SECONDS - start ))s)  see ${log}"
        echo "${label}" >> "${FAIL_LOG}"
    fi
}

# --- one queue, JOBS_PER_GPU workers per GPU -------------------------------
# The units go down a FIFO that every worker reads from, so a worker that
# finishes early immediately takes the next unit instead of waiting on a fixed
# share.
#
# The pop MUST be serialised. bash's `read` consumes a pipe one byte at a time
# (so it cannot over-read past the newline), which is NOT atomic against other
# readers on the same fd: concurrent workers interleave their single-byte reads
# and each ends up with a subset of the line's characters. That is silent -- it
# surfaces as mangled unit labels ("AMOS_clf_tr_vl/fatty_lie") and units that
# never run. It is a race, so it hides at low worker counts and bites at high
# ones. flock around the read makes exactly one worker pop at a time.
queue="$(mktemp -u)"; mkfifo "${queue}"
printf '%s\n' "${units[@]}" > "${queue}" &
exec 3< "${queue}"
rm -f "${queue}"

# Each worker OPENS the lock file itself. flock() attaches the lock to the open
# file description, so workers that merely inherit one parent fd all share a
# single lock and exclude nobody -- the pop stays racy and the corruption above
# survives. A per-worker open gives each its own description, which is what
# makes flock actually serialise them.
queue_lock="$(mktemp)"

for gpu in "${GPU_LIST[@]}"; do
    for (( slot = 0; slot < JOBS_PER_GPU; slot++ )); do
        (
            exec 9> "${queue_lock}"
            while :; do
                flock 9
                if IFS='|' read -r label script diseases extra <&3; then
                    flock -u 9
                else
                    flock -u 9
                    break
                fi
                run_unit "${label}" "${script}" "${diseases}" "${extra}" "${gpu}"
            done
        ) &
    done
done
wait
exec 3<&-
rm -f "${queue_lock}"

failed=$(wc -l < "${FAIL_LOG}")
echo "============================================================"
if (( failed > 0 )); then
    echo "FAILED units (${failed}/${#units[@]}):"
    sed 's/^/  /' "${FAIL_LOG}"
    echo "Logs: ${LOG_DIR}"
    exit 1
fi
echo "All ${#units[@]} units OK. Logs: ${LOG_DIR}"
