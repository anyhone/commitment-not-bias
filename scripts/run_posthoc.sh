#!/usr/bin/env bash
# Runs for the post-hoc monitor analysis, at the FairSteer configurations already selected on dev:
#   - FairSteer without the monitor (--threshold -1) on dev and test;
#   - CAA at FairSteer's configuration on test (on dev it already exists in the sweep).
# Order: dev, then test at the BBQ configurations, last the StereoSet configurations.
# usage: scripts/run_posthoc.sh [model ...]   (default: llama mistral_instruct qwen)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(llama mistral_instruct qwen)

# prints: layer, coefficient (:g format as in the evaluate.py tags), mode (rel|raw)
get() { $PY -c "import json;s=json.load(open('results/$1/report/selection.json'));c=s['fairsteer']['$2'];print(int(c['layer']), format(c['coef'],'g'), c.get('coef_mode','rel'))"; }
flag() { [ "$1" = raw ] && echo "--raw" || true; }
run() { local m=$1 split=$2 d=results/$1/eval/$2/$3; shift 3
        if [ -f "$d/metrics.json" ]; then echo "skipping $d (already done)"; else $PY scripts/evaluate.py --model "$m" --split "$split" "$@"; fi; }
# configurations: <bench> -> "L C M"; SS only if different from BBQ
cfgs() { local m=$1; read BL BC BM < <(get "$m" bbq); read SL SC SM < <(get "$m" ss)
         echo "bbq $BL $BC $BM"; [ "$SL $SC $SM" != "$BL $BC $BM" ] && echo "ss $SL $SC $SM" || true; }

ungated() { local m=$1 split=$2 b=$3 L=$4 C=$5 M=$6
            run "$m" "$split" "fairsteer_L${L}_${M}${C}_t-1" --method fairsteer --layer "$L" --coef "$C" $(flag "$M") --threshold -1 --mmlu-sources "$b"; }
caafs()   { local m=$1 b=$2 L=$3 C=$4 M=$5
            run "$m" test "caa_L${L}_${M}${C}" --method caa --layer "$L" --coef "$C" $(flag "$M") --mmlu-sources "$b"; }

echo "== dev: FairSteer without monitor"
for m in "${MODELS[@]}"; do while read b L C M; do ungated "$m" dev "$b" "$L" "$C" "$M"; done < <(cfgs "$m"); done
echo "== test, BBQ configurations"
for m in "${MODELS[@]}"; do while read b L C M; do [ "$b" = bbq ] || continue
    ungated "$m" test bbq "$L" "$C" "$M"; caafs "$m" bbq "$L" "$C" "$M"; done < <(cfgs "$m"); done
echo "== test, StereoSet configurations"
for m in "${MODELS[@]}"; do while read b L C M; do [ "$b" = ss ] || continue
    ungated "$m" test ss "$L" "$C" "$M"; caafs "$m" ss "$L" "$C" "$M"; done < <(cfgs "$m"); done
echo "posthoc done"
