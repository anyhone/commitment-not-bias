#!/usr/bin/env bash
# Final evaluation on the test split (run once) with the configurations selected on dev
# (results/<model>/report/selection.json, criterion fixed in report.py) + controls.
# usage: scripts/run_test.sh <model>
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:?model}
PY=.venv/bin/python
SEL=results/$M/report/selection.json
[ -f "$SEL" ] || { echo "missing $SEL: run sweep and report first"; exit 1; }
# prints: layer, coefficient (:g format as in the evaluate.py tags), mode (rel|raw)
get() { $PY -c "import json;s=json.load(open('$SEL'));c=s['$1']['$2'];print(int(c['layer']), format(c['coef'],'g'), c.get('coef_mode','rel'))"; }
read CB_L CB_C CB_M < <(get caa bbq)
read CS_L CS_C CS_M < <(get caa ss)
read FB_L FB_C FB_M < <(get fairsteer bbq)
read FS_L FS_C FS_M < <(get fairsteer ss)
flag() { [ "$1" = raw ] && echo "--raw" || true; }
echo "selection: CAA bbq L$CB_L $CB_M$CB_C | CAA ss L$CS_L $CS_M$CS_C | FS bbq L$FB_L $FB_M$FB_C | FS ss L$FS_L $FS_M$FS_C"
EV="$PY scripts/evaluate.py --model $M --split test"
run() { local d=results/$M/eval/test/$1; shift
        if [ -f "$d/metrics.json" ]; then echo "skipping $d (already done)"; else $EV "$@"; fi; }
run baseline                                   --method baseline
# main configurations (global vector): MMLU with the vector of the benchmark they were selected for
run "caa_L${CB_L}_${CB_M}${CB_C}"              --method caa       --layer $CB_L --coef $CB_C $(flag $CB_M) --mmlu-sources bbq
run "fairsteer_L${FB_L}_${FB_M}${FB_C}"        --method fairsteer --layer $FB_L --coef $FB_C $(flag $FB_M) --mmlu-sources bbq
[ "$CS_L $CS_C $CS_M" != "$CB_L $CB_C $CB_M" ] && run "caa_L${CS_L}_${CS_M}${CS_C}" --method caa --layer $CS_L --coef $CS_C $(flag $CS_M) --mmlu-sources ss
[ "$FS_L $FS_C $FS_M" != "$FB_L $FB_C $FB_M" ] && run "fairsteer_L${FS_L}_${FS_M}${FS_C}" --method fairsteer --layer $FS_L --coef $FS_C $(flag $FS_M) --mmlu-sources ss
# directional control: random vector of equal norm at the selected CAA configurations
run "random_L${CB_L}_${CB_M}${CB_C}"           --method random    --layer $CB_L --coef $CB_C $(flag $CB_M) --mmlu-sources bbq
[ "$CS_L $CS_C $CS_M" != "$CB_L $CB_C $CB_M" ] && run "random_L${CS_L}_${CS_M}${CS_C}" --method random --layer $CS_L --coef $CS_C $(flag $CS_M) --mmlu-sources ss
# ablations at the BBQ configuration: per-category vector (oracle) and text-specification BAD labels
run "caa_L${CB_L}_${CB_M}${CB_C}_cat"          --method caa       --layer $CB_L --coef $CB_C $(flag $CB_M) --vector category --mmlu-sources bbq
run "fairsteer_L${FB_L}_${FB_M}${FB_C}_cat"    --method fairsteer --layer $FB_L --coef $FB_C $(flag $FB_M) --vector category --mmlu-sources bbq
run "fairsteer_L${FB_L}_${FB_M}${FB_C}_probes_text" --method fairsteer --layer $FB_L --coef $FB_C $(flag $FB_M) --probe-set probes_text --mmlu-sources bbq
$PY scripts/report.py --model $M
echo "test done for $M"
