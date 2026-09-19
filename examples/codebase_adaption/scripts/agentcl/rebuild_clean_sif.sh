#!/usr/bin/env bash
# Rebuild the BigCodeBench eval SIF without bundled benchmark data (answer leak).
set -euo pipefail
source /home/qixinx/apps/apptainer/env.sh
D=/mnt/tmp/6016127/claude-6016127/-home-qixinx/2ad71a1e-0178-4da9-90ac-4a152bc5c82c/scratchpad
export APPTAINER_TMPDIR=$D/apptainer_tmp APPTAINER_CACHEDIR=$D/apptainer_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
SRC=/project/flame/qixinx/agentcl/sif/bigcodebench-evaluate-v0.2.4.sif
BOX=$D/bcb_sandbox
OUT=/project/flame/qixinx/agentcl/sif/bigcodebench-evaluate-v0.2.4-clean.sif
rm -rf "$BOX"
apptainer build --sandbox "$BOX" "$SRC"
chmod -R u+w "$BOX/root" "$BOX/bigcodebench" "$BOX/usr/local/lib/python3.10/site-packages" 2>/dev/null || true
rm -rf "$BOX/root/.cache" "$BOX/bigcodebench" \
       "$BOX/usr/local/lib/python3.10/site-packages/bigcodebench" \
       "$BOX"/usr/local/lib/python3.10/site-packages/bigcodebench-*
echo "remaining data-like files:"; find "$BOX" -xdev \( -iname "*bigcodebench*" -o -iname "*humaneval*" -o -iname "*mbpp*" -o -iname "*codeeval*" \) 2>/dev/null | head
apptainer build "$OUT" "$BOX"
rm -rf "$BOX"
ls -la "$OUT"
echo REBUILD_DONE
