#!/bin/bash
# R04: regenerate every table in results/ that is computed from stored results/*.json (no GPU, no audio). usage (from the topic folder): bash src/99_make_all_tables.sh
set -e
python3 src/80_make_tables.py        > /dev/null   # results/tables_main.txt
python3 src/81_quality_kad_table.py  > /dev/null   # results/quality_kad_table.txt
python3 src/82_tagger_table.py       > /dev/null   # results/discogs_table.txt
python3 src/84_fd_table.py           > /dev/null   # results/fd_vgg_table.txt
python3 src/82b_tagger_six.py base_ge7=tri_base_m3e7+trix_base_m3e7 lora_ge7=tri_lora_m3e7+trix_lora_m3e7 lora_ge10=tri_lora_m3e10+trix_lora_m3e10 > /dev/null   # results/discogs_six.txt
python3 src/78_six_decades.py base_ge7=results/tri_base_m3e7_eval.json+results/trix_base_m3e7_eval.json lora_ge7=results/tri_lora_m3e7_eval.json+results/trix_lora_m3e7_eval.json lora_ge10=results/tri_lora_m3e10_eval.json+results/trix_lora_m3e10_eval.json lora_ge15=results/tri_lora_m3e15_eval.json+results/trix_lora_m3e15v1_eval.json lora_ge20=results/tri_lora_m3e20_eval.json+results/trix_lora_m3e20v1_eval.json seed1_ge10=results/tri_lora_m3e10v1s1_eval.json+results/trix_lora_m3e10v1s1_eval.json seed2_ge10=results/tri_lora_m3e10v1s2_eval.json+results/trix_lora_m3e10v1s2_eval.json > /dev/null   # results/six_decades.txt
python3 src/79_extra_content_table.py trix_base_m3e7 trix_lora_m3e7 trix_lora_m3e10 trix_lora_m3e15v1 trix_lora_m3e20v1 trix_lora_m3e10v1s1 trix_lora_m3e10v1s2 > /dev/null   # results/extra_content.txt
python3 src/79b_extra_quality_table.py trix_base_m3e7 trix_lora_m3e7 trix_lora_m3e10 trix_lora_m3e15v1 trix_lora_m3e20v1 > /dev/null   # results/extra_quality.txt
python3 src/78b_prompt_variants_decade.py 3 ge10=results/trp90_lora_m3e10_eval.json:results/trp90_lora_m3e10_content.json ge15=results/trp90_lora_m3e15_eval.json:results/trp90_lora_m3e15_content.json > /dev/null   # results/prompt_variants_decade.txt
python3 src/73_probe_augment.py --help > /dev/null 2>&1 || true
echo "tables regenerated"
