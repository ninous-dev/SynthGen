import os
import os.path as op
import logging
import torchio as tio
import re
import argparse
from generate import add_anomalies_map, load_config, get_label_set_config

# Configure logging
logger = logging.getLogger(__name__)

# TODO
raise RuntimeError("Not updatae call (model is now dataset for example ..etc)")
#TODO add to a post processing tool or a analysis tool, not in the generation script, it is not really a transform but a post processing step to get the anomaly maps from the generated images. It is also not really specific to the generation of anomalies, it can be used for any kind of transformation that generates an anomaly map from a label map.
def parse_args():
    p = argparse.ArgumentParser(description="Rebuild automatically anomalies maps from generate_*")
    p.add_argument("--root_dir", required=True, help="Root directories containing generate_*")
    p.add_argument("--model", type=int, required=True, help="Model number (0=vasc, 1=skull, 2=mida)")
    p.add_argument("--config", default="configs.ini", help="Config filename, by default `configs.ini`")
    p.add_argument("--ano_label", type=int, required=True, help="First label index for anomalies. All labels from this value up to the maximum label will be treated as anomalies.")
    p.add_argument("--output_dir", "-o", default=None, type=str, help="Output directory, by default [root_dir]/anomaly_maps")

    args = p.parse_args()
    args.config_path = op.join(args.root_dir, args.config)

    return args

# Main
if __name__ == "__main__":
    args = parse_args()

    if not op.exists(args.config_path):
        raise FileNotFoundError(f"configs.ini in {args.root_dir}")

    # Load configs
    config = load_config(args.root_dir, args.model, args.config)
    label_set_name, files, dict_lab, dict_map_synth, dict_map_target, keep_labels_dict = get_label_set_config(args.model, config.model)

    # Build map suj to file
    subject_map = {f"{label_set_name}_Suj_{i:02d}": files[i] for i in range(len(files))}
    gen_dirs = [op.join(args.root_dir, d) for d in os.listdir(args.root_dir) if d.startswith("generate_") and op.isdir(op.join(args.root_dir, d))]

    if not gen_dirs:
        raise FileNotFoundError(f"generate_* in {args.root_dir}")

    amaps_dir = args.output_dir or op.join(args.root_dir, "anomaly_maps")
    os.makedirs(amaps_dir, exist_ok=False)

    for gen_dir in gen_dirs:
        logger.info(f"--- Processing {gen_dir} ---")
        ano_files = [f for f in os.listdir(gen_dir) if f.startswith("Ano_") and f.endswith(".nii.gz")]
        if not ano_files:
            logger.warning(f"No Ano_*.nii.gz for {gen_dir}")
            continue

        for fname in ano_files:
            base = fname[:-7]
            m = re.match(r"^Ano_(gen\d+)_([A-Za-z0-9]+)_Suj_(\d+)", base)
            if not m:
                raise ValueError(f"Ano regex not corresponding.")

            gen_id, labelset, suj_idx = m.groups()
            subject_name = f"{labelset}_Suj_{int(suj_idx):02d}"

            ano_path = op.join(gen_dir, fname)


            if subject_name not in subject_map:
                raise ValueError(f"No original labels found for {subject_name} got {subject_map.keys()}")

            label_path = subject_map[subject_name]

            # Load anomalies map
            lm_ano = tio.LabelMap(ano_path)

            subject_anomalies = tio.Subject(
                label=lm_ano,
                anomalies_labels=list(range(args.ano_label, int(lm_ano.data.data.max().item() + 1)))
            )

            add_anomalies_map(
                subject_name=subject_name,
                subject_anomalies=subject_anomalies,
                output_dir=amaps_dir,
                label_file=label_path
            )