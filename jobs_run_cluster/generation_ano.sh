#!/usr/bin/env bash

source env.sh
[[ "$*" == *"--test"* ]] && export GENERATION_DIR=$GENERATION_DIR_TEST CONFIG_SRC=$CONFIG_SRC_TEST

# Remove --test from args so getopts doesn't choke on it
args=()
for arg in "$@"; do [[ "$arg" != "--test" ]] && args+=("$arg"); done
set -- "${args[@]}"

# ===== DEFAULT VALUES =====
dnum=$DEFAULT_DSNUM
nbcon=$NB_CONTRASTS
anotype=(0)
ano_names=()

# ===== OPTION HANDLING =====
while getopts "d:z:a:s:h:n" opt; do
  case $opt in

    d)
      dnum="$OPTARG"
      ;;
    z)
      nbcon="$OPTARG"
      ;;
    a)
      # Convert the string "1 3" into a bash array (1 3)
      read -ra anotype <<< "$OPTARG"
      ;;
    s)
      SUBJECT_IDX="$OPTARG"
      ;;
    n)
      number_of_subjects="true"
      ;;
    h)
      echo "Usage: $0 [-d dnum] [-z nbcon] [-a \"anomaly_list\"]"
      echo ""
      echo "Options:"
      echo "  -d    Dataset index (1-6)"
      echo "  -z    Number of contrasts"
      echo "  -a    List of anomalies to generate"
      echo "  -s    Index of the subject to generate (default: all subjects)"
      echo "  -h    Show this help message"
      echo "  -n    Show number of subjects in the dataset"
      echo "        Examples: -a \"1 3\"    or    -a \"2\""
      echo "        Anomaly names: ${ano_names[*]}"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# get ano_names from args -a: find each ID's position in ANO_LIST to get the matching name
all_ano_ids=($ANO_LIST)
all_ano_names=($ANO_NAMES)
ano_names=()
for a in "${anotype[@]}"; do
  for idx in "${!all_ano_ids[@]}"; do
    if [[ "${all_ano_ids[$idx]}" == "$a" ]]; then
      ano_names+=("${all_ano_names[$idx]}")
      break
    fi
  done
done

# ===== MAIN LOGIC =====
dname=${DSNAME[$dnum-1]}

echo "Dataset: $dname"
echo "Requested anomalies: ${anotype[*]}"



working_dir=$GENERATION_DIR/${dname}

# Loop over the anotype array elements
i=0;
for nba in "${anotype[@]}"
do
    WD="${working_dir}/Ano_${ano_names[$i]}"
    mkdir -p "$WD"
    
    # Create a symbolic link to the configs.ini file in the working directory if it doesn't already exist
    if [ ! -e "${WD}/configs.ini" ]; then
      ln -s "${CONFIG_SRC}" "${WD}/configs.ini"
    fi

    if [ "$number_of_subjects" == "true" ]; then
      echo "Getting number of subjects in dataset [$dnum] ($dname)"
      echo python "${ROOT_PROG}/generate.py" -o "$WD"  -d "${dnum}"  -c configs.ini -a "$nba"  -z "${nbcon}" --get_nb_subjects
      python "${ROOT_PROG}/generate.py" -o "$WD"  -d "${dnum}"  -c configs.ini -a "$nba"  -z "${nbcon}" --get_nb_subjects
      exit 0
    fi
    
    echo "Starting GENERATION Anomalies [$nba] (${ano_names[$i]})"
    if [ -n "$SUBJECT_IDX" ]; then
      echo python "${ROOT_PROG}/generate.py" -o "$WD"  -d "${dnum}"  -c configs.ini -a "$nba"  -z "${nbcon}" -s "$SUBJECT_IDX"
      python "${ROOT_PROG}/generate.py" -o "$WD"  -d "${dnum}"  -c configs.ini -a "$nba"  -z "${nbcon}" -s "$SUBJECT_IDX"
    else
      echo python "${ROOT_PROG}/generate.py" -o "$WD"  -d "${dnum}"  -c configs.ini -a "$nba"  -z "${nbcon}"
      python "${ROOT_PROG}/generate.py" -o "$WD"  -d "${dnum}"  -c configs.ini -a "$nba"  -z "${nbcon}"
    fi
    i=$((i+1))
done
