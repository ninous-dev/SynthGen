import os
import pickle

#TODO use that instead of create_jobs

def load_params(params_fname):
    """Load parameters from a file or dictionary."""
    if isinstance(params_fname, str):
        with open(params_fname, "rb") as f:
            return pickle.load(f)
    elif isinstance(params_fname, dict):
        return params_fname
    raise ValueError("Invalid params_fname. Must be a file path or a dictionary.")


def setup_directories(params):
    """Set up necessary output directories."""
    output_dir = params["output_directory"]
    os.makedirs(output_dir, exist_ok=True)

    if "output_results" in params:
        os.makedirs(params["output_results"], exist_ok=True)

    # Confirm before clearing existing directories
    if os.path.isdir(output_dir):
        confirm = input(
            "Are you sure you want to delete the Job directory? (press Enter to confirm): "
        )
        if not confirm:
            os.system(f"rm -rf {output_dir}")
        else:
            print("Directory cleanup skipped.")

    os.makedirs(output_dir, exist_ok=True)


def copy_scripts(params):
    """Copy scripts to the designated output directory."""
    if "scripts_to_copy" in params:
        scripts_dir = os.path.join(
            params.get("output_results", params["output_directory"]), "scripts"
        )
        os.system(f"rm -rf {scripts_dir}")
        os.makedirs(scripts_dir, exist_ok=True)
        os.system(
            f"rsync -aux --exclude=*pyc --exclude=*.git {params['scripts_to_copy']} {scripts_dir}"
        )


def write_job_files(params, jobs_dir, logs_dir, jobs):
    """Generate job scripts and the associated batch files."""
    do_all_local_file = os.path.join(jobs_dir, "do_all_local.bash")
    with open(do_all_local_file, "w") as fd_all:
        fd_all.write("# parallel -j 12 < " + do_all_local_file + "\n")

        for i, job in enumerate(jobs, 1):
            job_file = os.path.join(jobs_dir, f"j{i:03d}.bash")
            log_file = os.path.join(logs_dir, f"log-{i}")
            err_file = os.path.join(logs_dir, f"err-{i}")

            with open(job_file, "w") as fd_job:
                job_script = [
                    "#!/bin/bash",
                    f"#SBATCH {params['cluster_queue']}",
                    f"#SBATCH -n {params['cpus_per_task']}",
                    f"#SBATCH --mem={params['mem']}",
                    job,
                ]
                # Add gres line only if it exists in params
                if "gres" in params:
                    job_script.insert(2, f"#SBATCH {params['gres']}")

                fd_job.write("\n".join(job_script))
            os.chmod(job_file, 0o755)

            # Add job to the local batch file
            fd_all.write(f"bash {job_file} > {log_file} 2> {err_file}\n")

    os.chmod(do_all_local_file, 0o755)


def create_sbatch_file(params, jobs, jobs_dir):
    """Create batch job scripts for submission."""
    setup_directories(params)
    copy_scripts(params)

    logs_dir = os.path.join(jobs_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    write_job_files(params, jobs_dir, logs_dir, jobs)

def create_do_qsub_script(params, jobs_dir):
    """Create the do_qsub.bash file."""
    directory = jobs_dir
    logs_dir = os.path.join(jobs_dir, "logs")
    os.makedirs(directory, exist_ok=True)

    array_size = len(os.listdir(jobs_dir))

    script_content = (
        f"export jobid=`sbatch {params['cluster_queue']} "
        f"-t {params['walltime']} "
        f"-N 1 --cpus-per-task={params['cpus_per_task']} "
    )

    if "mem_per_cpu_MB" in params:
        script_content += f"--mem-per-cpu={params['mem_per_cpu_MB']} "
    if "mem" in params:
        script_content += f"--mem={params['mem']} "
    if "sbatch_args" in params:
        script_content += f"{params['sbatch_args']} "

    script_content += (
        f"--job-name={params['job_name']} "
        f"-o {logs_dir}/log-%A_%a -e {logs_dir}/err-%A_%a --array=1-{array_size} "
        f"{os.path.join(directory, 'do_job_array.bash')} |awk '{{print $4}}'`\n"
        f"echo submitted job $jobid\n"
    )

    script_path = os.path.join(directory, "do_qsub.bash")
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.write(script_content)
    os.chmod(script_path, 0o755)

def create_do_job_array_script(jobs_dir):
    """Create the do_job_array.bash file."""
    do_job_array_file = os.path.join(jobs_dir, 'do_job_array.bash')
    with open(do_job_array_file, "w", encoding="utf-8") as script_file:
        script_file.write( "#!/bin/bash\n\n" )
        script_file.write( "echo started on $HOSTNAME\n" )
        script_file.write( "date\n" )
        script_file.write( "tic=\"$(date +%s)\"\n" )
        script_file.write( "cmd=$( printf \"j%.3d.bash\" ${SLURM_ARRAY_TASK_ID})\n" )
        script_file.write( "bash " + jobs_dir + "/$cmd\n" )
        script_file.write( "toc=\"$(date +%s)\";\n" )
        script_file.write( "sec=\"$(expr $toc - $tic)\";\n" )
        script_file.write( "min=\"$(expr $sec / 60)\";\n" )
        script_file.write( "heu=\"$(expr $sec / 3600)\";\n" )
        script_file.write( "echo Elapsed time: $min min $heu H \n" )

    os.system( 'chmod +x ' + do_job_array_file )
