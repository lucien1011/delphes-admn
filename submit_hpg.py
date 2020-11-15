def submit(pyscript,cfg,base_path):
    import os,pickle
    
    from PyUtilities.mkdir_p import mkdir_p
    from SLURMWorker.SLURMWorker import SLURMWorker

    mkdir_p(cfg.output_path)
    script_file_name = os.path.join(cfg.output_path,cfg.job_name+".cfg")
    pickle_file_name = os.path.join(cfg.output_path,cfg.job_name+".pkl")
    pickle.dump(cfg,open(pickle_file_name,"wb"))
    
    commands = """echo \"{job_name}\"
    cd {base_path}
    source setup_hpg.sh
    python {pyscript} {cfg_path}
    cp /scratch/local/$SLURM_JOBID/_%j.log {output_path}
    """.format(
            job_name=cfg.job_name,
            pyscript=pyscript,
            base_path=base_path,
            cfg_path=pickle_file_name,
            output_path=cfg.output_path,
            )
    
    worker = SLURMWorker()
    worker.make_sbatch_script(
            script_file_name,
            cfg.job_name,
            "kin.ho.lo@cern.ch",
            "1",
            "128gb",
            "72:00:00",
            cfg.log_output_path,
            commands,
            gpu="geforce:2",
            )
    worker.sbatch_submit(script_file_name)

if __name__ == "__main__":
    import os,sys,importlib

    spec = importlib.util.spec_from_file_location("preprocess_data_cfg",sys.argv[2])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.cfg

    submit(sys.argv[1],cfg,os.environ['BASE_PATH'],)
