import argparse
import os
import subprocess as sp
import sys
import json
import hashlib
import numpy as np

from cosmos.api import Cosmos


def evaluate(parameters, out_s3_uri, train, calibrate, test, sleep):
    parameters_bash = json.dumps(parameters).replace('"', "'")
    return f"""
    python evaluate.py --parameters \"{parameters_bash}\" --train {train} --calibrate {calibrate} --test {test} --id {id}
    aws s3 cp score.json {out_s3_uri}
    
    sleep {sleep}
    """


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--id",
        type=str,
        help="evaluation id used to load parameters from file with name parameters_to_evaluate-id.json",
        required=True,
    )
    p.add_argument(
        "-i",
        "--container-image",
        help="you must have the aws command line tools installed, and it must have access to copy from the s3 bucket "
        "defined in --s3-prefix-for-command-script-temp-files (ie the container should be able to run "
        "`s3 cp s3://bucket ./`",
        required=False,
        default="467123434025.dkr.ecr.us-east-2.amazonaws.com/bachelor-test:latest",
    )
    p.add_argument(
        "-b",
        "--s3-prefix-for-command-script-temp-files",
        help="Bucket to use for storing command scripts as temporary files, ex: s3://my-bucket/cosmos/tmp_files",
        required=False,
        default="s3://danielgafni-personal/bachelor/tmp",
    )
    p.add_argument(
        "-q",
        "--default-queue",
        default="bachelor",
        help="aws batch queue",
        required=False,
    )
    p.add_argument(
        "-o",
        "--out-s3-uri",
        help="s3 uri to store output of tasks",
        required=False,
        default="s3://danielgafni-personal/bachelor",
    )
    p.add_argument(
        "--core-req", help="number of cores to request for the job", default=4
    )
    p.add_argument(
        "--mem-req", help="memory (mb) to request for the job", default=10000
    )
    p.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="number of seconds to have the job sleep for.  Useful for debugging so "
        "that you can ssh into the instance running a task",
    )
    p.add_argument(
        "--train",
        type=int,
        help="number of train iterations",
        default=5000,
        required=False,
    )
    p.add_argument(
        "--calibrate",
        type=int,
        help="number of calibration iterations",
        default=10000,
        required=False,
    )
    p.add_argument(
        "--test",
        type=int,
        help="number of test iterations",
        default=10000,
        required=False,
    )

    p.add_argument(
        "--max-attempts",
        default=10,
        help="Number of times to retry a task.  A task will only be retried if it is a spot instance because of the "
        "retry_only_if_status_reason_matches value we set in the code.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cosmos = Cosmos(
        "sqlite:///%s/sqlite.db" % os.path.dirname(os.path.abspath(__file__)),
        default_drm="awsbatch",
        default_drm_options=dict(
            container_image=args.container_image,
            s3_prefix_for_command_script_temp_files=args.s3_prefix_for_command_script_temp_files,
            # only retry on spot instance death
            retry_only_if_status_reason_matches="Host EC2 .+ terminated.",
        ),
        default_queue=args.default_queue,
    )

    cosmos.initdb()

    # sp.check_call("mkdir -p analysis_output/ex1", shell=True)
    # os.chdir("analysis_output/ex1")
    workflow = cosmos.start(f"Evaluate_{args.id}", restart=True, skip_confirm=True)

    parameters = np.load(f"optimize_awsbatch/parameters/{args.id}.npy")

    for i, par in enumerate(parameters):
        parameters = dict(
            mean_weight=par[0],
            c_w=par[1],
            tau_pos=par[2],
            tau_neg=par[3],
            A_pos=par[4],
            A_neg=par[5],
            weight_devay=par[6],
            n_filters=25,
            time_max=250,
            crop=20,
            kernel_size=16,
            stride=4,
            intensity=127.5,
            c_w_min=None,
            c_l=True,
            network_type="LC_SNN",

        )
        workflow.add_task(
            func=evaluate,
            params=dict(
                parameters=parameters,
                out_s3_uri=f"{args.out_s3_uri}/scores/{args.id}/{i}.json",
                sleep=args.sleep,
                train=args.train,
                calibrate=args.calibrate,
                test=args.test
            ),
            uid=str(i),
            time_req=None,
            max_attempts=args.max_attempts,
            core_req=args.core_req,
            mem_req=args.mem_req,
        )
    workflow.run()

    sys.exit(0 if workflow.successful else 1)


if __name__ == "__main__":
    main()
