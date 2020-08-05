import argparse
import json
import sys

from thesis.nets import LC_SNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--file",
        help="network parameters",
        required=True,
    )
    p.add_argument(
        "--train",
        help="number of train iterations",
        default=5000,
        required=False,
    )
    p.add_argument(
        "--calibrate",
        help="number of calibration iterations",
        default=False,
        required=False,
    )
    p.add_argument(
        "--test",
        help="number of test iterations",
        default=5000,
        required=False,
    )

    args = p.parse_args()

    with open(args.file, 'r') as file:
        parameters = json.load(file)

    if parameters.network_type == 'LC_SNN':
        net = LC_SNN(
            mean_weight=parameters.mean_weight,
            c_w=parameters.c_w,
            time_max=parameters.time_max,
            crop=parameters.crop,
            kernel_size=parameters.kernel_size,
            n_filters=parameters.n_filters,
            stride=parameters.stride,
            intensity=parameters.intensity,
            tau_pos=parameters.tau_pos,
            tau_neg=parameters.tau_neg,
            c_w_min=parameters.c_w_min,
            c_l=parameters.c_l,
            A_pos=parameters.A_pos,
            A_neg=parameters.A_neg,
            weight_decay=parameters.weight_decay,
            immutable_name=None,
            foldername=None,
            loaded_from_disk=False,
            n_iter=0
        )
        net.train(5000)
        net.calibrate(10000)
        net.calculate_accuracy(10000)

        sys.stdout.write(str(net.score))


if __name__ == '__main__':
    main()
