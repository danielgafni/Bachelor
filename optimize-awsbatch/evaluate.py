import argparse
import json
import sys

from thesis.nets import LC_SNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parameters", type=str, help="network parameters", required=True,
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
        default=False,
        required=False,
    )
    p.add_argument(
        "--test",
        type=int,
        help="number of test iterations",
        default=5000,
        required=False,
    )

    args = p.parse_args()

    parameters = json.loads(args.parameters.replace("'", '"'))

    if parameters["network_type"] == "LC_SNN":
        net = LC_SNN(
            mean_weight=parameters["mean_weight"],
            c_w=parameters["c_w"],
            time_max=parameters["time_max"],
            crop=parameters["crop"],
            kernel_size=parameters["kernel_size"],
            n_filters=parameters["n_filters"],
            stride=parameters["stride"],
            intensity=parameters["intensity"],
            tau_pos=parameters["tau_pos"],
            tau_neg=parameters["tau_neg"],
            c_w_min=parameters["c_w_min"],
            c_l=parameters["c_l"],
            A_pos=parameters["A_pos"],
            A_neg=parameters["A_neg"],
            weight_decay=parameters["weight_decay"],
            immutable_name=None,
            foldername=None,
            loaded_from_disk=False,
            n_iter=0,
        )
    else:
        raise NotImplementedError
    net.train(args.train, download=True)
    net.calibrate(args.calibrate)
    net.calculate_accuracy(args.test)

    with open("score.json", "w") as file:
        json.dump(net.score, file)

    with open("name.txt", "w") as file:
        file.write(f"{net.name}.json")


if __name__ == "__main__":
    main()
