import pyperclip

from .nets import LC_SNN, C_SNN, FC_SNN, AbstractSNN
import os
import torch
import json
import hashlib
import pandas as pd
import plotly.graph_objs as go
from bindsnet.network.monitors import Monitor
from shutil import rmtree
from sqlite3 import connect


def view_network(name):
    """
    Get network parameters as dict
    :param name: network name
    :return: dict with network parameters
    """
    if not os.path.exists(f"networks//{name}"):
        print("Network with such name does not exist")
        return None
    else:
        try:
            with open(f"networks//{name}//parameters.json", "r") as file:
                parameters = json.load(file)
            with open(f"networks//{name}//score.json", "r") as file:
                score = json.load(file)
            best_method = "patch_voting"
            best_accuracy = 0
            for method in score.keys():
                if method == 'lc':
                    continue
                if score[method]["accuracy"] is not None:
                    if score[method]["accuracy"] > best_accuracy:
                        best_method = method
                        best_accuracy = score[method]["accuracy"]
            parameters["accuracy"] = score[best_method]["accuracy"]
            parameters["accuracy_method"] = best_method
            parameters["error"] = score[best_method]["error"]
            parameters["n_iter_accuracy"] = score[best_method]["n_iter"]

            return parameters
        except FileNotFoundError:
            return None


def view_database():
    """
    Get a pandas.DataFrame with all available networks
    :return: pandas.DataFrame with all network names and their parameters
    """
    columns = [
        "name",
        "network_type",
        "accuracy",
        "error",
        "n_iter",
        "n_filters",
        "kernel_size",
        "stride",
        "c_l",
        "tau_pos",
        "tau_neg",
        "A_pos",
        "A_neg",
        "weight_decay",
        "mean_weight",
        "c_w",
        "c_w_min",
        "crop",
        "time_max",
        "intensity",
        "dt",
        "train_method",
        "accuracy_method",
    ]
    database = pd.DataFrame(columns=[])
    for name in os.listdir("networks"):
        if "." not in name:
            if os.path.exists(f"networks//{name}//parameters.json"):
                parameters = view_network(name)

                try:
                    parameters["name"] = name
                except TypeError:
                    pass

                database = database.append(parameters, ignore_index=True)
    database = database[columns]
    return database


def plot_database(
    n_filters=100, network_type="LC_SNN", kernel_size=12, stride=4, c_l=False
):
    """
    Plots available networks as a 3D plot
    :param n_filters:
    :param network_type:
    :param kernel_size:
    :param stride:
    :param c_l:
    :return: plot
    """
    data = view_database()
    data = data[data["network_type"] == network_type]
    data = data[data["c_l"] == c_l]
    data = data[data["n_filters"] == n_filters]
    color = data["n_iter"]
    colorname = "n_iter"

    if network_type == "LC_SNN" or network_type == "C_SNN":
        data = data[data["kernel_size"] == kernel_size]
        data = data[data["stride"] == stride]
        figname = f"{network_type} networks with kernel size {kernel_size}, stride {stride} and {n_filters} filters"

    elif network_type == "FC_SNN":
        figname = f"{network_type} networks with {n_filters} filters"

    # data["error"] = (
    #     (data["accuracy"] * (1 - data["accuracy"]) / data["n_iter"]) ** 0.5
    # ).values

    fig = go.Figure(
        go.Scatter3d(
            x=data["c_w"],
            y=data["mean_weight"],
            z=data["accuracy"],
            hovertext=data["name"],
            error_z=dict(
                array=data["error"], visible=True, thickness=3, width=3, color='blue'
            ),
            mode="markers",
            marker=dict(
                size=5,
                cmax=color.max(),
                cmin=0,
                color=color,
                colorbar=dict(title=colorname),
                colorscale="Viridis",
            ),
        ),
    ).update_layout(
        title=go.layout.Title(text=figname, xref="paper"),
        height=1000,
        width=1000,
        margin={"l": 20, "r": 20, "b": 20, "t": 40, "pad": 4},
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgb(200,200, 230)",
                gridcolor="white",
                showbackground=True,
                title_text="c_w",
            ),
            yaxis=dict(
                backgroundcolor="rgb(230,200,230)",
                gridcolor="white",
                showbackground=True,
                title_text="mean_weight",
            ),
            zaxis=dict(
                backgroundcolor="rgb(230,230,200)",
                gridcolor="white",
                showbackground=True,
                title_text="accuracy",
            ),
        ),
    )

    fig = go.FigureWidget(fig)
    fig.data[0].on_click(click_point)
    return fig


def click_point(trace, points, selector):
    text = list(trace.hovertext)
    for i in points.point_inds:
        pyperclip.copy(text[i])


def load_network(name):
    """
    Load network from disk.
    :param name: network name
    :return: network
    """
    path = f"networks//{name}"
    try:
        with open(path + "//parameters.json", "r") as file:
            parameters = json.load(file)
            mean_weight = parameters["mean_weight"]
            c_w = parameters["c_w"]
            c_w_min = None
            if "c_w_min" in parameters.keys():
                c_w_min = parameters["c_w_min"]
            n_iter = parameters["n_iter"]
            time_max = parameters["time_max"]
            crop = parameters["crop"]
            if "kernel_size" in parameters.keys():
                kernel_size = parameters["kernel_size"]
            train_method = None
            if "train_method" in parameters.keys():
                train_method = parameters["train_method"]
            n_filters = parameters["n_filters"]
            if "stride" in parameters.keys():
                stride = parameters["stride"]
            intensity = parameters["intensity"]
            network_type = parameters["network_type"]
            c_l = False
            if "c_l" in parameters.keys():
                c_l = parameters["c_l"]
            A_pos = parameters["A_pos"]
            A_neg = parameters["A_neg"]
            tau_pos = parameters["tau_pos"]
            tau_neg = parameters["tau_neg"]

    except FileNotFoundError:
        print("Network folder is corrupted.")
        raise FileNotFoundError

    if network_type == "LC_SNN":
        net = LC_SNN(
            mean_weight=mean_weight,
            c_w=c_w,
            c_w_min=c_w_min,
            time_max=time_max,
            crop=crop,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            intensity=intensity,
            c_l=c_l,
            A_pos=A_pos,
            A_neg=A_neg,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            weight_decay=parameters["weight_decay"],
            immutable_name=parameters["immutable_name"],
            foldername=name,
            loaded_from_disk=True,
            n_iter=n_iter,
        )

    elif network_type == "C_SNN":
        net = C_SNN(
            mean_weight=mean_weight,
            c_w=c_w,
            c_w_min=c_w_min,
            c_l=c_l,
            A_pos=A_pos,
            A_neg=A_neg,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            weight_decay=parameters["weight_decay"],
            time_max=time_max,
            crop=crop,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            intensity=intensity,
            immutable_name=parameters["immutable_name"],
            foldername=name,
            loaded_from_disk=True,
            n_iter=n_iter,
        )

    elif network_type == "FC_SNN":
        net = FC_SNN(
            mean_weight=mean_weight,
            c_w=c_w,
            c_w_min=c_w_min,
            c_l=c_l,
            A_pos=A_pos,
            A_neg=A_neg,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            weight_decay=parameters["weight_decay"],
            time_max=time_max,
            crop=crop,
            n_filters=n_filters,
            intensity=intensity,
            immutable_name=parameters["immutable_name"],
            foldername=name,
            loaded_from_disk=True,
            n_iter=n_iter,
        )

    else:
        print("This network type is not implemented for loading yet")
        raise NotImplementedError

    try:
        loaded_network = torch.load(path + "//network")
    except FileNotFoundError:
        raise FileNotFoundError("Network file not found")

    for c in loaded_network.connections:
        net.network.connections[c].w.data = loaded_network.connections[c].w.data

    if os.path.exists(path + "//votes"):
        votes = torch.load(path + "//votes")
        net.calibrated = True
    conf_matrix = None
    if os.path.exists(path + "//confusion_matrix"):
        conf_matrix = torch.load(path + "//confusion_matrix")
    with open(f"networks//{name}//score.json", "r") as file:
        score = json.load(file)

    net.votes = votes
    net.score = score
    net.conf_matrix = conf_matrix

    net.spikes = {}
    net.spikes["Y"] = Monitor(
        net.network.layers["Y"], state_vars=["s"], time=net.time_max
    )
    net.network.add_monitor(net.spikes["Y"], name="Y_spikes")

    net.voltages = {}
    net.voltages["Y"] = Monitor(
        net.network.layers["Y"], state_vars=["v"], time=net.time_max
    )
    net.network.add_monitor(net.voltages["Y"], name="Y_voltages")

    net.thetas = {}
    net.thetas["Y"] = Monitor(
        net.network.layers["Y"], state_vars=["theta"], time=net.time_max
    )
    net.network.add_monitor(net.thetas["Y"], name="Y_thetas")

    net.network.train(False)
    net.train_method = train_method
    for c in net.network.connections:
        net.network.connections[c].learning = False

    return net


def delete_network(name, sure=False):
    """
    Delete network from disk.
    :param name: network name
    :param sure: True to skip deletion dialog
    """
    if not sure:
        print("Are you sure you want to delete the network? [Y/N]")
        if input() == "Y":
            rmtree(f"networks//{name}")
            conn = connect(r"networks/networks.db")
            crs = conn.cursor()
            crs.execute(f"DELETE FROM networks WHERE name = ?", (name,))
            conn.commit()
            conn.close()
            print("Network deleted!")
        else:
            print("Deletion canceled...")
    else:
        rmtree(f"networks//{name}")
        conn = connect(r"networks/networks.db")
        crs = conn.cursor()
        crs.execute(f"DELETE FROM networks WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        print("Network deleted!")


def clean_database():
    """
    Clean database. Renames networks according to their current parameters.
    Run this if any problems with the database happened.
    """

    if not os.path.exists(r"networks/networks.db"):
        conn = connect(r"networks/networks.db")
        crs = conn.cursor()
        crs.execute(
            """CREATE TABLE networks(
             name BLOB,
             accuracy REAL,
             type BLOB
             )"""
        )
        conn.commit()
        conn.close()

    #  Clear existing database
    conn = connect(r"networks/networks.db")
    crs = conn.cursor()
    crs.execute("DELETE FROM networks")
    conn.close()

    for name_ in os.listdir("networks"):
        if name_ != "networks.db":
            #  Delete networks without saved parameters
            if not os.path.exists(f"networks//{name_}//parameters.json"):
                rmtree(f"networks//{name_}")

    for name in os.listdir("networks"):
        if name != "networks.db":
            with open(f"networks//{name}//parameters.json", "r") as file:
                parameters = json.load(file)

            # Remove networks with duplicated parameters
            for other_name in os.listdir("networks"):
                if other_name != name:
                    if other_name != "networks.db":
                        with open(
                            f"networks//{other_name}//parameters.json", "r"
                        ) as file:
                            other_parameters = json.load(file)

                        if parameters == other_parameters:
                            rmtree(f"networks//{other_name}")

            #  Rename networks according to current parameters and add it to the database
            if not parameters["immutable_name"]:
                rename_network(name)


def rename_network(name, new_name=None):
    if isinstance(name, AbstractSNN):
        name.rename(new_name)
    else:
        with open(f"networks//{name}//parameters.json", "r") as file:
            parameters = json.load(file)

            if new_name is None:
                parameters["immutable_name"] = False
                new_name = hashlib.sha224(str(parameters).encode("utf8")).hexdigest()
            else:
                parameters["immutable_name"] = True

            os.rename(f"networks//{name}", f"networks//{new_name}")
            if os.path.exists(f"activity//{name}"):
                os.rename(f"activity//{name}", f"activity//{new_name}")

            with open(f"networks//{new_name}//parameters.json", "w") as file:
                json.dump(parameters, file)

            if not os.path.exists(r"networks/networks.db"):
                conn = connect(r"networks/networks.db")
                crs = conn.cursor()
                crs.execute(
                    """CREATE TABLE networks(
                     name BLOB,
                     accuracy REAL,
                     type BLOB
                     )"""
                )
                conn.commit()
                conn.close()

            conn = connect(r"networks/networks.db")
            crs = conn.cursor()
            crs.execute("SELECT name FROM networks WHERE name = ?", (name,))
            result = crs.fetchone()
            if result:
                crs.execute(
                    """UPDATE networks set name = ? WHERE name = ?""", (new_name, name),
                )
            else:
                with open(f"networks//{new_name}//score.json", "r") as file_score:
                    accuracy = json.load(file_score)["accuracy"]
                network_type = parameters["network_type"]
                crs.execute(
                    "INSERT INTO networks VALUES (?, ?, ?)",
                    (new_name, accuracy, network_type),
                )

            conn.commit()
            conn.close()
