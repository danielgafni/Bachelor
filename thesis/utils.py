from .nets import LC_SNN, C_SNN, FC_SNN
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
    if not os.path.exists(f"networks//{name}"):
        print("Network with such name does not exist")
        return None
    else:
        try:
            with open(f"networks//{name}//parameters.json", "r") as file:
                parameters = json.load(file)
            return parameters
        except FileNotFoundError:
            return None


def view_database():
    database = pd.DataFrame(
        columns=[
            "name",
            "accuracy",
            "n_iter",
            "mean_weight",
            "n_filters",
            "c_w",
            "crop",
            "kernel_size",
            "stride",
            "time_max",
        ]
    )
    for name in os.listdir("networks"):
        if "." not in name:
            if os.path.exists(f"networks//{name}//parameters.json"):
                parameters = view_network(name)
                if os.path.exists(f"networks//{name}//accuracy"):
                    parameters["accuracy"] = torch.load(f"networks//{name}//accuracy")
                else:
                    parameters["accuracy"] = None
                if os.path.exists(f"networks//{name}//error"):
                    parameters["error"] = torch.load(f"networks//{name}//error")
                else:
                    parameters["error"] = None
                try:
                    parameters["name"] = name
                except TypeError:
                    pass

                database = database.append(parameters, ignore_index=True)

    return database


def plot_database(
    n_filters=100, network_type="LC_SNN", kernel_size=12, stride=4, c_l=False
):
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
                array=data["error"], visible=True, thickness=10, width=5, color="purple"
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

    return fig


def load_network(name):
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
            nu_pre = parameters["nu_pre"]
            nu_post = parameters["nu_post"]
            t_pre = parameters["t_pre"]
            t_post = parameters["t_post"]

    except FileNotFoundError:
        raise FileNotFoundError

    accuracy = None
    votes = None
    conf_matrix = None

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
            nu_pre=nu_pre,
            nu_post=nu_post,
            t_pre=t_pre,
            t_post=t_post,
            immutable_name=True,
            foldername=name,
            n_iter=n_iter,
        )
        if os.path.exists(path + "//votes"):
            votes = torch.load(path + "//votes")
            net.calibrated = True
        if os.path.exists(path + "//accuracy"):
            accuracy = torch.load(path + "//accuracy")
        if os.path.exists(path + "//confusion_matrix"):
            conf_matrix = torch.load(path + "//confusion_matrix")
        network = torch.load(path + "//network")
        net.network.connections[("X", "Y")].w = network.connections[("X", "Y")].w
        net.network.connections[("Y", "Y")].w = network.connections[("Y", "Y")].w

        net.votes = votes
        net.accuracy = accuracy
        net.conf_matrix = conf_matrix
        net.spikes = {}
        for layer in set(net.network.layers):
            net.spikes[layer] = Monitor(
                net.network.layers[layer], state_vars=["s"], time=net.time_max
            )
            net.network.add_monitor(net.spikes[layer], name="%s_spikes" % layer)
        net._spikes = {
            "X": net.spikes["X"].get("s").view(net.time_max, -1),
            "Y": net.spikes["Y"].get("s").view(net.time_max, -1),
        }

        net.network.train(False)
        net.network.connections[("Y", "Y")].learning = False
        net.network.connections[("X", "Y")].learning = False
        net.train_method = train_method
        for c in net.network.connections:
            net.network.connections[c].learning = False

    elif network_type == "C_SNN":
        net = C_SNN(
            mean_weight=mean_weight,
            c_w=c_w,
            time_max=time_max,
            crop=crop,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            intensity=intensity,
            immutable_name=True,
            foldername=name,
            n_iter=n_iter,
        )

        net.n_iter = n_iter
        if os.path.exists(path + "//votes"):
            votes = torch.load(path + "//votes")
            net.calibrated = True

        if os.path.exists(path + "//accuracy"):
            accuracy = torch.load(path + "//accuracy")

        if os.path.exists(path + "//confusion_matrix"):
            conf_matrix = torch.load(path + "//confusion_matrix")

        network = torch.load(path + "//network")
        net.network.connections[("X", "Y")].w = network.connections[("X", "Y")].w
        net.network.connections[("Y", "Y")].w = network.connections[("Y", "Y")].w
        net.votes = votes
        net.accuracy = accuracy
        net.conf_matrix = conf_matrix

        net.spikes = {}
        for layer in set(net.network.layers):
            net.spikes[layer] = Monitor(
                net.network.layers[layer], state_vars=["s"], time=net.time_max
            )
            net.network.add_monitor(net.spikes[layer], name="%s_spikes" % layer)

        net._spikes = {
            "X": net.spikes["X"].get("s").view(net.time_max, -1),
            "Y": net.spikes["Y"].get("s").view(net.time_max, -1),
        }

        net.network.train(False)

    elif network_type == "FC_SNN":
        net = FC_SNN(
            mean_weight=mean_weight,
            c_w=c_w,
            time_max=time_max,
            crop=crop,
            n_filters=n_filters,
            intensity=intensity,
            immutable_name=True,
            foldername=name,
            n_iter=n_iter,
        )

        net.n_iter = n_iter
        if os.path.exists(path + "//votes"):
            votes = torch.load(path + "//votes")
            net.calibrated = True

        if os.path.exists(path + "//accuracy"):
            accuracy = torch.load(path + "//accuracy")

        if os.path.exists(path + "//confusion_matrix"):
            conf_matrix = torch.load(path + "//confusion_matrix")

        network = torch.load(path + "//network")
        net.network.connections[("X", "Y")].w = network.connections[("X", "Y")].w
        net.network.connections[("Y", "Y")].w = network.connections[("Y", "Y")].w
        net.votes = votes
        net.accuracy = accuracy
        net.conf_matrix = conf_matrix

        net.spikes = {}
        for layer in set(net.network.layers):
            net.spikes[layer] = Monitor(
                net.network.layers[layer], state_vars=["s"], time=net.time_max
            )
            net.network.add_monitor(net.spikes[layer], name="%s_spikes" % layer)

        net._spikes = {
            "X": net.spikes["X"].get("s").view(net.time_max, -1),
            "Y": net.spikes["Y"].get("s").view(net.time_max, -1),
        }

        net.network.train(False)

    else:
        print("This network type is not implemented for loading yet")
        raise NotImplementedError

    return net


def delete_network(name, sure=False):
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
    #  Clear existing database
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
    crs.execute("DELETE FROM networks")

    for name in os.listdir("networks"):
        if "." not in name:

            #  Delete networks without saved parameters
            if not os.path.exists(f"networks//{name}//parameters.json"):
                rmtree(f"networks//{name}")
                continue

            #  Rename networks according to current parameters
            with open(f"networks//{name}//parameters.json", "r") as file:
                parameters = json.load(file)
            new_name = hashlib.sha224(str(parameters).encode("utf8")).hexdigest()
            accuracy = torch.load(f"networks//{name}//accuracy")
            network_type = parameters["network_type"]
            os.rename(f"networks//{name}", f"networks//{new_name}")

            #  Add network to database
            crs.execute(
                "INSERT INTO networks VALUES (?, ?, ?)",
                (new_name, accuracy, network_type),
            )
            conn.commit()
    conn.close()
