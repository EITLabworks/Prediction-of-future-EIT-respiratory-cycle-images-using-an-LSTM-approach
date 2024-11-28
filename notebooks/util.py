import json
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import imageio
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit import mesh
from pyeit.eit.fem import EITForward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from sciopy.sciopy_dataclasses import ScioSpecMeasurementSetup
from tqdm import tqdm
from IPython.display import Image, display


def load_data(data_set: int, mat_complex=False, info=True):
    timestamp = list()
    perm_array = list()
    d = list()
    eit = list()
    for ele in np.sort(glob(f"../data/#{data_set}/DATA/*.npz")):
        tmp = np.load(ele, allow_pickle=True)
        timestamp.append(tmp["timestamp"])
        perm_array.append(tmp["perm_arr"])
        d.append(tmp["d"])
        eit.append(tmp["eit"])
    timestamp = np.array(timestamp)
    perm_array = np.array(perm_array)
    d = np.array(d)
    eit = np.array(eit)
    eit = z_score_normalization(eit)
    if not mat_complex:
        eit = np.abs(eit)
    if info:
        # Load and display the image
        print(
            f"Time of\n\tfirst: {convert_timestamp(min(timestamp))}\n\tlast: {convert_timestamp(max(timestamp))}\nmeasurement."
        )
        print(f"Shape of EIT data: {eit.shape}")
        print(f"Shape of Permittivity Data: {perm_array.shape}")
        img = Image(filename=f"../data/#{data_set}/PCA.png")
        display(img)
    return eit, perm_array, d, timestamp


def z_score_normalization(data, axis=(1, 2)):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / std


def convert_timestamp(date_str):
    if len(str(date_str).split(".")) > 2:
        timestamp = datetime.strptime(date_str, "%Y.%m.%d. %H:%M:%S.%f")
        return timestamp.timestamp()
    else:
        date_time = datetime.fromtimestamp(float(date_str))
        return date_time.strftime("%Y.%m.%d. %H:%M:%S.%f")


def get_fps(timestamps):
    diff = np.diff(timestamps)
    fps = 1 / diff
    print(f"Mean fps of {np.mean(fps):.2f}")


# Function to ensure that a particular directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_empty_mesh(
    mesh_obj, perm_array, ax=None, title="Empty Mesh", sample_index=None
):
    el_pos = np.arange(mesh_obj.n_el)
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    vmin, vmax = 0, 10

    im = ax.tripcolor(
        x,
        y,
        tri,
        perm_array,
        shading="flat",
        edgecolor="k",
        alpha=0.8,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    # Annotate each element with its index
    for j, el in enumerate(el_pos):
        ax.text(pts[el, 0], pts[el, 1], str(j + 1), color="red", fontsize=8)

    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    if sample_index is not None:
        ax.set_title(f"{title} Sample {sample_index}")
    else:
        ax.set_title(title)

    # Create colorbar with limits
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Permittivity")
    cbar.ax.tick_params(labelsize=8)

    if ax is None:
        plt.show()


# Function to plot mesh
def plot_mesh_permarray(mesh_obj, perm_array, ax=None, title="Mesh", sample_index=None):
    el_pos = np.arange(mesh_obj.n_el)
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.tripcolor(x, y, tri, perm_array, shading="flat", edgecolor="k", alpha=0.8)

    # Annotate each element with its index
    for j, el in enumerate(el_pos):
        ax.text(pts[el, 0], pts[el, 1], str(j + 1), color="red")

    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    if sample_index is not None:
        ax.set_title(f"{title} Sample {sample_index}")
    else:
        ax.set_title(title)
    if ax is None:
        fig.colorbar(im, ax=ax)
        plt.show()
    else:
        plt.colorbar(im, ax=ax)


# Functions to compute FEM deviations for box plot
def compute_perm_deviation(
    mesh_obj,
    perm_true: np.ndarray,
    perm_pred: np.ndarray,
    obj_threshold: Union[int, float],
    plot: bool = False,
) -> int:
    # Identify object indices based on threshold
    obj_idx_true = np.where(perm_true > obj_threshold)[0]
    obj_idx_pred = np.where(perm_pred > obj_threshold)[0]

    perm_dev = len(obj_idx_pred) - len(obj_idx_true)

    return perm_dev


def calculate_perm_error(X_true, X_pred):
    perm_error = list()
    obj_threshold = (np.max(X_true) - np.min(X_true)) / 2
    mesh_obj = mesh.create(n_el=32, h0=0.05)

    for perm_true, perm_pred in zip(X_true, X_pred):
        perm_error.append(
            compute_perm_deviation(
                mesh_obj, perm_true, perm_pred, obj_threshold, plot=False
            )
        )
    perm_error = np.array(perm_error)

    return perm_error


# Function to select a number of random instances for mesh plots comparison
def select_random_instances(x_test, y_test, predicted_permittivities, num_instances=10):
    random_indices = random.sample(range(x_test.shape[0]), num_instances)
    selected_true_perms = y_test[random_indices]
    selected_predicted_perms = predicted_permittivities[random_indices]
    return random_indices, selected_true_perms, selected_predicted_perms


# Function to create a mesh instance
def create_mesh(n_el=32, h0=0.05):
    return mesh.create(n_el, h0=h0)


# Function to create box plot
def plot_boxplot(
    data, ylabel, title, savefig_name, save_dir="plots", figsize=(6, 8), dpi=300
):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=figsize)
    plt.boxplot(data)
    plt.ylabel(ylabel)
    plt.title(title)
    save_path = os.path.join(save_dir, savefig_name)
    plt.savefig(save_path, format="png", dpi=dpi)
    plt.show()


def seq_data(eit, perm, n_seg=4):
    sequence = [eit[i : i + n_seg] for i in range(len(eit) - n_seg)]
    aligned_perm = perm[n_seg:]
    return np.array(sequence), np.array(aligned_perm)


# Function to add noise to a signal
def add_noise(signal, snr):
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise


def time_to_seconds(time_str):
    """
    String of "hh:mm:ss" in hours, minutes and seconds represent the measurement duration.
    For example "02:30:00" means that the measurement will take 2 hours and 30 minutes.
    """
    h, m, s = map(int, time_str.split(":"))

    # Gesamtzeit in Sekunden berechnen
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds


def extract_timestamps(path: str):
    arduino_path = f"{path}arduino/*.npz"
    sciospec_path = f"{path}sciospec/*.npz"

    arduino_files = np.sort(glob(arduino_path))
    sciospec_files = np.sort(glob(sciospec_path))
    print(
        f"In total we have\n\t{len(arduino_files)} arduino\n\t{len(sciospec_files)} sciospec\nfiles."
    )

    arduino_times = np.array(
        [np.load(ele, allow_pickle=True)["timestamp"] for ele in arduino_files]
    )
    sciospec_times = np.array(
        [np.load(ele, allow_pickle=True)["timestamp"] for ele in sciospec_files]
    )

    assert len(arduino_times) == len(arduino_files)
    assert len(sciospec_times) == len(sciospec_files)
    return arduino_times, sciospec_times


def samples_per_second(timestamps_arr):
    """
    Compute the frequency in Hz of an array of timestamps.
    """
    time_diffs = np.diff(timestamps_arr)
    avg_time_diff = np.mean(time_diffs)
    if avg_time_diff > 0:
        samples_per_sec = 1 / avg_time_diff
    else:
        raise ValueError("Can´t compute frequency.")
    return samples_per_sec


def plot_aligned_timestamps(arduino_times, sciospec_times):
    start_time = max(np.min(arduino_times), np.min(sciospec_times))
    stop_time = min(np.max(arduino_times), np.max(sciospec_times))

    x1 = np.linspace(start_time, stop_time, len(arduino_times))
    x2 = np.linspace(start_time, stop_time, len(sciospec_times))

    sps_arduino = samples_per_second(arduino_times)
    sps_sciospec = samples_per_second(sciospec_times)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].scatter(x1, arduino_times, marker="x")
    axs[0, 0].set_title("Arduino")
    axs[0, 0].set_xlabel("Timestamp (common range)")
    axs[0, 0].set_ylabel("Timestamps")

    axs[0, 1].scatter(x2, sciospec_times, marker="x", color="C1")
    axs[0, 1].set_title("Sciospec")
    axs[0, 1].set_xlabel("Timestamp (common range)")
    axs[0, 1].set_ylabel("Timestamps")

    axs[1, 0].scatter(x1, arduino_times, marker="x", label="Sciospec")
    axs[1, 0].scatter(x2, sciospec_times, marker="x", label="Arduino")
    axs[1, 0].legend()
    axs[1, 0].set_title("Arduino and Sciospec")
    axs[1, 0].set_xlabel("Timestamp (common range)")
    axs[1, 0].set_ylabel("Timestamps")

    axs[1, 1].set_title("Information")
    axs[1, 1].text(
        0.4, 0.65, "Measurements per second:", fontsize=12, ha="center", va="center"
    )
    axs[1, 1].text(
        0.3, 0.55, f"Arduino: {sps_arduino:.2f} Hz", fontsize=12, ha="left", va="center"
    )
    axs[1, 1].text(
        0.3,
        0.45,
        f"Sciospec: {sps_sciospec:.2f} Hz",
        fontsize=12,
        ha="left",
        va="center",
    )

    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    plt.tight_layout()
    plt.show()


def timings() -> Tuple[float, str]:
    """
    timings generates a timestamp and a human readable timestamp.

    Returns
    -------
    Tuple[float, str]
        compute timestamp, human timestamp
    """
    now = datetime.now()
    stamp = now.timestamp()
    human = datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S,%f")
    return stamp, human


def arduino_callback(serial, msgs=False) -> Tuple[float, float, str, float]:
    """
    Sends the ASCII symbol "g" for "get" to the Arduino R4 to receive the ultrasonic sensor's distance callback.

    Parameters
    ----------
    serial : serial connection
        serial connection to the Arduino R4 Minima
    msgs : bool, optional
        print callback information, by default False

    Returns
    -------
    Tuple[float, float,str,float]
        _description_
    """
    t_stamp_start, t_human_start = timings()
    p_start = time.time()

    try:
        serial.write(b"g")
        callback = float(serial.readline().decode("ascii"))
    except BaseException:
        print("Connection error...")
        return None
    p_stop = time.time()
    # t_stamp_stop, t_human_stop = timings()

    uncernity = (p_stop - p_start) * 1e3
    if msgs:
        print(f"Got callback at {t_human_start} from arduino.")
        print(f"\t d = {callback:.2f} cm")
        print(f"\t Δt = {uncernity:.5f} ms")
    return callback, t_stamp_start, t_human_start, uncernity


# Function to create mesh plots and save them for comparison
def mesh_plot_comparisons(
    mesh_obj,
    selected_indices,
    selected_true_perms,
    selected_predicted_perms,
    save_dir="comparison_plots",
    gif_name="comparison.gif",
    gif_title="Mesh Comparison",
    fps=1,
):
    os.makedirs(save_dir, exist_ok=True)

    images = []

    for i in range(len(selected_indices)):
        true_perm = selected_true_perms[i].flatten()
        pred_perm = selected_predicted_perms[i].flatten()

        assert len(true_perm) == len(
            mesh_obj.element
        ), f"Length of true_perm ({len(true_perm)}) does not match number of elements ({len(mesh_obj.element)})"
        assert len(pred_perm) == len(
            mesh_obj.element
        ), f"Length of pred_perm ({len(pred_perm)}) does not match number of elements ({len(mesh_obj.element)})"

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(gif_title, fontsize=16)

        plot_mesh_permarray(
            mesh_obj,
            true_perm,
            ax=axs[0],
            title="Original",
            sample_index=selected_indices[i],
        )
        plot_mesh_permarray(
            mesh_obj,
            pred_perm,
            ax=axs[1],
            title="Predicted",
            sample_index=selected_indices[i],
        )

        filename = os.path.join(save_dir, f"comparison_{i + 1}.png")
        plt.savefig(filename, format="png", dpi=300)
        plt.savefig(filename + ".pdf")
        plt.show()

        images.append(imageio.imread(filename))
        plt.close(fig)

    gif_path = os.path.join(save_dir, gif_name)
    duration_per_frame = 1000 / fps
    imageio.mimsave(gif_path, images, duration=duration_per_frame, loop=0)

    png_dats = glob(os.path.join(save_dir, "*.png"))
    for dat in png_dats:
        os.remove(dat)
