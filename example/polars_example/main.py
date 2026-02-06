import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from robokots.kots import Kots
from robokots.core.state_json import iter_jsonl_rows, write_jsonl


def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    kots.set_order(3)

    n_steps = 5
    dt = 0.01
    states = []
    times = []
    for step in range(n_steps):
        motion = np.random.rand(kots.order() * kots.dof())
        kots.import_motions(motion)
        kots.kinematics()
        states.append(kots.state_dict_)
        times.append(step * dt)

    out_path = "out_state.jsonl"
    rows = iter_jsonl_rows(
        states,
        times=times,
        meta={"robot_id": "sample"},
        schema_version=1,
    )
    write_jsonl(out_path, rows)

    df = pl.read_ndjson(out_path)
    if "t" in df.columns:
        df = df.select(["t"] + [c for c in df.columns if c != "t"])
    print(df)

    # 末端リンクの位置・速度の時間変化をプロット
    t = df["t"].to_numpy()
    pos = np.array(df["end_link_pos"].to_list())  # (N, 3)
    vel6 = np.array(df["end_link_vel"].to_list())  # (N, 6) = [wx, wy, wz, vx, vy, vz]
    vel = vel6[:, 3:6]  # 並進速度成分のみを使用

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t, pos[:, 0], label="x")
    axes[0].plot(t, pos[:, 1], label="y")
    axes[0].plot(t, pos[:, 2], label="z")
    axes[0].set_ylabel("position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, vel[:, 0], label="vx")
    axes[1].plot(t, vel[:, 1], label="vy")
    axes[1].plot(t, vel[:, 2], label="vz")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("velocity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    backend = plt.get_backend().lower()
    if "agg" in backend:
        out_fig = "end_link_pos_vel.png"
        plt.savefig(out_fig, dpi=150)
        print(f"saved plot: {out_fig}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
