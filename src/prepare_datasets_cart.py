import os
import config as cfg
from random import shuffle, seed
import csv

data_path = cfg.DATA_PATH


def process_flame2():
    flame2_path = os.path.join(data_path, "FLAME2")

    intervals = []
    with open(os.path.join(flame2_path, "Frame_Pair_Labels.txt"), "r") as f:
        text = f.read()
        labels = text.split("\n")[4:]
        for label in labels:
            if not label.strip():
                continue
            interval = label.split("\t")
            intervals.append((int(interval[0]), int(interval[1]), interval[2][0]))

    splits = {"train": [], "val": [], "test": []}

    def add_range(split_name, start, end, step, label):
        for i in range(start, end + 1, step):
            splits[split_name].append(
                (
                    os.path.join(
                        flame2_path, "254p RGB Images", f"254p RGB Frame ({i}).jpg"
                    ),
                    os.path.join(
                        flame2_path,
                        "254p Thermal Images",
                        f"254p Thermal Frame ({i}).jpg",
                    ),
                    label,
                )
            )

    # Interval 0 is No_Fire (1 - 13700):
    # Val takes the initial block (1 - 1800), followed by a 500-frame buffer (1801 - 2300),
    # and Train takes 2301 - 13700 (including the horizon/clouds frames 12500 - 13700)
    add_range("val", 1, 1800, 12, 0.0)
    add_range("train", 2301, 13700, 13, 0.0)

    # Intervals 1..18 are Fire (13701 - 53451):
    # Train takes intervals 1..15 (13701 - 45279), Val takes intervals 16..18 after a 500-frame buffer
    for idx, (start, end, _) in enumerate(intervals[1:], start=1):
        if idx <= 15:
            add_range("train", start, end, 36, 1.0)
        else:
            s = max(start, 45780)  # 500-frame buffer after interval 15 (45279)
            if s <= end:
                add_range("val", s, end, 48, 1.0)

    return splits


def process_flame3():
    flame3_path = os.path.join(data_path, "FLAME3")

    splits = {"train": [], "val": [], "test": []}

    fire_path = os.path.join(flame3_path, "Fire")
    nofire_path = os.path.join(flame3_path, "No Fire")

    fire_imgs = sorted(os.listdir(os.path.join(fire_path, "RGB", "Corrected FOV")))
    nofire_imgs = sorted(os.listdir(os.path.join(nofire_path, "RGB", "Corrected FOV")))

    # All FLAME3 images go exclusively to the Test set (100% unseen dataset)
    for img in fire_imgs:
        splits["test"].append(
            (
                os.path.join(fire_path, "RGB", "Corrected FOV", img),
                os.path.join(fire_path, "Thermal", "Raw JPG", img),
                1.0,
            )
        )

    for img in nofire_imgs:
        splits["test"].append(
            (
                os.path.join(nofire_path, "RGB", "Corrected FOV", img),
                os.path.join(nofire_path, "Thermal", "Raw JPG", img),
                0.0,
            )
        )

    return splits


def process_fireman():
    fireman_path = os.path.join(
        data_path, "FireMan-UAV-RGBT", "Binary", "Multimodal", "rgbt"
    )

    splits = {"train": [], "val": [], "test": []}
    fire_steps = {"train": 4, "val": 2, "test": 3}

    for orig_split in ["train", "val", "test"]:
        target_split = "val" if orig_split == "val" else "train"
        for cls_name, label in [("Fire", 1.0), ("No_Fire", 0.0)]:
            cls_path = os.path.join(fireman_path, orig_split, cls_name)
            step = fire_steps[orig_split] if label == 1.0 else 1

            for video_dir in sorted(os.listdir(cls_path)):
                video_path = os.path.join(cls_path, video_dir)
                if not os.path.isdir(video_path):
                    continue

                subdirs = os.listdir(video_path)
                rgb_sub = next(s for s in subdirs if s.endswith(("_rgb", "_w")))
                thermal_sub = next(s for s in subdirs if s.endswith(("_thermal", "_t")))

                rgb_dir = os.path.join(video_path, rgb_sub)
                thermal_dir = os.path.join(video_path, thermal_sub)

                rgb_imgs = sorted(os.listdir(rgb_dir))
                thermal_imgs = sorted(os.listdir(thermal_dir))

                pairs = list(zip(rgb_imgs, thermal_imgs))[::step]
                for rgb_img, thermal_img in pairs:
                    splits[target_split].append(
                        (
                            os.path.join(rgb_dir, rgb_img),
                            os.path.join(thermal_dir, thermal_img),
                            label,
                        )
                    )

    return splits


def process_3m():
    rgbt3m_path = os.path.join(data_path, "RGBT-3M")

    splits = {"train": [], "val": [], "test": []}
    val_videos = {"video5"}

    for orig_split in ["train", "test"]:
        rgb_dir = os.path.join(rgbt3m_path, "rgb", orig_split)
        thermal_dir = os.path.join(rgbt3m_path, "thermal", orig_split)

        for img_name in sorted(os.listdir(rgb_dir)):
            label = 1.0
            video_id = img_name.split("_")[0]
            target_split = "val" if video_id in val_videos else "train"

            splits[target_split].append(
                (
                    os.path.join(rgb_dir, img_name),
                    os.path.join(thermal_dir, img_name),
                    label,
                )
            )

    steps = {"train": 14, "val": 15, "test": 1}
    for k in ["train", "val"]:
        splits[k].sort(key=lambda x: os.path.basename(x[0]))
        splits[k] = splits[k][::steps[k]]

    return splits


def process_cart():
    cart_base = os.path.join(data_path, "labeled_rgbt_pairs")
    color_dir = os.path.join(cart_base, "color")
    thm8_dir = os.path.join(cart_base, "thermal8")

    all_files = sorted(os.listdir(color_dir))

    groups = {
        "forest": [],
        "canyon": [],
        "lake_water": [],
    }

    for f in all_files:
        if f.startswith(("big_bear", "caltech-coregistered-nature")):
            groups["forest"].append(f)
        elif f.startswith("2022-05-15_ColoradoRiver"):
            groups["canyon"].append(f)
        elif f.startswith(("2022-12-20_Castaic_Lake", "caltech_duck")):
            groups["lake_water"].append(f)

    # Sample ~500 pairs evenly across the 3 natural environments:
    # 165 forest + 180 canyon + 159 lake/water = 504 pairs
    sampled_files = [
        *groups["forest"][::3],
        *groups["canyon"][::2],
        *groups["lake_water"][::9],
    ]

    cart_samples = []
    for cf in sampled_files:
        tf = cf.replace("_eo-", "_thermal-")
        cart_samples.append(
            (
                os.path.join(color_dir, cf),
                os.path.join(thm8_dir, tf),
                0.0,
            )
        )

    return cart_samples


def split_and_save(flame2, flame3, fireman, _3m, cart_samples):
    train_set = [
        *flame2["train"],
        *flame3["train"],
        *fireman["train"],
        *_3m["train"],
        *cart_samples,
    ]
    val_set = [*flame2["val"], *flame3["val"], *fireman["val"], *_3m["val"]]
    test_set = [*flame2["test"], *flame3["test"], *fireman["test"], *_3m["test"]]

    seed(42)
    shuffle(train_set)

    with open(os.path.join(data_path, "train_set.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "thermal_path", "label"])
        writer.writerows(train_set)

    with open(os.path.join(data_path, "val_set.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "thermal_path", "label"])
        writer.writerows(val_set)

    with open(os.path.join(data_path, "test_set.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "thermal_path", "label"])
        writer.writerows(test_set)


def main():
    flame2_splits = process_flame2()
    flame3_splits = process_flame3()
    fireman_splits = process_fireman()
    splits_3m = process_3m()
    cart_samples = process_cart()
    split_and_save(flame2_splits, flame3_splits, fireman_splits, splits_3m, cart_samples)


if __name__ == "__main__":
    main()
