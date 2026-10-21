import os
import config as cfg
from random import shuffle, seed
import csv
from PIL import Image
from PIL.ExifTags import TAGS

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

    # Interval 0 is No_Fire (1 - 13000, excluding ground/landing tail 13001-13700):
    # 3 contiguous macro-blocks separated by 500-frame buffers
    add_range("train", 1, 8800, 12, 0.0)
    add_range("val", 9301, 10900, 11, 0.0)
    add_range("test", 11401, 13000, 11, 0.0)

    # Intervals 1..18 are Fire (13701 - 53451):
    # 3 contiguous macro-blocks separated by 500-frame buffers at split boundaries
    for idx, (start, end, _) in enumerate(intervals[1:], start=1):
        if idx <= 13:
            add_range("train", start, end, 33, 1.0)
        elif idx <= 15:
            s = max(start, 38654)  # 500-frame buffer after interval 13 (38153)
            if s <= end:
                add_range("val", s, end, 42, 1.0)
        else:
            s = max(start, 45780)  # 500-frame buffer after interval 15 (45279)
            if s <= end:
                add_range("test", s, end, 48, 1.0)

    return splits


def _get_exif_datetime(img_path):
    exif = Image.open(img_path)._getexif() or {}
    for k, v in exif.items():
        if TAGS.get(k, k) == "DateTimeOriginal":
            return str(v)
    return ""


def process_flame3():
    flame3_path = os.path.join(data_path, "FLAME3")

    splits = {"train": [], "val": [], "test": []}

    fire_path = os.path.join(flame3_path, "Fire")
    nofire_path = os.path.join(flame3_path, "No Fire")

    # Sort by true EXIF capture timestamp (since filenames in FLAME3 were shuffled)
    fire_imgs = sorted(
        os.listdir(os.path.join(fire_path, "RGB", "Corrected FOV")),
        key=lambda img: (
            _get_exif_datetime(os.path.join(fire_path, "RGB", "Corrected FOV", img)),
            img,
        ),
    )
    nofire_imgs = sorted(
        os.listdir(os.path.join(nofire_path, "RGB", "Corrected FOV")),
        key=lambda img: (
            _get_exif_datetime(os.path.join(nofire_path, "RGB", "Corrected FOV", img)),
            img,
        ),
    )

    # Split Fire by separate flight sessions (>1 hour apart):
    # Train: 2022:10:25, 2022:10:26, and 2022:10:27 17:xx (396 imgs)
    # Val  : 2022:10:27 15:xx flight (116 imgs)
    # Test : 2022:10:27 19:xx flight (110 imgs)
    for img in fire_imgs:
        rgb_p = os.path.join(fire_path, "RGB", "Corrected FOV", img)
        thm_p = os.path.join(fire_path, "Thermal", "Raw JPG", img)
        dt = _get_exif_datetime(rgb_p)
        if dt.startswith("2022:10:27 15:"):
            splits["val"].append((rgb_p, thm_p, 1.0))
        elif dt.startswith("2022:10:27 19:"):
            splits["test"].append((rgb_p, thm_p, 1.0))
        else:
            splits["train"].append((rgb_p, thm_p, 1.0))

    # Split chronologically ordered No_Fire with buffer gaps
    n = len(nofire_imgs)
    n_train = int(n * 0.68)
    n_val_start = int(n * 0.72)
    n_val_end = int(n * 0.84)
    n_test_start = int(n * 0.88)

    for split_name, subset in [
        ("train", nofire_imgs[:n_train]),
        ("val", nofire_imgs[n_val_start:n_val_end]),
        ("test", nofire_imgs[n_test_start:]),
    ]:
        for img in subset:
            splits[split_name].append(
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

    for split in ["train", "val", "test"]:
        for cls_name, label in [("Fire", 1.0), ("No_Fire", 0.0)]:
            cls_path = os.path.join(fireman_path, split, cls_name)
            step = fire_steps[split] if label == 1.0 else 1

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
                    splits[split].append(
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
    test_videos = {"video4", "video9"}
    val_videos = {"video5"}

    for orig_split in ["train", "test"]:
        rgb_dir = os.path.join(rgbt3m_path, "rgb", orig_split)
        thermal_dir = os.path.join(rgbt3m_path, "thermal", orig_split)

        for img_name in sorted(os.listdir(rgb_dir)):
            # Both fire (1) and smoke (0) in RGBT-3M are active wildfire frames
            label = 1.0

            video_id = img_name.split("_")[0]
            if video_id in test_videos:
                target_split = "test"
            elif video_id in val_videos:
                target_split = "val"
            else:
                target_split = "train"

            splits[target_split].append(
                (
                    os.path.join(rgb_dir, img_name),
                    os.path.join(thermal_dir, img_name),
                    label,
                )
            )

    steps = {"train": 12, "val": 15, "test": 15}
    for k in splits:
        splits[k].sort(key=lambda x: os.path.basename(x[0]))
        splits[k] = splits[k][::steps[k]]

    return splits


def split_and_save(flame2, flame3, fireman, _3m):
    train_set = [*flame2["train"], *flame3["train"], *fireman["train"], *_3m["train"]]
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
    flame2_train = process_flame2()
    flame3_train = process_flame3()
    fireman_split = process_fireman()
    split_3m = process_3m()
    split_and_save(flame2_train, flame3_train, fireman_split, split_3m)


if __name__ == "__main__":
    main()
