from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_dataset_class_mapping(self) -> None:
        config = yaml.safe_load((ROOT / "config/data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["names"][0], "normal_person")
        self.assertEqual(config["names"][1], "potentially_dangerous_person")

    def test_recorded_variants(self) -> None:
        config = yaml.safe_load((ROOT / "config/experiments.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(config["variants"]), {
            "yolo11n",
            "yolo11s",
            "yolo11s_preprocessing",
            "yolo11m",
            "yolo11l",
        })
        self.assertEqual(config["shared"]["batch"], 8)
        self.assertEqual(config["shared"]["image_size"], 832)


if __name__ == "__main__":
    unittest.main()
