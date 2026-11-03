"""Properties of the feature distances (needs torch; notebook 07 runs the same checks before the sweep)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rftm.features import crop_valid, feature_distances, letterbox_valid_box, linear_cka  # noqa: E402


def test_identical_maps_have_zero_distance():
    a = torch.randn(1, 32, 20, 30)
    d = feature_distances(a, a.clone())
    assert abs(d["cos"]) < 1e-9 and abs(d["cka"]) < 1e-9 and d["rel_l2"] == 0 and abs(d["norm_ratio"] - 1) < 1e-12


def test_cka_invariances_and_scale_sensitivity():
    torch.manual_seed(0)
    a = torch.randn(1, 16, 24, 24)
    q, _ = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))
    rotated = torch.einsum("dc,bchw->bdhw", q.float(), a) * 3.0          # rotate channels, scale by 3
    assert abs(linear_cka(a, rotated) - 1) < 1e-5                        # CKA ignores both
    d = feature_distances(a, 3.0 * a)                                    # 3.0 * a is float32: ~1e-7 rounding
    assert abs(d["cos"]) < 1e-6 and abs(d["rel_l2"] - 4.0) < 1e-5       # cosine ignores scale, rel_l2 does not
    assert linear_cka(a, torch.randn(1, 16, 24, 24)) < 0.2               # unrelated maps: low CKA


def test_letterbox_crop():
    # 427 x 640 COCO image -> r = 1, padded to 448 x 640, 10 rows of padding on top, 11 below
    t, b, l, r = letterbox_valid_box(427, 640, 448, 640)
    assert (t, b, l, r) == (10, 437, 0, 640)
    f = torch.zeros(1, 4, 448 // 8, 640 // 8)                            # a stride-8 map
    c = crop_valid(f, (t, b, l, r), 448, 640)
    assert c.shape[2] == 54 and c.shape[3] == 80                         # rows 1..54 of 56 overlap pixels 10..436


if __name__ == "__main__":
    test_identical_maps_have_zero_distance(); test_cka_invariances_and_scale_sensitivity(); test_letterbox_crop()
    print("all feature tests passed")
