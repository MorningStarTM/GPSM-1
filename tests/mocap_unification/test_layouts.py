import pytest

from src.gpsm.smplx.layouts import detect_layout, get_layout, list_layouts
from src.gpsm.smplx.layouts.correspondence import smplx_joint_index


def test_all_builtin_layouts_registered():
    names = list_layouts()
    for expected in ("coco_17", "openpose_25", "mediapipe_33", "h36m_17", "vicon_50"):
        assert expected in names


def test_layout_correspondences_resolve_to_valid_indices():
    for name in list_layouts():
        layout = get_layout(name)
        indices = layout.target_indices()
        assert len(indices) == layout.num_points
        assert all(isinstance(i, int) and i >= 0 for i in indices)


def test_joint_only_layouts_support_legacy_smplx_indices():
    for name in ("coco_17", "openpose_25", "mediapipe_33", "h36m_17"):
        layout = get_layout(name)
        assert layout.is_joint_only
        assert layout.smplx_indices() == layout.target_indices()


def test_vicon_50_is_vertex_only_with_real_vertex_ids():
    from src.gpsm.smplx.layouts.builtin import VICON_50_VERTEX_IDS
    from src.gpsm.smplx.layouts.correspondence import SMPLX_NUM_VERTICES

    layout = get_layout("vicon_50")
    assert layout.num_points == 50
    assert not layout.is_joint_only
    assert all(k == "vertex" for k in layout.target_kinds())
    assert all(0 <= i < SMPLX_NUM_VERTICES for i in layout.target_indices())
    # Spot-check a couple of the real extracted marker->vertex ids.
    assert VICON_50_VERTEX_IDS["C7"] == 6607
    assert VICON_50_VERTEX_IDS["LFHD"] == 1974

    import pytest
    with pytest.raises(ValueError):
        layout.smplx_indices()  # not joint-only; legacy accessor must refuse


def test_unknown_joint_name_raises_with_suggestions():
    with pytest.raises(KeyError):
        smplx_joint_index("not_a_real_joint")


def test_detect_layout_unambiguous():
    assert detect_layout(25).name == "openpose_25"
    assert detect_layout(33).name == "mediapipe_33"


def test_detect_layout_ambiguous_raises():
    # coco_17 and h36m_17 both have 17 points.
    with pytest.raises(ValueError):
        detect_layout(17)


def test_detect_layout_unknown_count_raises():
    with pytest.raises(ValueError):
        detect_layout(9999)


def test_mediapipe_landmarks_carry_reduced_weight():
    layout = get_layout("mediapipe_33")
    assert all(w <= 0.5 for w in layout.weights())


def test_openpose_landmarks_carry_full_weight():
    layout = get_layout("openpose_25")
    assert all(w == 1.0 for w in layout.weights())
