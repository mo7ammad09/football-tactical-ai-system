import numpy as np

from src.field_filter import FieldFilter


def test_field_filter_keeps_foot_point_inside_pitch():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[20:100, 20:140] = (0, 140, 0)
    field_filter = FieldFilter(
        {
            "kernel_erode_size": 3,
            "kernel_dilate_size": 3,
            "kernel_close_size": 5,
            "morph_iterations": 1,
            "min_area_ratio": 0.01,
            "hsv_upper": [95, 255, 200],
        }
    )

    items = [
        {"xyxy": [40, 40, 60, 80], "id": "inside"},
        {"xyxy": [145, 5, 155, 18], "id": "outside"},
    ]

    filtered = field_filter.filter_track_items(frame, items)

    assert [item["id"] for item in filtered] == ["inside"]
