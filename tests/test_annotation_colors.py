from src.utils.annotation_colors import (
    PLAYER_FALLBACK_COLOR,
    resolve_player_annotation_color,
)


def test_corrected_player_uses_team_color_when_display_color_is_none():
    player = {
        "role": "player",
        "team_color": (245, 245, 245),
        "display_role": "player",
        "display_color": None,
    }

    assert resolve_player_annotation_color(player) == (245, 245, 245)


def test_player_without_any_color_uses_neutral_fallback_not_legacy_red():
    player = {
        "role": "player",
        "display_role": "player",
        "display_color": None,
    }

    assert resolve_player_annotation_color(player) == PLAYER_FALLBACK_COLOR


def test_non_goalkeeper_never_renders_goalkeeper_magenta():
    player = {
        "role": "goalkeeper",
        "team_color": (255, 0, 255),
        "display_role": "player",
        "display_color": (255, 0, 255),
    }

    assert resolve_player_annotation_color(player) == PLAYER_FALLBACK_COLOR
