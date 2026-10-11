import pytest

from sports_prob_models import matching as m

ALIASES = {
    "natus vincere": "navi",
    "na'vi": "navi",
    "navi": "navi",
    "virtus.pro": "vp",
    "team liquid": "liquid",
}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("FaZe Clan", "faze"),
        ("FAZE CLAN", "faze"),
        ("Team Spirit", "spirit"),
        ("G2 Esports", "g2"),
        ("Manchester United FC", "manchesterunited"),
        ("Bad News Eagles", "badnewseagles"),
        ("Team", "team"),  # generic word kept when it is the whole name
        ("  spirit  ", "spirit"),
        ("", ""),
    ],
)
def test_normalize_without_aliases(raw, expected):
    assert m.normalize_name(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Natus Vincere", "navi"),
        ("NAVI", "navi"),
        ("Na'Vi", "navi"),
        ("Virtus.pro", "vp"),
        ("Natus Vincere CS2", "navi"),  # whole-word containment
        ("Team Liquid", "liquid"),
        ("Liquid", "liquid"),  # generic-word stripping not needed, no alias -> plain slug
    ],
)
def test_normalize_with_aliases(raw, expected):
    assert m.normalize_name(raw, ALIASES) == expected


def test_alias_containment_is_whole_word():
    aliases = {"og": "og"}
    assert m.normalize_name("OG", aliases) == "og"
    assert m.normalize_name("Dogma", aliases) == "dogma"


def test_match_signature_is_order_independent():
    assert m.match_signature("FaZe Clan", "Natus Vincere", ALIASES) == "faze__navi"
    assert m.match_signature("NaVi", "faze", ALIASES) == "faze__navi"
    assert m.match_signature("Arsenal", "Chelsea") == m.match_signature("Chelsea FC", "Arsenal FC")
    with pytest.raises(ValueError):
        m.match_signature("", "faze")


def test_fuzzy_match():
    cands = ["faze", "navi", "liquid", "vitality"]
    assert m.fuzzy_match("faze", cands) == "faze"
    assert m.fuzzy_match("FaZe Clan", cands) == "faze"
    assert m.fuzzy_match("Fase", cands) == "faze"
    assert m.fuzzy_match("Liquidd", cands) == "liquid"
    assert m.fuzzy_match("Vitalty", cands) == "vitality"
    assert m.fuzzy_match("Natus Vincere", cands, aliases=ALIASES) == "navi"
    assert m.fuzzy_match("Astralis", cands) is None
    assert m.fuzzy_match("", cands) is None
    assert m.fuzzy_match("faze", []) is None


def test_edit_distance():
    assert m.edit_distance("kitten", "sitting") == 3
    assert m.edit_distance("", "abc") == 3
    assert m.edit_distance("abc", "abc") == 0


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Will FaZe beat NaVi in Map 1?", ("FaZe", "NaVi")),
        ("Will Team Vitality defeat G2 Esports?", ("Team Vitality", "G2 Esports")),
        ("FaZe vs NaVi - Map 1 Winner", ("FaZe", "NaVi")),
        ("FaZe vs. NaVi Map 1 Winner", ("FaZe", "NaVi")),
        ("Map 1: FaZe vs NaVi", ("FaZe", "NaVi")),
        ("CS2: FaZe Clan vs Natus Vincere - Map 1", ("FaZe Clan", "Natus Vincere")),
        ("Who wins Map 1: FaZe vs Natus Vincere?", ("FaZe", "Natus Vincere")),
        ("Team Vitality vs G2 Esports", ("Team Vitality", "G2 Esports")),
        ("Heroic vs NAVI - Winner", ("Heroic", "NAVI")),
        ("Arsenal against Chelsea - Match Winner", ("Arsenal", "Chelsea")),
        ("Will Arsenal win the league?", None),
        ("", None),
    ],
)
def test_extract_teams_from_question(question, expected):
    assert m.extract_teams_from_question(question) == expected


def test_extracted_names_pair_up_across_sources():
    a = m.extract_teams_from_question("Will FaZe beat Natus Vincere in Map 1?")
    b = m.extract_teams_from_question("NAVI vs FaZe Clan - Map 1")
    assert a and b
    assert m.match_signature(*a, aliases=ALIASES) == m.match_signature(*b, aliases=ALIASES)
