from chroma.__main__ import selbsttest


def test_selbsttest(tmp_path):
    selbsttest(tmp_path)
    assert (tmp_path / "haupt" / "bericht.md").exists()
    assert (tmp_path / "haupt" / "register.md").exists()
    assert (tmp_path / "vorstudie" / "bericht.md").exists()
