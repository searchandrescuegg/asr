from asr.audio.hash import sha256_hex


def test_deterministic():
    assert sha256_hex(b"hello") == sha256_hex(b"hello")


def test_length_64_hex():
    h = sha256_hex(b"some-bytes")
    assert len(h) == 64
    int(h, 16)


def test_distinct_inputs_distinct_hashes():
    assert sha256_hex(b"a") != sha256_hex(b"b")


def test_empty_input_has_known_hash():
    # SHA-256 of empty input is a well-known constant; locks regression.
    assert sha256_hex(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
