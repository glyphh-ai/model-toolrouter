"""Test NL query attribute extraction."""

from encoder import encode_query


def test_verb_extraction_build():
    r = encode_query("create a release branch")
    assert r["attributes"]["verb"] == "build"


def test_verb_extraction_deploy():
    r = encode_query("release the SDK")
    assert r["attributes"]["verb"] == "deploy"


def test_verb_extraction_query():
    r = encode_query("check the build status")
    assert r["attributes"]["verb"] == "query"


def test_verb_extraction_manage():
    r = encode_query("delete the release branch")
    assert r["attributes"]["verb"] == "manage"


def test_verb_extraction_restart():
    r = encode_query("restart the runtime")
    assert r["attributes"]["verb"] == "manage"


def test_object_extraction_branch():
    r = encode_query("create a release branch")
    assert r["attributes"]["object"] == "branch"


def test_object_extraction_sdk():
    r = encode_query("ship the SDK")
    assert r["attributes"]["object"] == "sdk"


def test_object_extraction_runtime():
    r = encode_query("rebuild the runtime container")
    assert r["attributes"]["object"] == "runtime"


def test_domain_release():
    r = encode_query("merge the release branch")
    assert r["attributes"]["domain"] == "release"


def test_domain_test():
    r = encode_query("run the pytest suite")
    assert r["attributes"]["domain"] == "test"


def test_domain_docker():
    r = encode_query("rebuild the docker containers")
    assert r["attributes"]["domain"] == "docker"


def test_domain_build():
    r = encode_query("check the CI workflow")
    assert r["attributes"]["domain"] == "build"


def test_keywords_exclude_stop_words():
    r = encode_query("please run the tests for me")
    kw = r["attributes"]["keywords"]
    assert "please" not in kw
    assert "the" not in kw
    assert "for" not in kw
    assert "me" not in kw
    assert "run" in kw
    assert "tests" in kw
