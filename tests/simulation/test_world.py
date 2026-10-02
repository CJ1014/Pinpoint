"""The simulation environment must model real tool semantics.

If the world is wrong, every scenario built on it is worthless — so the world
gets its own tests before anything relies on it.
"""

from tests.simulation.world import ERROR, PARTIAL, SILENT, STALE, TIMEOUT


# ── Filesystem ────────────────────────────────────────────────────────────────

def test_write_actually_creates_a_real_file(world):
    world.dispatch("write_file", {"filename": "a.txt", "content": "hello"})
    assert world.exists("a.txt") and world.read("a.txt") == "hello"


def test_write_reports_the_real_path(world):
    result = world.dispatch("write_file", {"filename": "a.txt", "content": "hello"})
    assert world.path("a.txt") in result


def test_read_of_a_missing_file_fails_like_the_real_tool(world):
    assert "File not found" in world.dispatch("read_file", {"filename": "nope.txt"})


def test_delete_removes_the_file(world):
    world.dispatch("write_file", {"filename": "a.txt", "content": "x"})
    world.dispatch("delete_file", {"path": "a.txt"})
    assert not world.exists("a.txt")


def test_delete_of_a_missing_file_errors(world):
    assert "no such file" in world.dispatch("delete_file", {"path": "ghost.txt"})


# ── Script semantics ──────────────────────────────────────────────────────────

def test_a_clean_script_exits_zero(world):
    world.write("ok.py", "print('hi')")
    assert "exit code: 0" in world.dispatch("run_python", {"filename": "ok.py"})


def test_script_output_is_modelled(world):
    world.write("ok.py", "print('hello world')")
    assert "hello world" in world.dispatch("run_python", {"filename": "ok.py"})


def test_missing_import_produces_a_real_traceback(world):
    world.write("app.py", "import flask\nprint('x')")
    result = world.dispatch("run_python", {"filename": "app.py"})
    assert "ModuleNotFoundError" in result and "flask" in result
    assert "exit code: 1" in result


def test_installing_the_module_fixes_the_script(world):
    world.write("app.py", "import flask\nprint('x')")
    world.dispatch("pip_install", {"package": "flask"})
    assert "exit code: 0" in world.dispatch("run_python", {"filename": "app.py"})


def test_a_failing_assertion_exits_nonzero(world):
    world.write("t.py", "assert False")
    assert "exit code: 1" in world.dispatch("run_python", {"filename": "t.py"})


def test_running_a_missing_script_fails(world):
    assert "No such file" in world.dispatch("run_python", {"filename": "ghost.py"})


# ── Shell ─────────────────────────────────────────────────────────────────────

def test_shell_runs_python_files(world):
    world.write("ok.py", "print('via shell')")
    assert "via shell" in world.dispatch("run_shell", {"command": "python ok.py"})


def test_shell_pip_install_registers_the_module(world):
    world.dispatch("run_shell", {"command": "pip install requests"})
    assert "requests" in world.installed_modules


def test_unknown_command_is_not_found(world):
    result = world.dispatch("run_shell", {"command": "frobnicate --hard"})
    assert "command not found" in result and "exit code: 127" in result


def test_cat_of_a_missing_file_fails(world):
    assert "No such file" in world.dispatch("run_shell", {"command": "cat ghost.txt"})


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_tests_reports_none_found(world):
    assert "no tests found" in world.dispatch("run_tests", {})


def test_passing_tests_exit_zero(world):
    world.write("test_a.py", "print('ok')")
    assert "exit code: 0" in world.dispatch("run_tests", {})


def test_failing_tests_exit_nonzero(world):
    world.write("test_a.py", "assert False")
    result = world.dispatch("run_tests", {})
    assert "exit code: 1" in result and "1 failed" in result


def test_fixing_the_test_makes_it_pass(world):
    world.write("test_a.py", "assert False")
    assert "exit code: 1" in world.dispatch("run_tests", {})
    world.write("test_a.py", "print('fixed')")
    assert "exit code: 0" in world.dispatch("run_tests", {})


# ── Network and ports ─────────────────────────────────────────────────────────

def test_unknown_url_404s(world):
    assert "404" in world.dispatch("fetch_url", {"url": "https://nope.example"})


def test_served_url_returns_its_body(world):
    world.serve("https://docs.example", body="the documentation")
    assert "documentation" in world.dispatch("fetch_url", {"url": "https://docs.example"})


def test_server_binds_a_real_port(world):
    import socket
    result = world.dispatch("start_server", {"port": 0})
    port = int(result.split("port ")[1].split()[0])
    with socket.create_connection(("127.0.0.1", port), timeout=1):
        pass       # a real connection to a real listener


# ── Failure injection ─────────────────────────────────────────────────────────

def test_injected_error_replaces_the_result(world):
    world.inject("run_python", ERROR, "Error: interpreter exploded")
    assert "exploded" in world.dispatch("run_python", {"filename": "a.py"})


def test_injection_expires_after_its_count(world):
    world.write("ok.py", "print('x')")
    world.inject("run_python", ERROR, "Error: transient", times=1)
    assert "transient" in world.dispatch("run_python", {"filename": "ok.py"})
    assert "exit code: 0" in world.dispatch("run_python", {"filename": "ok.py"})


def test_silent_failure_claims_success_and_does_nothing(world):
    world.inject("write_file", SILENT)
    result = world.dispatch("write_file", {"filename": "a.txt", "content": "hello"})
    assert "Written 5 chars" in result       # the tool says it worked
    assert not world.exists("a.txt")         # the world says otherwise


def test_partial_failure_writes_half_the_content(world):
    world.inject("write_file", PARTIAL)
    world.dispatch("write_file", {"filename": "a.txt", "content": "abcdefgh"})
    assert 0 < len(world.read("a.txt")) < 8


def test_stale_state_removes_the_file_afterwards(world):
    world.inject("write_file", STALE)
    world.dispatch("write_file", {"filename": "a.txt", "content": "hello"})
    assert not world.exists("a.txt")


def test_timeout_injection_delays_the_call(world):
    import time
    world.inject("run_python", TIMEOUT, delay=0.2)
    started = time.perf_counter()
    world.dispatch("run_python", {"filename": "a.py"})
    assert time.perf_counter() - started >= 0.2


def test_injection_can_target_specific_parameters(world):
    world.write("good.py", "print('fine')")
    world.write("bad.py", "print('fine')")
    world.inject("run_python", ERROR, "Error: only this one", contains="bad.py")
    assert "exit code: 0" in world.dispatch("run_python", {"filename": "good.py"})
    assert "only this one" in world.dispatch("run_python", {"filename": "bad.py"})


# ── Providers ─────────────────────────────────────────────────────────────────

def test_provider_confirms_by_default(world):
    from pinpoint.communication import contacts as C
    C.contacts().add(C.Contact(name="Sarah", phone="+15550001111"))
    world.install_providers()
    assert "confirmation_id" in world.dispatch("send_message",
                                               {"to": "Sarah", "body": "hi"})


def test_provider_can_withhold_confirmation(world):
    from pinpoint.communication import contacts as C
    C.contacts().add(C.Contact(name="Sarah", phone="+15550001111"))
    world.install_providers(confirm=False)
    result = world.dispatch("send_message", {"to": "Sarah", "body": "hi"})
    assert "confirmation_id" not in result


def test_unconfigured_provider_is_unavailable(world):
    from pinpoint.communication import contacts as C
    C.contacts().add(C.Contact(name="Sarah", phone="+15550001111"))
    world.install_providers(configured=False)
    result = world.dispatch("send_message", {"to": "Sarah", "body": "hi"})
    assert "no credentials configured" in result and "confirmation_id" not in result


# ── Bookkeeping ───────────────────────────────────────────────────────────────

def test_every_call_is_logged(world):
    world.dispatch("think", {"reasoning": "x"})
    world.dispatch("list_files", {})
    assert world.call_count() == 2 and world.call_count("think") == 1


def test_summary_counts_by_tool(world):
    world.dispatch("think", {"reasoning": "a"})
    world.dispatch("think", {"reasoning": "b"})
    assert "think×2" in world.summary()
