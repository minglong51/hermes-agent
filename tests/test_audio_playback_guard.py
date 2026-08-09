"""Regression tests for the test-suite audio-playback guard.

The incident: a plain test run spoke the words "partial answer complete" out
of a developer's speakers. That string is a fake ``final_response`` from
``tests/test_tui_gateway_server.py``. The route was entirely in-process — no
leaked shell variable, so ``scripts/run_tests.sh``'s ``env -i`` was no help:

  1. A test drives the ``voice.toggle`` RPC with ``action="tts"``. The
     handler flips the flag by writing the real process environment:
     ``os.environ["HERMES_VOICE_TTS"] = "1"``.
  2. The flag outlives that test (``monkeypatch.delenv`` on an *absent* key
     records no undo entry), so every later test in the process sees it.
  3. Any later test that drives a turn to completion hits the TTS dispatch in
     ``prompt.submit``, which calls ``hermes_cli.voice.speak_text`` on a
     daemon thread with the turn's final response text.
  4. ``speak_text`` needs no API key to be audible — ``tools/tts_tool.py``
     defaults to the keyless ``edge`` provider.

Two independent defences in ``tests/conftest.py``, one test each below:

  * ``_HERMES_BEHAVIORAL_VARS`` now blanks ``HERMES_VOICE`` /
    ``HERMES_VOICE_TTS`` at every test setup, so step 2 cannot cross a test
    boundary.
  * ``_audio_playback_guard`` stubs ``speak_text`` outright, so step 3 stays
    silent even *within* the test that set the flag itself.

The second is the load-bearing one: env blanking alone cannot stop code under
test from re-setting the variable mid-test.
"""

import os
import shutil
import subprocess
import sys
import types
import wave

import pytest

from tui_gateway import server

# Resolved once, before any guard is installed. Gates the end-to-end test on
# "is there a player to block?" rather than on the platform — the previous
# Darwin-only gate meant Linux CI never ran it.
_INSTALLED_PLAYER = next(
    (p for p in ("afplay", "ffplay", "aplay", "paplay") if shutil.which(p)), None
)


def test_voice_toggle_still_leaks_the_env_var_but_speech_is_stubbed(monkeypatch):
    """The dangerous primitive is neutralised even when the flag IS set.

    This reproduces the incident's first step for real — it drives the same
    ``voice.toggle`` RPC that the original culprit test does, and asserts the
    handler really does write ``os.environ`` (so the guard is being tested
    against live behaviour, not a straw man). It then walks the second step:
    ``speak_text`` is called exactly as ``prompt.submit`` calls it, with the
    exact string the user heard — and must not reach the TTS backend.
    """
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {}})
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    resp = server.dispatch(
        {"id": "tts", "method": "voice.toggle", "params": {"action": "tts"}}
    )

    # The handler mutates the real process environment. This is the leak;
    # it is upstream behaviour we are guarding against, not asserting away.
    assert resp["result"]["tts"] is True
    assert os.environ.get("HERMES_VOICE_TTS") == "1"
    assert server._voice_tts_enabled() is True

    # Any call into the TTS backend from here on would be real synthesis and
    # real playback. Stand a recorder in front of it — a *recorder*, not a
    # raising stub, because ``speak_text`` wraps its whole body in
    # ``except Exception`` and would swallow an AssertionError, quietly
    # turning this test green whether or not the guard is doing anything.
    import tools.tts_tool as tts_tool

    calls = []
    monkeypatch.setattr(
        tts_tool,
        "text_to_speech_tool",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    # Called exactly as tui_gateway/server.py's prompt.submit completion path
    # calls it (a late `from hermes_cli.voice import speak_text`), with the
    # exact fixture string that came out of the speakers.
    from hermes_cli.voice import speak_text

    assert speak_text("partial answer complete") is None
    assert calls == [], (
        "audio guard breached: speak_text reached the TTS backend — on an "
        "unguarded run this is real synthesis through the keyless 'edge' "
        "provider, played through the speakers"
    )


def test_voice_env_does_not_leak_into_the_next_test():
    """Second defence: the flag the previous test set must not have survived.

    Ordering matters — this test only means anything because it runs after the
    one above, in the same process, which left ``HERMES_VOICE_TTS=1`` set.
    Before ``HERMES_VOICE``/``HERMES_VOICE_TTS`` were added to
    ``_HERMES_BEHAVIORAL_VARS``, this assertion failed.
    """
    assert "HERMES_VOICE_TTS" not in os.environ
    assert "HERMES_VOICE" not in os.environ
    assert server._voice_tts_enabled() is False
    assert server._voice_mode_enabled() is False


def test_guard_can_be_opted_out_of_explicitly():
    """The stub is a guard, not a lobotomy — the real function is reachable."""
    import hermes_cli.voice as voice

    assert voice.speak_text.__name__ == "_blocked_speak_text"


@pytest.mark.real_audio_playback
def test_bypass_marker_restores_the_real_speak_text():
    """``@pytest.mark.real_audio_playback`` hands back the real primitive.

    Asserts identity only — it does not call it, which would speak aloud.
    """
    import hermes_cli.voice as voice

    assert voice.speak_text.__name__ == "speak_text"


# =====================================================================
# Second incident: the route that went around every stub above.
#
# "Hello world." came out of the speakers on an ordinary test run, from
# ``tests/gateway/test_voice_command.py::TestStreamTtsToSpeaker::
# test_none_sentinel_flushes_buffer``. That test calls the real
# ``tools.tts_tool.stream_tts_to_speaker``, which synthesizes each sentence
# through the keyless ``edge`` provider and plays it with a late
# ``from tools.voice_mode import play_audio_file`` — touching neither
# ``hermes_cli.voice`` (where the stubs above live) nor any attribute of
# ``tools.tts_tool`` a patch could catch.
#
# The lesson is that enumerating routes does not converge: the fix is to
# guard the *egress* — the player subprocess every route must eventually
# reach. The tests below pin that behaviour.
#
# Note on arguments: each command below is given a target that is silent
# even if the guard is broken (a nonexistent file, a nonexistent binary
# path). A test that proves the speakers stay shut must not be the thing
# that opens them when it regresses.
# =====================================================================


def test_unmarked_test_cannot_launch_afplay():
    """``afplay`` is refused for every subprocess entry point."""
    for call, invoke in (
        ("run", lambda cmd: subprocess.run(cmd)),
        ("call", lambda cmd: subprocess.call(cmd)),
        ("check_call", lambda cmd: subprocess.check_call(cmd)),
        ("check_output", lambda cmd: subprocess.check_output(cmd)),
        ("Popen", lambda cmd: subprocess.Popen(cmd)),
    ):
        with pytest.raises(RuntimeError, match="audio guard") as excinfo:
            invoke(["afplay", "/nonexistent/silent.wav"])
        assert "real_audio_playback" in str(excinfo.value), (
            f"subprocess.{call} was blocked but the error does not tell the "
            "author how to opt out"
        )


def test_unmarked_test_cannot_launch_say():
    """macOS' ``say`` is refused, including via an absolute path.

    Basename matching is the point: ``/usr/bin/say`` is the same speaker.
    """
    # Bare name, so basename matching is exercised against the real binary.
    # ``say -o <file>`` renders to a file instead of the speakers, so this
    # stays silent even on the day the guard regresses.
    with pytest.raises(RuntimeError, match="audio guard"):
        subprocess.run(
            ["say", "-o", "/nonexistent/dir/out.aiff", "never spoken"]
        )

    with pytest.raises(RuntimeError, match="audio guard"):
        subprocess.Popen(
            ["/nonexistent/bin/say", "-o", "/nonexistent/out.aiff", "never"]
        )


def test_shell_wrapped_player_is_refused():
    """``sh -c "… afplay …"`` is the WSL2 playback path, not a hypothetical.

    ``play_audio_file`` inserts ``["/bin/sh", "-c", "( ffmpeg … && "
    "powershell.exe … Media.SoundPlayer … )"]`` at the FRONT of its player
    list on WSL2. A guard matching only the command head sees ``sh`` and lets
    the real incident through on that platform.
    """
    with pytest.raises(RuntimeError, match="audio guard"):
        subprocess.run(["/bin/sh", "-c", "afplay /nonexistent/x.wav"])

    with pytest.raises(RuntimeError, match="audio guard"):
        subprocess.Popen(["bash", "-c", "( ffplay -nodisp /nonexistent/x.wav )"])


def test_bytes_and_quoted_argv_are_refused():
    """Byte-string argv and quoted shell strings must not slip past.

    ``str(b"afplay")`` is ``"b'afplay'"``, which matches no player name — so a
    parser that stringifies without decoding waves real playback through.
    """
    with pytest.raises(RuntimeError, match="audio guard"):
        subprocess.run([b"/usr/bin/afplay", b"/nonexistent/x.wav"])

    with pytest.raises(RuntimeError, match="audio guard"):
        subprocess.run('"/usr/bin/afplay" /nonexistent/x.wav', shell=True)


def test_os_system_and_popen_are_refused():
    """``os.system`` reaches a player without touching ``subprocess``."""
    with pytest.raises(RuntimeError, match="audio guard"):
        os.system("afplay /nonexistent/x.wav")

    with pytest.raises(RuntimeError, match="audio guard"):
        os.popen("say -o /nonexistent/dir/out.aiff hello")


def test_guard_does_not_block_unrelated_subprocesses():
    """The block is a scalpel — ordinary subprocesses still run.

    Without this, a guard that over-matched (e.g. substring rather than
    command basename) would pass every test above while quietly breaking
    unrelated suites.
    """
    result = subprocess.run(
        [sys.executable, "-c", "print('ok')"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


@pytest.mark.skipif(
    _INSTALLED_PLAYER is None,
    reason="no system audio player on PATH, so there is nothing to block",
)
def test_tts_tool_playback_route_cannot_reach_the_speakers(tmp_path, monkeypatch):
    """End-to-end: the exact call the TTS pipeline makes, and it stays silent.

    ``tools.voice_mode.play_audio_file`` is called here unmocked, with a real
    playable WAV — precisely what ``stream_tts_to_speaker`` does per sentence.
    It must fail closed. ``play_audio_file`` swallows the guard's RuntimeError
    in its per-player ``except Exception`` and falls through to "no player
    available", so ``False`` is the observable signature of a blocked play.

    Runs on every platform with a player installed, not just macOS. Forcing
    ``_sounddevice_output_allowed`` off routes WAV playback to the system
    player everywhere, so Linux CI exercises the same subprocess route macOS
    takes natively — without this the guard's only end-to-end assertion never
    ran in CI at all.
    """
    import tools.voice_mode as voice_mode

    monkeypatch.setattr(voice_mode, "_sounddevice_output_allowed", lambda: False)

    wav_path = tmp_path / "silence.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 2400)

    from tools.voice_mode import play_audio_file

    assert play_audio_file(str(wav_path)) is False, (
        "audio guard breached: play_audio_file reached a system player — on "
        "an unguarded run this is real sound out of the speakers"
    )


@pytest.mark.real_audio_playback
def test_bypass_marker_restores_real_subprocess():
    """The marker un-blocks the players for tests that genuinely need them.

    Uses a nonexistent binary so nothing is spawned: under the guard the
    command is rejected before exec with ``RuntimeError``; with the guard
    lifted it reaches exec and fails with ``FileNotFoundError``. The exception
    type is the whole assertion.
    """
    with pytest.raises(FileNotFoundError):
        subprocess.run(["/nonexistent/bin/afplay", "/nonexistent/x.wav"])
