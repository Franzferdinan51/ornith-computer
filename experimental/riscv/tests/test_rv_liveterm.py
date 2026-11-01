"""Tests for chipformer.rv.liveterm's served terminal page (no guest needed)."""

from chipformer.rv.liveterm import PAGE, MachineThread


class _FakeMachine:
    def __init__(self, out):
        self.out = bytearray(out)
        self.steps = 0
        self.halted = False
        self.fed = []
        self.on_qwen_prompt = None
        self._n = 0

    def feed_input(self, data):
        self.fed.append(bytes(data))

    def run(self, n):
        self.steps += n
        self._n += 1
        if self._n >= 3:
            self.halted = True


def test_page_folds_long_qwen_protocol_lines():
    # the full ~2KB QWEN: prompt repeats every agent round; the page
    # must fold those lines or the terminal drowns in prompt text
    assert "function fold(line)" in PAGE
    assert "line.startsWith('QWEN:')" in PAGE
    assert "chars]" in PAGE


def test_page_scroll_follows_only_at_bottom():
    # readers scrolled up into history must not get yanked to the bottom
    assert "term.scrollHeight - term.scrollTop - term.clientHeight" in PAGE


def test_autologin_types_root_once_at_login_prompt():
    m = _FakeMachine(b"Welcome to Buildroot\r\n\rbuildroot login: ")
    mt = MachineThread(b"", b"", None, machine=m)
    mt.run()
    assert m.fed == [b"root\n"]
    assert mt.phase == 1


def test_no_autologin_leaves_login_prompt_alone():
    m = _FakeMachine(b"Welcome to Buildroot\r\n\rbuildroot login: ")
    mt = MachineThread(b"", b"", None, machine=m, autologin=False)
    mt.run()
    assert m.fed == []
    assert mt.phase == 1
