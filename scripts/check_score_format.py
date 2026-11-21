"""Смотрит, в каком формате fast_align выдаёт скоры с флагом -s (первые строки)."""
import pathlib
import subprocess

src = pathlib.Path("align/Tatoeba.en-ru.txt").read_text(encoding="utf-8").splitlines()[:6]
tmp = pathlib.Path("tmp_score_check.txt")
tmp.write_text("\n".join(src) + "\n", encoding="utf-8", newline="\n")

res = subprocess.run(
    ["tools/fast_align/build/fast_align.exe", "-i", tmp.as_posix(), "-d", "-o", "-v", "-s"],
    capture_output=True, text=True, encoding="utf-8",
)
print("код:", res.returncode)
print("--- stdout ---")
print(res.stdout)
print("--- stderr ---")
print(res.stderr[:400])
tmp.unlink(missing_ok=True)
