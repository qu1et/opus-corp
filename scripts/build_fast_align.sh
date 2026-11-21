#!/usr/bin/bash
# Сборка fast_align и atools (задача 4) под Windows через MSYS2/MinGW.
# Запуск из PowerShell:  C:\msys64\usr\bin\bash.exe -lc "bash /d/practice/scripts/build_fast_align.sh"
set -e
export PATH=/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH

SRC=/d/practice/tools/fast_align
mkdir -p "$SRC"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 https://github.com/clab/fast_align.git "$SRC"
fi

cd "$SRC/build" 2>/dev/null || mkdir -p "$SRC/build" && cd "$SRC/build"
rm -f CMakeCache.txt
cmake -G "MinGW Makefiles" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
cmake --build . -j 4

# MinGW-бинари требуют runtime-DLL; кладём их рядом, чтобы exe работал без MSYS2 в PATH.
for dll in libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll; do
  if [ -f "/mingw64/bin/$dll" ]; then cp "/mingw64/bin/$dll" "$SRC/build/"; fi
done

ls -la
echo "binary: fast_align -> $(pwd)/fast_align.exe"
echo "binary: atools   -> $(pwd)/atools.exe"
