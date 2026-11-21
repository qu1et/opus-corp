#!/usr/bin/bash
export PATH=/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH
echo "== files in /mingw64/bin =="
for b in gcc.exe g++.exe cmake.exe make.exe mingw32-make.exe; do
  if [ -f "/mingw64/bin/$b" ]; then echo "OK    $b"; else echo "MISS  $b"; fi
done
echo "== which =="
command -v gcc || echo "gcc не в PATH"
command -v g++ || echo "g++ не в PATH"
command -v cmake || echo "cmake не в PATH"
command -v make || echo "make не в PATH"
command -v mingw32-make || echo "mingw32-make не в PATH"
echo "== versions =="
/mingw64/bin/gcc.exe --version 2>&1 | head -2
/mingw64/bin/cmake.exe --version 2>&1 | head -2
echo "== pacman list =="
pacman -Q mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake make 2>&1 | head -10
