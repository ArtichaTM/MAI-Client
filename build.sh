mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=/mnt/c/Users/Articha/Desktop/Temp/Diploma/Client/libtorch ..
cmake -Wno-dev .. > 0
cmake --build .
exit_status=$?
cd ..

if [ $exit_status -eq 0 ]; then
    ./build/MAIClient
fi