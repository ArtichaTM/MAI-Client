mkdir -p build
cd build
cmake .. > 0
cmake --build .
exit_status=$?
cd ..

if [ $exit_status -eq 0 ]; then
    ./build/MAIClient
fi