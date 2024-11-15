mkdir -p build
cd build
# cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake -Wno-dev .. > 0
cmake --build .
exit_status=$?
cd ..

if [ $exit_status -eq 0 ]; then
    ./build/MAIClient
fi
