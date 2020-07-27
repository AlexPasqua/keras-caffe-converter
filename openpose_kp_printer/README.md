## Description:
This script is used to calculate the key points on a video and print them into a csv file.

## Instructions:
1. Clone openpose from https://github.com/CMU-Perceptual-Computing-Lab/openpose and follow the installation guide
2. Install it
3. Put "op_kp_printer.cpp" into openpose/examples/user_code
4. Add this script to the list of the source files in openpose/examples/user_code/CMakeLists.txt
5. Go to openpose/build and type ```sudo make -j<n_coder-1>```
6. ```
cd ..
./build/examples/user_code/op_kp_printer.bin <flags>
```
