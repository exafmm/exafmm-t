## Build exafmm-t's python module 

exafmm-t uses pybind11 to create a python interface.

After cloning the repo locally, first checkout the `pybind11` branch:

```shell
git checkout pybind11
```

pybind11 is a git submodule of exafmm-t, thus we need to initialize and update pybind11:

```shell
git submodule init
git submodule update
```

You will see a `pybind11` directory in your repo after this step. Next, let's compile pybind11:

```shell
cd pybind11
mkdir build
cd build
cmake ..
make check -j 4
```
Use `cmake -DPYTHON_EXECUTABLE:FILEPATH=<path-to-python-executable> ..` instead of `cmake ..` when `cmake` does not detect the right Python version.

The last line will compile and run the tests. Please refer to [pybind11's documentation](https://pybind11.readthedocs.io/en/stable/basics.html) for details.
Now, we are ready to compile exafmm-t's python module. Let's go back to the root directory of exafmm-t's repo, create a `build` directory for a out-of-source build:

```shell
cd ../../
mkdir build
cd build
```

Finally, configure and compile the python module:

```shell
cmake ..
make exafmm
```

If you wish to append additional compiler flags, replace the first line with `cmake -D CMAKE_CXX_FLAGS="your_flags" ..`.
A shared library file `exafmm.cpython-37m-x86_64-linux-gnu.so` will be created during compilation. 
