# Contributing guidelines

To contribute to exafmm-t, please first fork the repository and push changes on your fork and then submit a pull request (PR).
For minor fixes, please make sure that your code passes all [current tests](https://exafmm.github.io/exafmm-t/compile.html#install-exafmm-t) before submitting a PR.
If your contribution introduces new features, please also go through the checklist below:

- add unit test source files in `tests` folder
- add compilation instructions to `tests/Makefile.am` (since we are using autotools)
- new functions and classes should have a Doxygen style docstring

Once your branch is merged into `master`, Travis CI will automatically generate new documentation and push to `gh-pages` branch.
You could follow [these intructions](https://exafmm.github.io/exafmm-t/documentation.html) to build documentation locally.