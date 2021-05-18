<p align="center"><img src="docs/includes/img/circt-logo.svg"/></p>

[![](https://github.com/circt/circt/workflows/Build%20and%20Test/badge.svg?event=push)](https://github.com/llvm/circt/actions?query=workflow%3A%22Build+and+Test%22)
[![Nightly integration tests](https://github.com/llvm/circt/workflows/Nightly%20integration%20tests/badge.svg)](https://github.com/llvm/circt/actions?query=workflow%3A%22Nightly+integration+tests%22)

# ⚡️ "CIRCT" / Circuit IR Compilers and Tools

"CIRCT" stands for "Circuit IR Compilers and Tools".  One might also interpret
it as the recursively as "CIRCT IR Compiler and Tools".  The T can be
selectively expanded as Tool, Translator, Team, Technology, Target, Tree, Type,
... we're ok with the ambiguity.

The CIRCT community is an open and welcoming community.  If you'd like to
participate, you can do so in a number of different ways:

1) Join our [Discourse Forum](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/) 
on the LLVM Discourse server.  To get a "mailing list" like experience click the 
bell icon in the upper right and switch to "Watching".  It is also helpful to go 
to your Discourse profile, then the "emails" tab, and check "Enable mailing list 
mode".  You can also do chat with us on [CIRCT channel](https://discord.com/channels/636084430946959380/742572728787402763) 
of LLVM discord server.

2) Join our weekly video chat.  Please see the
[meeting notes document](https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#)
for more information.

3) Contribute code.  CIRCT follows all of the LLVM Policies: you can create pull
requests for the CIRCT repository, and gain commit access using the [standard LLVM policies](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/).

## Motivation

The EDA industry has well-known and widely used proprietary and open source
tools.  However, these tools are inconsistent, have usability concerns, and were
not designed together into a common platform.  Furthermore
these tools are generally built with
[Verilog](https://en.wikipedia.org/wiki/Verilog) (also
[VHDL](https://en.wikipedia.org/wiki/VHDL)) as the IRs that they
interchange.  Verilog has well known design issues, and limitations, e.g.
suffering from poor location tracking support.

The CIRCT project is an (experimental!) effort looking to apply MLIR and
the LLVM development methodology to the domain of hardware design tools.  Many
of us dream of having reusable infrastructure that is modular, uses
library-based design techniques, is more consistent, and builds on the best
practices in compiler infrastructure and compiler design techniques.

By working together, we hope that we can build a new center of gravity to draw
contributions from the small (but enthusiastic!) community of people who work
on open hardware tooling.  In turn we hope this will propel open tools forward,
enables new higher-level abstractions for hardware design, and
perhaps some pieces may even be adopted by proprietary tools in time.

For more information, please see our longer [charter document](docs/Charter.md).

## Setting this up

These commands can be used to setup CIRCT project:

1) **Install Dependencies** of LLVM/MLIR according to [the
  instructions](https://mlir.llvm.org/getting_started/), including cmake and 
  ninja.

2) **Check out LLVM and CIRCT repos.**  CIRCT contains LLVM as a git
submodule.  The LLVM repo here includes staged changes to MLIR which
may be necessary to support CIRCT.  It also represents the version of
LLVM that has been tested.  MLIR is still changing relatively rapidly,
so feel free to use the current version of LLVM, but APIs may have
changed.

```
$ git clone git@github.com:circt/circt.git
$ cd circt
$ git submodule init
$ git submodule update
```

*Note:* The repository is set up so that `git submodule update` performs a 
shallow clone, meaning it downloads just enough of the LLVM repository to check 
out the currently specified commit. If you wish to work with the full history of
the LLVM repository, you can manually "unshallow" the the submodule:

```
$ cd llvm
$ git fetch --unshallow
```

3) **Build and test LLVM/MLIR:**

```
$ cd circt
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
$ ninja
$ ninja check-mlir
```

4) **Build and test CIRCT:**

```
$ cd circt
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
$ ninja
$ ninja check-circt
$ ninja check-circt-integration # Run the integration tests.
```

The `-DCMAKE_BUILD_TYPE=DEBUG` flag enables debug information, which makes the
whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.

To get something that runs fast, use `-DCMAKE_BUILD_TYPE=Release` or
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` if you want to go fast and optionally if
you want debug info to go with it.  `Release` mode makes a very large difference
in performance.

Consult the [Getting Started](docs/GettingStarted.md) page for detailed 
information on configuring and compiling CIRCT.

## Using the Python Bindings

If you are mainly interested in using CIRCT from Python scripts, you need to compile both LLVM/MLIR and CIRCT with Python bindings enabled. To do this, follow the steps in "Setting things up" above, adding the following options:

```
# in the LLVM/MLIR build directory
cmake [...] -DMLIR_BINDINGS_PYTHON_ENABLED=ON

# in the CIRCT build directory
cmake [...] -DCIRCT_BINDINGS_PYTHON_ENABLED=ON
```

Afterwards, use `ninja check-circt-integration` to ensure that the bindings work. (This will now additionally spin up a couple of Python scripts to test that they are accessible.)

### Without Installation

If you want to try the bindings fresh from the compiler without installation, you need to ensure Python can find both the additional modules as well as the compiled native libraries:

```
export PYTHONPATH="$PWD/llvm/build/python:$PWD/build/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$PWD/llvm/build/lib:$PWD/build/lib:$LD_LIBRARY_PATH"
```

### With Installation

If you are installing CIRCT through `ninja install` anyway, the libraries and Python modules will be installed into the correct location automatically.

### Trying things out

Now you are able to use the CIRCT dialects and infrastructure from a Python interpreter and script:

```python
# silicon.py
import circt
import mlir
from mlir.ir import *
from circt.dialects import hw, comb

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  i42 = IntegerType.get_signless(42)
  m = Module.create()
  with InsertionPoint(m.body):
    @hw.HWModuleOp.from_py_func(i42, i42)
    def magic(a, b):
      return comb.XorOp(i42, [a, b]).result
  print(m)
```

Running this script through `python3 silicon.py` should print the following MLIR:

```mlir
module  {
  hw.module @magic(%a: i42, %b: i42) -> (%result0: i42) {
    %0 = comb.xor %a, %b : i42
    hw.output %0 : i42
  }
}
```
