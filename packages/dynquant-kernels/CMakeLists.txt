# DynQuant compiled kernels.
#
# Driven by scikit-build-core (see pyproject.toml). Two modes:
#
#   CUDA found      -> real kernels, fat binary over several SM versions.
#   CUDA absent     -> the same extension with CPU implementations only.
#
# The CPU-only mode is not a consolation prize. It is what makes the sdist
# installable on a machine without a CUDA toolkit, and it is what lets a cheap
# CPU CI runner catch schema/registration/pybind errors in minutes rather than
# queueing for a GPU runner. Set DYNQUANT_REQUIRE_CUDA=ON in release builds so a
# toolkit that quietly failed to be detected becomes a hard error instead of a
# wheel that ships with no kernels in it.

cmake_minimum_required(VERSION 3.26)

project(dynquant_kernels LANGUAGES CXX)

option(DYNQUANT_REQUIRE_CUDA "Fail if no CUDA toolkit is found" OFF)
option(DYNQUANT_FAST_MATH "Allow -use_fast_math (do not: see below)" OFF)
set(DYNQUANT_CUDA_ARCHS "" CACHE STRING
    "Override CMAKE_CUDA_ARCHITECTURES, e.g. '80-real;90-real'. Empty = auto.")

# Every option above is also readable from the environment. Nobody invokes this
# file directly -- it is driven through `pip install`, which forwards nothing to
# CMake unless you spell `-C cmake.define.NAME=value`. An environment variable is
# what a person actually reaches for, and a build knob that silently does nothing
# is worse than not having it: the sm_80-only build below would have taken the
# default fat-binary path and looked like it worked.
foreach(_dq_opt DYNQUANT_REQUIRE_CUDA DYNQUANT_FAST_MATH DYNQUANT_CUDA_ARCHS)
  if(NOT ${_dq_opt} AND DEFINED ENV{${_dq_opt}})
    set(${_dq_opt} "$ENV{${_dq_opt}}")
    message(STATUS "${_dq_opt}=${${_dq_opt}} (from the environment)")
  endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Python and Torch
# ---------------------------------------------------------------------------

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Ask the *interpreter doing the build* where its torch is. Hardcoding a path or
# relying on CMAKE_PREFIX_PATH being set by the caller is how extensions end up
# linked against a different libtorch than the one they run with -- which
# manifests as an undefined-symbol ImportError, or worse, as a crash inside ATen.
#
# The CUDA field is printed with a `cuda=` prefix so that the line is never empty.
# On a CPU-only torch `torch.version.cuda` is None, and an empty final line is
# removed by OUTPUT_STRIP_TRAILING_WHITESPACE -- which left a two-element list and
# failed with `list index: 2 out of range` on exactly the CPU-only configuration
# the cheap CI runner exists to exercise. A non-empty sentinel survives stripping.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import torch;print(torch.utils.cmake_prefix_path);print(torch.__version__);print('cuda='+(torch.version.cuda or ''))"
  OUTPUT_VARIABLE _dq_torch_probe
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _dq_torch_rc)
if(NOT _dq_torch_rc EQUAL 0)
  message(FATAL_ERROR
      "Could not import torch with ${Python_EXECUTABLE}.\n"
      "dynquant-kernels must be built against the torch you will run with. If pip "
      "created an isolated build environment, torch there may differ from yours; "
      "build with `pip install --no-build-isolation` instead.")
endif()
string(REPLACE "\r" "" _dq_torch_probe "${_dq_torch_probe}")
string(REPLACE "\n" ";" _dq_torch_lines "${_dq_torch_probe}")
# Assert the shape rather than indexing hopefully: a probe that returns fewer
# lines than expected should say so, not fail with a bare list-index error.
list(LENGTH _dq_torch_lines _dq_torch_nlines)
if(_dq_torch_nlines LESS 3)
  message(FATAL_ERROR
      "The torch probe returned ${_dq_torch_nlines} lines, expected 3. Output was:\n"
      "${_dq_torch_probe}")
endif()
list(GET _dq_torch_lines 0 _dq_torch_cmake_dir)
list(GET _dq_torch_lines 1 DYNQUANT_TORCH_VERSION)
list(GET _dq_torch_lines 2 _dq_torch_cuda_field)
string(REGEX REPLACE "^cuda=" "" DYNQUANT_TORCH_CUDA_VERSION "${_dq_torch_cuda_field}")
message(STATUS "torch ${DYNQUANT_TORCH_VERSION} (CUDA '${DYNQUANT_TORCH_CUDA_VERSION}')")

list(APPEND CMAKE_PREFIX_PATH "${_dq_torch_cmake_dir}")
find_package(Torch REQUIRED)

# find_package(Torch) appends its own flags -- including the _GLIBCXX_USE_CXX11_ABI
# setting -- to CMAKE_CXX_FLAGS. That has to be honoured: mixing ABIs against
# libtorch produces link errors on std::string arguments that look like missing
# symbols.

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

include(CheckLanguage)
check_language(CUDA)

set(DYNQUANT_WITH_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(DYNQUANT_WITH_CUDA ON)
  message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION}")

  if(DYNQUANT_TORCH_CUDA_VERSION STREQUAL "")
    message(WARNING
        "A CUDA toolkit was found but this torch (${DYNQUANT_TORCH_VERSION}) is a "
        "CPU build, so the extension cannot link torch's CUDA libraries. Building "
        "CPU-only. Install a CUDA build of torch to get kernels.")
    set(DYNQUANT_WITH_CUDA OFF)
  else()
    # Toolkit and torch must agree on the CUDA *major* version. A cu12-built
    # extension loaded next to a cu11 torch finds two libcudart in the process and
    # the failure is a segfault inside a stream call, with nothing pointing here.
    string(REGEX MATCH "^([0-9]+)" _dq_torch_cuda_major "${DYNQUANT_TORCH_CUDA_VERSION}")
    string(REGEX MATCH "^([0-9]+)" _dq_tk_cuda_major "${CUDAToolkit_VERSION}")
    if(NOT _dq_torch_cuda_major STREQUAL _dq_tk_cuda_major)
      message(FATAL_ERROR
          "CUDA major version mismatch: torch was built for CUDA "
          "${DYNQUANT_TORCH_CUDA_VERSION} but the toolkit here is "
          "${CUDAToolkit_VERSION}. Install a matching toolkit, or a torch build "
          "matching this toolkit.")
    endif()
  endif()
elseif(DYNQUANT_REQUIRE_CUDA)
  message(FATAL_ERROR
      "No CUDA compiler found and DYNQUANT_REQUIRE_CUDA=ON. Set CUDACXX or add "
      "nvcc to PATH. (Unset the flag to build a CPU-only extension.)")
else()
  message(WARNING
      "No CUDA compiler found -- building a CPU-only extension. `dynquant doctor` "
      "will report backend=torch and quantized inference will run on the reference "
      "path, which is correct but not fast.")
endif()

if(DYNQUANT_WITH_CUDA)
  if(DYNQUANT_CUDA_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES "${DYNQUANT_CUDA_ARCHS}")
  else()
    # `-real` emits a cubin only; `-virtual` emits PTX only. Shipping cubins for
    # the architectures people actually have keeps launches instant, and one
    # trailing PTX target means a GPU newer than this toolkit still runs -- the
    # driver JITs on first launch (seconds, once, cached) instead of failing with
    # "no kernel image is available".
    set(_dq_archs 75-real 80-real 86-real)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND _dq_archs 89-real 90-real)
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
      list(APPEND _dq_archs 100-real 120-real)
      list(APPEND _dq_archs 120-virtual)
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND _dq_archs 90-virtual)
    else()
      list(APPEND _dq_archs 86-virtual)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_dq_archs}")
  endif()
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

  # find_package(Torch) appends a `-gencode` for every entry of its own
  # TORCH_CUDA_ARCH_LIST to CMAKE_CUDA_FLAGS. Those are additive to
  # CMAKE_CUDA_ARCHITECTURES, not an alternative to it, so leaving them in place
  # means nvcc compiles the whole (4 bits x 4 MROWS x 3 dtypes) instantiation
  # matrix once per torch arch *and* once per ours -- and DYNQUANT_CUDA_ARCHS
  # stops meaning anything. Architecture selection belongs to one mechanism.
  string(REGEX REPLACE "-gencode[ =]+arch=compute_[0-9]+a?,code=[][a-z_0-9,]+" ""
         CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Deliberately NOT CUDA_SEPARABLE_COMPILATION. Relocatable device code would let
  # device functions be called across translation units, which we never do, and it
  # blocks nvcc from inlining the n-bit unpack helpers into the GEMV inner loop --
  # the one place in this project where inlining is the difference between
  # memory-bound and instruction-bound.

  # ---- Where torch's CUDA headers actually are -------------------------------
  #
  # `ATen/cuda/CUDAContext.h` includes <cusparse.h> and <cublas_v2.h>. Since the
  # cu12 wheels, torch no longer requires a full local toolkit: it ships the math
  # libraries as `nvidia-*` pip packages and the system CUDA install may be the
  # compiler alone -- which is what `nvidia/cuda:*-devel` images and this project's
  # own GPU box provide. torch.utils.cpp_extension.include_paths() does not report
  # those bundled directories, so any extension including an ATen CUDA header fails
  # at the preprocessor with "cusparse.h: No such file or directory" and nothing
  # names the cause.
  #
  # So: look for the header in the toolkit first, and only reach into torch's wheel
  # tree when the toolkit genuinely lacks it. Toolkit headers keep priority when
  # both exist -- they are the ones nvcc's built-ins were built against.
  # Two searches, two cache entries. Reusing one variable for both would make the
  # second configure of an existing build tree find the *bundled* hit still cached
  # under the toolkit's name, skip this branch, and drop the include directory that
  # made the first configure work -- a build that succeeds once and then fails.
  set(DYNQUANT_EXTRA_CUDA_INCLUDES "")
  find_path(DYNQUANT_TOOLKIT_CUSPARSE cusparse.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
  if(NOT DYNQUANT_TOOLKIT_CUSPARSE)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
        "import pathlib,torch;p=pathlib.Path(torch.__file__).parent.parent/'nvidia';\
print(';'.join(sorted(str(d) for d in p.glob('*/include') if d.is_dir())) if p.is_dir() else '')"
      OUTPUT_VARIABLE DYNQUANT_EXTRA_CUDA_INCLUDES
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_path(DYNQUANT_BUNDLED_CUSPARSE cusparse.h
              HINTS ${DYNQUANT_EXTRA_CUDA_INCLUDES} NO_DEFAULT_PATH)
    if(NOT DYNQUANT_BUNDLED_CUSPARSE)
      message(FATAL_ERROR
          "cusparse.h is in neither the CUDA toolkit (${CUDAToolkit_INCLUDE_DIRS}) "
          "nor torch's bundled nvidia packages. ATen's CUDA headers need it. "
          "Install the full toolkit, or `pip install nvidia-cusparse-cu${_dq_tk_cuda_major}`.")
    endif()
    message(STATUS "CUDA headers from torch's bundled packages: ${DYNQUANT_BUNDLED_CUSPARSE}")

    # ---- Fill the gaps; shadow nothing --------------------------------------
    #
    # Adding the directory that has `cusparse.h` used to mean adding everything
    # next to it. torch's CUDA 13 wheel is one monolithic `nvidia/cu13/include`
    # holding `crt/`, `cuda_runtime.h` and the whole runtime alongside the math
    # headers, so borrowing the one header the toolkit lacks replaced all the
    # ones it has.
    #
    # That is not a warning-level problem, it is a build that compiles zero
    # translation units. This box runs nvcc 13.0.88 against a wheel built from
    # 13.3, and the two disagree about a macro in `crt/host_runtime.h`:
    #
    #   toolkit 13.0  #define __cudaLaunch(fun, isTileKernel)   <- 2 parameters
    #   wheel   13.3  #define __cudaLaunch(fun)                 <- 1 parameter
    #
    # nvcc emits the two-argument form into its generated stub, the preprocessor
    # reads the wheel's one-argument macro, and every kernel in the file fails
    # with `macro "__cudaLaunch" passed 2 arguments, but takes just 1` -- an error
    # in a file nobody wrote, about a symbol nobody called. Both headers stamp
    # `CUDART_VERSION 13000`, so no version guard can tell them apart.
    #
    # Include *order* does not fix this. The obvious reading of `-isystem` is
    # "search after -I", and measured against nvcc that reading is wrong: with
    # `-I<toolkit>` ahead of `-isystem<wheel>` on the same command line, the
    # preprocessor still resolves `crt/host_runtime.h` to the wheel. So this does
    # not rely on ordering semantics at all. It builds a directory of symlinks to
    # exactly those entries the toolkit does not have, and adds only that. A
    # header the toolkit provides is then not reachable through this path at all,
    # whatever nvcc does with the flags.
    set(_dq_shim "${CMAKE_CURRENT_BINARY_DIR}/cuda_header_shim")
    file(REMOVE_RECURSE "${_dq_shim}")
    file(MAKE_DIRECTORY "${_dq_shim}")
    set(_dq_shimmed "")
    foreach(_dq_dir IN LISTS DYNQUANT_EXTRA_CUDA_INCLUDES)
      file(GLOB _dq_entries RELATIVE "${_dq_dir}" "${_dq_dir}/*")
      foreach(_dq_entry IN LISTS _dq_entries)
        set(_dq_in_toolkit FALSE)
        foreach(_dq_tk IN LISTS CUDAToolkit_INCLUDE_DIRS)
          if(EXISTS "${_dq_tk}/${_dq_entry}")
            set(_dq_in_toolkit TRUE)
            break()
          endif()
        endforeach()
        # First wheel to offer a name wins, and only if the toolkit offers none.
        if(NOT _dq_in_toolkit AND NOT EXISTS "${_dq_shim}/${_dq_entry}")
          file(CREATE_LINK "${_dq_dir}/${_dq_entry}" "${_dq_shim}/${_dq_entry}" SYMBOLIC)
          list(APPEND _dq_shimmed "${_dq_entry}")
        endif()
      endforeach()
    endforeach()
    list(LENGTH _dq_shimmed _dq_nshim)
    message(STATUS "CUDA header shim: ${_dq_nshim} entries the toolkit lacks")
    # The header that sent us down this path must survive the filter. If the
    # toolkit turns out to have had it after all, the detection above is wrong and
    # silently building without the shim would hide that.
    if(NOT EXISTS "${_dq_shim}/cusparse.h")
      message(FATAL_ERROR
          "the CUDA header shim did not pick up cusparse.h, but the toolkit was "
          "found not to have it -- the two checks disagree, so one is wrong.")
    endif()
    set(DYNQUANT_EXTRA_CUDA_INCLUDES "${_dq_shim}")
  endif()
endif()

# ---------------------------------------------------------------------------
# The extension
# ---------------------------------------------------------------------------

set(_dq_sources csrc/bindings.cpp)
if(DYNQUANT_WITH_CUDA)
  list(APPEND _dq_sources
       csrc/probe/probe_cuda.cu
       csrc/probe/probe_cublaslt.cpp
       csrc/quant/dequant.cu
       csrc/gemv/gemv.cu
       csrc/moe/grouped_gemv.cu)
endif()

Python_add_library(_C MODULE WITH_SOABI ${_dq_sources})

target_include_directories(_C PRIVATE csrc/include)

# `torch`, and deliberately not `torch_python`. The module entry point is a bare
# CPython init with no methods; every op reaches Python through the dispatcher, so
# nothing here touches pybind11 or <torch/python.h>. Linking libtorch_python from
# an out-of-tree extension pulls a second copy of torch's pybind type registry into
# the process, which turns into "generic_type: type ... is already registered" at
# import on some builds -- and here it would not even buy anything. (It is also
# passed as a bare -ltorch_python by TorchConfig with no matching -L, so it does
# not link at all unless the caller adds torch/lib to the search path.)
target_link_libraries(_C PRIVATE torch)

if(DYNQUANT_WITH_CUDA)
  target_compile_definitions(_C PRIVATE DYNQUANT_WITH_CUDA=1)
  if(DYNQUANT_EXTRA_CUDA_INCLUDES)
    # This is the shim built above, not the vendored tree itself: it contains only
    # headers the toolkit does not have, so it cannot shadow the toolkit no matter
    # where nvcc decides to search it. SYSTEM because warnings from headers we did
    # not write are not ours to fix.
    target_include_directories(_C SYSTEM PRIVATE ${DYNQUANT_EXTRA_CUDA_INCLUDES})
  endif()
  target_link_libraries(_C PRIVATE CUDA::cudart CUDA::cublas)
  if(TARGET CUDA::cublasLt)
    target_link_libraries(_C PRIVATE CUDA::cublasLt)
  else()
    # Older FindCUDAToolkit versions do not define the target even though the
    # library is present next to libcublas.
    find_library(DYNQUANT_CUBLASLT cublasLt
                 HINTS "${CUDAToolkit_LIBRARY_DIR}" REQUIRED)
    target_link_libraries(_C PRIVATE "${DYNQUANT_CUBLASLT}")
  endif()

  target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      -O3
      # Source/line info in the cubin. Costs nothing at runtime and is what makes
      # compute-sanitizer and Nsight point at a line instead of an address.
      -lineinfo
      --expt-relaxed-constexpr
      --expt-extended-lambda
      # Torch's half types rely on these being available in device code.
      -U__CUDA_NO_HALF_OPERATORS__
      -U__CUDA_NO_HALF_CONVERSIONS__
      -U__CUDA_NO_HALF2_OPERATORS__
      -U__CUDA_NO_BFLOAT16_CONVERSIONS__
      >)

  if(DYNQUANT_FAST_MATH)
    # Off by default and it should stay off. -use_fast_math turns on
    # -ftz=true (denormals to zero) and approximate division, and a dequantized
    # weight near the bottom of a 2-bit group's range is exactly the denormal
    # case. Accuracy claims measured with this on are not reproducible.
    target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
    message(WARNING "DYNQUANT_FAST_MATH=ON: accuracy results from this build are not comparable.")
  endif()
endif()

if(MSVC)
  target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
  if(DYNQUANT_WITH_CUDA)
    target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
  endif()
else()
  target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -fvisibility=hidden>)

  # ---- The C++ runtime ABI gate ---------------------------------------------
  #
  # 0.5.0 shipped a wheel that installed, imported, and did not work: built by
  # GCC 14, it needed `__cxa_call_terminate` (CXXABI_1.3.15), and every image
  # older than GCC 14 -- Ubuntu 22.04 included -- defines no such symbol. The
  # extension failed to load, `is_available()` returned False, and DynQuant fell
  # back to the torch path *silently*. On an L40 that fallback left 32 decode
  # tokens unfinished after 32 minutes at 0% GPU utilisation, against 20.97 tok/s
  # with the kernels loaded. No error, no warning, correct results, unusable.
  #
  # auditwheel did not catch it and could not have: the manylinux tag is computed
  # from the ELF's versioned symbol requirements, which topped out at
  # CXXABI_1.3.9 here -- because the one symbol that broke the wheel is
  # *unversioned*. So the check has to resolve C++ runtime symbols by name.
  #
  # POST_BUILD rather than a test: a wheel that cannot load must not reach the
  # point of being packaged, let alone uploaded. Failing at link time costs
  # seconds; failing at the user's import costs a silent 60x slowdown.
  add_custom_command(TARGET _C POST_BUILD
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_cxx_runtime_abi.py"
            "$<TARGET_FILE:_C>"
    COMMENT "Checking the C++ runtime ABI of $<TARGET_FILE_NAME:_C>"
    VERBATIM)
endif()

# ---------------------------------------------------------------------------
# Build stamp
# ---------------------------------------------------------------------------
#
# Written into the wheel so the loader can explain a mismatch in terms of what to
# install, rather than reporting an undefined symbol.

set(DYNQUANT_CUDA_VERSION "${CUDAToolkit_VERSION}")
if(DYNQUANT_WITH_CUDA)
  set(DYNQUANT_WITH_CUDA_PY 1)
else()
  set(DYNQUANT_CUDA_VERSION "")
  set(DYNQUANT_WITH_CUDA_PY 0)
endif()

# Parse the ABI version out of the header instead of repeating it here. The header
# is the C++ side's single source of truth; duplicating the number in CMake would
# create a third place for it to drift, and a stale build stamp would make the
# loader's handshake pass while the binary disagreed.
file(READ csrc/include/dynquant/abi.h _dq_abi_header)
string(REGEX MATCH "#define[ \t]+DYNQUANT_ABI_VERSION[ \t]+([0-9]+)" _dq_abi_match
       "${_dq_abi_header}")
if(NOT _dq_abi_match)
  message(FATAL_ERROR "Could not parse DYNQUANT_ABI_VERSION from csrc/include/dynquant/abi.h")
endif()
set(DYNQUANT_ABI_VERSION_PY "${CMAKE_MATCH_1}")
message(STATUS "DynQuant kernel ABI version ${DYNQUANT_ABI_VERSION_PY}")

configure_file(cmake/_build_info.py.in
               "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" @ONLY)

install(TARGETS _C LIBRARY DESTINATION dynquant_kernels RUNTIME DESTINATION dynquant_kernels)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" DESTINATION dynquant_kernels)
