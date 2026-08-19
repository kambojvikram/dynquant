"""Render ``docs/images/cuda-kernel-architecture.png``.

The diagram is generated rather than drawn, for the same reason every number in
``docs/format-spec.md`` is generated: a hand-drawn architecture picture starts as
documentation and ends as folklore. This script is the source, so when the build
changes the picture is regenerated from a diff someone can read.

Every box traces to a file under ``packages/dynquant-kernels/``. Nothing here
describes a kernel that does not exist -- the P5-P8 compute kernels appear only in
the last band, as what the P0 probes de-risk, never as shipped work.

Usage::

    python docs/diagrams/kernel_architecture.py

Requires Pillow. Deliberately not matplotlib: this is boxes and text at exact pixel
positions, which is the one job a plotting library makes harder rather than easier.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------

SS = 2
"""Supersampling factor.

PIL antialiases glyphs but not rectangle edges or polygon arrowheads, so the whole
page is drawn at 2x and reduced with LANCZOS. 3x is visibly no better and costs
four times the memory.
"""

WIDTH = 2400
MARGIN = 64
CONTENT = WIDTH - 2 * MARGIN
RAIL = 92
"""Width of the left band-number rail, subtracted from every band's content."""

BG = "#0A0E13"
PANEL = "#131A22"
PANEL_ALT = "#0F151C"
CODE_BG = "#0A0F15"
BORDER = "#28323E"
BORDER_SOFT = "#1E2731"
FG = "#E8EFF6"
DIM = "#94A3B3"
MUTED = "#66727F"

CUDA = "#76B900"  # NVIDIA green: device code, nvcc, Thrust, cuBLASLt
TORCH = "#EE4C2C"  # torch orange: host C++, the dispatcher, libtorch linkage
PY = "#58A6FF"  # Python
BUILD = "#BC8CFF"  # CMake, cibuildwheel, release plumbing
WARN = "#E3B341"
FAIL = "#F85149"
TEAL = "#2DD4BF"  # the ABI contract

FONT_ROOTS = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/TTF"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
)


def _font(candidates: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    for name in candidates:
        for root in FONT_ROOTS:
            path = root / name
            if path.is_file():
                return ImageFont.truetype(str(path), size * SS)
    raise SystemExit(
        f"No usable font. Tried {', '.join(candidates)} under "
        f"{', '.join(str(r) for r in FONT_ROOTS)}"
    )


def sans(size: int) -> ImageFont.FreeTypeFont:
    return _font(("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"), size)


def semi(size: int) -> ImageFont.FreeTypeFont:
    return _font(("seguisb.ttf", "calibrib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"), size)


def heavy(size: int) -> ImageFont.FreeTypeFont:
    return _font(("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"), size)


def mono(size: int) -> ImageFont.FreeTypeFont:
    return _font(("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf", "Menlo.ttc"), size)


def monob(size: int) -> ImageFont.FreeTypeFont:
    return _font(("consolab.ttf", "DejaVuSansMono-Bold.ttf", "courbd.ttf"), size)


F_TITLE = heavy(46)
F_SUB = sans(21)
F_BAND = heavy(27)
F_BAND_N = heavy(30)
F_BAND_SUB = sans(18)
F_PTITLE = semi(23)
F_TAG = semi(14)
F_CODE = mono(16)
F_CODEB = monob(16)
F_TXT = sans(17)
F_KEY = semi(17)
F_SMALL = sans(15)
F_TINY = sans(13)
F_CHIP = semi(15)

LEAD_CODE = 25
LEAD_TXT = 24
LEAD_SMALL = 21
GAP = 22
BAND_GAP = 62
BAND_HEAD = 74


# --------------------------------------------------------------------------
# Primitives. All coordinates are logical; SS is applied here and nowhere else.
# --------------------------------------------------------------------------


class Sheet:
    def __init__(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        self.used = 0.0
        self.img = Image.new("RGB", (width * SS, height * SS), BG)
        self.d = ImageDraw.Draw(self.img)

    def text(
        self,
        x: float,
        y: float,
        s: str,
        font: ImageFont.FreeTypeFont,
        fill: str,
        anchor: str = "la",
    ) -> None:
        self.d.text((x * SS, y * SS), s, font=font, fill=fill, anchor=anchor)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str | None = None,
        outline: str | None = None,
        radius: float = 0,
        width: int = 1,
    ) -> None:
        box = (x * SS, y * SS, (x + w) * SS, (y + h) * SS)
        if radius:
            self.d.rounded_rectangle(
                box, radius=radius * SS, fill=fill, outline=outline, width=width * SS
            )
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=width * SS)

    def line(
        self, x1: float, y1: float, x2: float, y2: float, fill: str = BORDER, width: int = 1
    ) -> None:
        self.d.line((x1 * SS, y1 * SS, x2 * SS, y2 * SS), fill=fill, width=width * SS)

    def dashed(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        fill: str = BORDER,
        width: int = 1,
        dash: int = 9,
        space: int = 7,
    ) -> None:
        """Only the two orientations this diagram uses."""
        if y1 == y2:
            x = x1
            while x < x2:
                self.line(x, y1, min(x + dash, x2), y1, fill, width)
                x += dash + space
        else:
            y = y1
            while y < y2:
                self.line(x1, y, x1, min(y + dash, y2), fill, width)
                y += dash + space

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        fill: str = BORDER,
        width: int = 2,
        head: float = 9,
    ) -> None:
        self.line(x1, y1, x2, y2, fill, width)
        if x1 == x2:
            sign = 1 if y2 > y1 else -1
            pts = [
                (x2 * SS, y2 * SS),
                ((x2 - head * 0.6) * SS, (y2 - sign * head) * SS),
                ((x2 + head * 0.6) * SS, (y2 - sign * head) * SS),
            ]
        else:
            sign = 1 if x2 > x1 else -1
            pts = [
                (x2 * SS, y2 * SS),
                ((x2 - sign * head) * SS, (y2 - head * 0.6) * SS),
                ((x2 - sign * head) * SS, (y2 + head * 0.6) * SS),
            ]
        self.d.polygon(pts, fill=fill)

    def diamond(self, cx: float, cy: float, r: float, outline: str, width: int = 2) -> None:
        pts = [
            (cx * SS, (cy - r) * SS),
            ((cx + r) * SS, cy * SS),
            (cx * SS, (cy + r) * SS),
            ((cx - r) * SS, cy * SS),
        ]
        self.d.line([*pts, pts[0]], fill=outline, width=width * SS, joint="curve")


def wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Greedy wrap. A ``\\n`` starts a new line; an empty line stays empty."""
    out: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            trial = f"{cur} {word}".strip()
            if not cur or font.getlength(trial) <= max_w * SS:
                cur = trial
            else:
                out.append(cur)
                cur = word
        out.append(cur)
    return out


# --------------------------------------------------------------------------
# Panels
#
# Body content is a list of (kind, text) items:
#   code   monospace on a recessed strip -- literal source
#   codeb  monospace bold in the accent colour -- the headline symbol
#   key    semibold prose -- what it is
#   txt    prose -- why it is that way
#   sub    small muted prose -- the failure it prevents
#   gap    vertical space, text is the pixel count
#   rule   hairline divider
# --------------------------------------------------------------------------

Item = tuple[str, object]


def measure(items: list[Item], inner_w: float) -> float:
    h = 0.0
    for kind, text in items:
        if kind == "gap":
            h += float(str(text))
        elif kind == "rule":
            h += 15
        elif kind in {"code", "codeb"}:
            h += LEAD_CODE * len(wrap(str(text), F_CODE, inner_w - 20)) + 8
        elif kind == "key":
            h += LEAD_TXT * len(wrap(str(text), F_KEY, inner_w))
        elif kind == "txt":
            h += LEAD_TXT * len(wrap(str(text), F_TXT, inner_w))
        elif kind == "sub":
            h += LEAD_SMALL * len(wrap(str(text), F_SMALL, inner_w))
        else:
            raise ValueError(f"unknown item kind {kind!r}")
    return h


def body(s: Sheet, x: float, y: float, inner_w: float, items: list[Item], accent: str) -> float:
    cur = y
    for kind, text in items:
        if kind == "gap":
            cur += float(str(text))
        elif kind == "rule":
            s.line(x, cur + 7, x + inner_w, cur + 7, BORDER_SOFT, 1)
            cur += 15
        elif kind in {"code", "codeb"}:
            lines = wrap(str(text), F_CODE, inner_w - 20)
            s.rect(x - 8, cur - 2, inner_w + 16, LEAD_CODE * len(lines) + 8, fill=CODE_BG, radius=4)
            for ln in lines:
                s.text(
                    x + 2,
                    cur + 1,
                    ln,
                    F_CODEB if kind == "codeb" else F_CODE,
                    accent if kind == "codeb" else "#C8D6E4",
                )
                cur += LEAD_CODE
            cur += 8
        else:
            font, fill, lead = {
                "key": (F_KEY, FG, LEAD_TXT),
                "txt": (F_TXT, DIM, LEAD_TXT),
                "sub": (F_SMALL, MUTED, LEAD_SMALL),
            }[kind]
            for ln in wrap(str(text), font, inner_w):
                s.text(x, cur, ln, font, fill)
                cur += lead
    return cur


def panel_height(items: list[Item], w: float, titled: bool = True) -> float:
    return (52 if titled else 20) + measure(items, w - 44) + 22


def panel(
    s: Sheet,
    x: float,
    y: float,
    w: float,
    items: list[Item],
    title: str = "",
    tag: str = "",
    accent: str = PY,
    height: float | None = None,
) -> float:
    h = height if height is not None else panel_height(items, w, bool(title))
    s.rect(x, y, w, h, fill=PANEL, outline=BORDER, radius=10, width=1)
    s.rect(x, y, 4, h, fill=accent, radius=2)
    top = y + 20
    if title:
        s.text(x + 22, top, title, F_PTITLE, FG)
        if tag:
            s.text(x + w - 22, top + 5, tag.upper(), F_TAG, accent, anchor="ra")
        top += 34
    body(s, x + 22, top, w - 44, items, accent)
    return h


def chip_width(text: str, font: ImageFont.FreeTypeFont = F_CHIP, pad: float = 11) -> float:
    return font.getlength(text) / SS + pad * 2


def chip(
    s: Sheet, x: float, y: float, text: str, color: str, font: ImageFont.FreeTypeFont = F_CHIP
) -> float:
    w = chip_width(text, font)
    s.rect(x, y, w, 28, outline=color, radius=14, width=1)
    s.text(x + w / 2, y + 6, text, font, color, anchor="ma")
    return w


def band(s: Sheet, n: str, y: float, title: str, subtitle: str, color: str, spine: float) -> None:
    """Left rail: a marker disc, a spine reaching the next band, and a heading."""
    cx = MARGIN + 27
    if spine > 58:
        s.rect(MARGIN + 25, y + 56, 4, spine - 56, fill=BORDER_SOFT, radius=2)
    s.d.ellipse(
        ((cx - 25) * SS, y * SS, (cx + 25) * SS, (y + 50) * SS), outline=color, width=2 * SS
    )
    if n:
        s.text(cx, y + 9, n, F_BAND_N, color, anchor="ma")
    else:
        s.diamond(cx, y + 25, 11, color, 2)
    s.text(MARGIN + RAIL, y + 4, title, F_BAND, FG)
    s.text(MARGIN + RAIL, y + 34, subtitle, F_BAND_SUB, MUTED)


# ==========================================================================
# Content. Long item lists live at module scope so `build()` stays a layout
# routine and the prose can be reviewed as prose.
# ==========================================================================

SOURCE_PANELS: list[tuple[str, str, str, list[Item]]] = [
    (
        "abi.h",
        "contract",
        TEAL,
        [
            ("codeb", "#define DYNQUANT_ABI_VERSION 1"),
            (
                "code",
                "#define DYNQUANT_BITS_LIST 2, 3, 4, 8\n#define DYNQUANT_GROUP_SIZE_ALIGNMENT 32",
            ),
            (
                "txt",
                "The one number that must not drift. Also a constexpr function, so the value is baked "
                "into the .so and a stale header on the reader's side cannot fake it.",
            ),
            (
                "sub",
                "Bump on a schema change, a packed-layout change, or a change to the affine convention "
                "w \u2248 q\u00b7scale + offset. Never for an optimisation that keeps all three fixed -- "
                "those are exactly the changes that should ship without a reinstall.",
            ),
            ("rule", None),
            (
                "sub",
                "GROUP_SIZE_ALIGNMENT 32 is why the layout admits a kernel at all: group_size % 32 == 0 "
                "makes group_size \u00d7 bits a whole number of 32-bit words at every width above, so a "
                "group never begins mid-word and every shift is a compile-time constant.",
            ),
            ("rule", None),
            ("key", "BITS_LIST is checked against Python too."),
            (
                "sub",
                "The same test asserts it equals dynquant.constants.BIT_OPTIONS. So a fifth width cannot be "
                "half-added: the set the C++ templates are instantiated over and the set the format "
                "validator accepts come from one line, or the build is red.",
            ),
        ],
    ),
    (
        "common.cuh",
        "device utils",
        CUDA,
        [
            ("codeb", "DYNQUANT_CHECK_LAUNCH()"),
            (
                "txt",
                "Used immediately after every <<<>>>. Launch errors are asynchronous, so a bad config or "
                "an out-of-bounds write otherwise surfaces at the next synchronising call, somewhere "
                "unrelated in user code.",
            ),
            ("sub", "Turns \u201closs went NaN three steps later\u201d into a line number."),
            ("rule", None),
            ("code", "kDefaultBlock = 256\ngrid_for(n)  // capped at 65535"),
            (
                "txt",
                "256 threads is enough warps to hide global latency and small enough that a block's "
                "register budget does not cap occupancy on sm_75. The cap keeps the grid resident, so it "
                "amortises launch cost and keeps L2 warm.",
            ),
            (
                "sub",
                "Throws std::runtime_error, not a bespoke type: anything not derived from std::exception "
                "reaches Python as \u201cunknown exception\u201d with the cause discarded.",
            ),
            ("rule", None),
            ("code", "constexpr kWarpSize = 32\n__host__ __device__ ceil_div(a, b)"),
            (
                "sub",
                "constexpr and both-sided, so the host computes the grid and the kernel computes its bounds "
                "check from the same expression. Two copies of that formula is how a grid-stride loop ends "
                "up one element short on exactly the sizes no test uses.",
            ),
        ],
    ),
    (
        "bindings.cpp",
        "host \u00b7 schemas",
        TORCH,
        [
            ("codeb", "TORCH_LIBRARY(dynquant, m)"),
            ("code", 'm.def("probe_axpy(Tensor x, Tensor y,\n      float alpha) -> Tensor")'),
            (
                "txt",
                "Registered in the dispatcher, not exposed through pybind. An op with a schema is one "
                "torch.compile can trace, autograd and AMP see a real node, and CUDA Graph capture "
                "works. A pybind function is opaque to all of it.",
            ),
            (
                "sub",
                "Which means a graph break at every quantized Linear -- that is, at every layer.",
            ),
            ("rule", None),
            ("key", "TORCH_LIBRARY_IMPL(\u2026, CPU, m)"),
            (
                "txt",
                "Three CPU reference impls, accumulating in fp32 to match the device kernel rounding for "
                "rounding. Not for speed: they are the oracle the CUDA path is tested against.",
            ),
            (
                "sub",
                "And they are what lets a GPU-less runner exercise registration, schemas and fake impls, "
                "so a packaging bug surfaces minutes after the push rather than at the end of a GPU "
                "matrix.",
            ),
            ("rule", None),
            ("code", "PyObject* PyInit__C(void)"),
            (
                "sub",
                "A bare CPython module with no methods; the static initializers above are the whole "
                "mechanism. Not PYBIND11_MODULE: ~1MB of template instantiation and a second ABI to keep "
                "in step with torch's, in exchange for nothing.",
            ),
        ],
    ),
    (
        "probe_cuda.cu",
        "nvcc \u00b7 thrust",
        CUDA,
        [
            ("codeb", "axpy_kernel<<<grid_for(n), 256>>>"),
            (
                "txt",
                "Grid-stride, fp32 accumulate even for half inputs. Proves nvcc compiled device code, "
                "the fatbin holds a cubin this GPU can execute, and a launch reaches it.",
            ),
            (
                "sub",
                "Fails when CMAKE_CUDA_ARCHITECTURES misses the device. Same kernel shape the decode GEMV "
                "will use, so it validates the launch-geometry helper too.",
            ),
            ("rule", None),
            ("code", "thrust::transform_reduce(\n  thrust::cuda::par.on(currentStream), \u2026)"),
            (
                "txt",
                "Thrust and CUB headers resolve and their device code links -- dispatched onto torch's "
                "current stream, not the default one.",
            ),
            (
                "sub",
                "Getting that wrong is silent: the reduction races with whatever produced the input and "
                "reads partially-written memory only under load. Reduces to fp32, because summing 10\u2078 "
                "halves in half precision saturates the mantissa long before the end.",
            ),
            ("rule", None),
            ("code", "compiled_architectures()\n  \u2190 __CUDA_ARCH_LIST__"),
            (
                "sub",
                "So doctor can say \u201cno cubin for your sm_120, the first launch will JIT from PTX and "
                "stall for seconds\u201d instead of the user meeting it as a mystery warm-up cost. CUDA "
                "< 11.5 does not define the list, and reports \u201cunknown\u201d rather than claiming "
                "coverage it cannot verify.",
            ),
        ],
    ),
    (
        "probe_cublaslt.cpp",
        "cublaslt",
        CUDA,
        [
            ("codeb", "cublasLtMatmul(\u2026)"),
            (
                "txt",
                "Row-major C = A\u00b7B asked of a column-major library as D = B\u1d40\u00b7A\u1d40. Every "
                "transpose is free, being just the other reading of the same bytes -- no copies, and it "
                "is the same mapping the fused kernel will use.",
            ),
            ("rule", None),
            ("code", "CUBLAS_COMPUTE_32F"),
            (
                "txt",
                "Not 16F. fp16 accumulation loses roughly three mantissa bits per doubling of K and would "
                "bury the quantization error budget in the noise of the GEMM itself.",
            ),
            ("rule", None),
            ("key", "Workspace from torch's caching allocator"),
            (
                "txt",
                "A raw cudaMalloc here would fragment the allocator's arena and -- worse -- is illegal "
                "during CUDA Graph capture, which P8 depends on.",
            ),
            (
                "sub",
                "Why this is in P0 and not P7: cuBLASLt is a separate library with its own import lib, "
                "and whether it resolves depends on the toolkit version, on FindCUDAToolkit defining the "
                "target, and on Windows whether the DLL sits next to torch's. Finding that out in P7 "
                "means redesigning the build with three phases stacked on it.",
            ),
        ],
    ),
]

BUILD_PANELS: list[tuple[str, str, str, list[Item]]] = [
    (
        "Configure",
        "cmakelists.txt",
        BUILD,
        [
            ("code", "find_package(Python 3.10 REQUIRED\n  COMPONENTS Development.Module)"),
            ("key", "Ask the interpreter, never a path"),
            (
                "code",
                'execute_process(${Python_EXECUTABLE} -c\n "print(torch.utils.cmake_prefix_path)")',
            ),
            (
                "txt",
                "Hardcoding a prefix, or trusting the caller's CMAKE_PREFIX_PATH, is how an extension "
                "ends up linked against a different libtorch than the one it runs with.",
            ),
            (
                "sub",
                "Which manifests as an undefined-symbol ImportError, or worse, as a crash inside ATen. "
                "If it cannot import torch at all it fails here with the --no-build-isolation advice, "
                "rather than building something unusable.",
            ),
            ("rule", None),
            ("code", "find_package(Torch REQUIRED)"),
            (
                "txt",
                "Its flags are honoured wholesale, _GLIBCXX_USE_CXX11_ABI included. A mixed ABI against "
                "libtorch fails on std::string arguments and reads as missing symbols.",
            ),
            ("rule", None),
            ("code", "include(CheckLanguage)\ncheck_language(CUDA)"),
            ("txt", "The fork, resolved four ways \u2192"),
            ("rule", None),
            ("code", "Python_add_library(_C MODULE WITH_SOABI)"),
            (
                "sub",
                "WITH_SOABI puts the interpreter tag in the filename, so extensions for two CPythons can "
                "coexist and pip cannot install the wrong one over the right one.",
            ),
            ("rule", None),
            ("code", "[build-system] scikit-build-core"),
            (
                "sub",
                "So pip install . and python -m build drive this same configure. There is no second, "
                "differently-configured build path that only CI ever takes -- which is the usual way a "
                "release wheel ends up unlike every wheel that was tested.",
            ),
        ],
    ),
    (
        "The CUDA fork",
        "four outcomes",
        WARN,
        [
            ("codeb", "nvcc + CUDA torch \u2192 device code"),
            (
                "sub",
                "DYNQUANT_WITH_CUDA=1, and probe_cuda.cu plus probe_cublaslt.cpp join the source list.",
            ),
            ("gap", 6),
            ("codeb", "nvcc + CPU torch \u2192 WARNING"),
            (
                "sub",
                "A toolkit is present but this torch cannot link CUDA libraries. Downgrades to CPU-only "
                "rather than emitting a wheel that cannot load.",
            ),
            ("gap", 6),
            ("codeb", "major mismatch \u2192 FATAL_ERROR"),
            (
                "sub",
                "A cu12 extension beside a cu11 torch puts two libcudart in one process. The failure is a "
                "segfault inside a stream call, with nothing pointing here -- so it is refused at "
                "configure time instead.",
            ),
            ("gap", 6),
            ("codeb", "no nvcc \u2192 CPU-only extension"),
            (
                "sub",
                "The same ops, CPU impls only. Not a consolation prize: it is what makes the sdist "
                "installable without a toolkit, and what lets a free CI runner catch schema, "
                "registration and pybind errors in minutes rather than queueing for a GPU.",
            ),
            ("gap", 6),
            ("codeb", "\u2026 unless DYNQUANT_REQUIRE_CUDA=ON"),
            (
                "sub",
                "Set for every release build, so a toolkit that quietly failed detection becomes a build "
                "failure instead of a wheel advertised as containing kernels that contains none.",
            ),
            ("rule", None),
            (
                "txt",
                "Whichever branch was taken is recorded in the build stamp below, so the wheel can be "
                "asked afterwards what it actually is.",
            ),
            (
                "sub",
                "All four are decided at configure time, none at import time. A wheel either contains "
                "device code or it does not, and it can say which without being run on a GPU.",
            ),
        ],
    ),
    (
        "Codegen and link",
        "nvcc \u00b7 fatbin",
        CUDA,
        [
            ("key", "Architecture ladder"),
            (
                "code",
                "75-real 80-real 86-real\n+ 89-real 90-real    (toolkit \u2265 11.8)\n"
                "+ 100-real 120-real  (toolkit \u2265 12.8)\n+ one -virtual tail",
            ),
            (
                "txt",
                "Cubins for the cards people actually have, so launches are instant; one trailing PTX "
                "target so a GPU newer than this toolkit still runs -- the driver JITs on first launch, "
                "seconds, once, cached.",
            ),
            ("sub", "Without the PTX tail that GPU gets \u201cno kernel image is available\u201d."),
            ("rule", None),
            (
                "code",
                "-O3  -lineinfo\n--expt-relaxed-constexpr\n--expt-extended-lambda\n"
                "-U__CUDA_NO_HALF_OPERATORS__",
            ),
            (
                "sub",
                "-lineinfo costs nothing at runtime and is what makes compute-sanitizer and Nsight name a "
                "line instead of an address. The -U flags are what torch's half types need in device "
                "code.",
            ),
            ("rule", None),
            ("code", "# NOT CUDA_SEPARABLE_COMPILATION\n# NOT -use_fast_math"),
            (
                "txt",
                "Relocatable device code would let device functions be called across translation units, "
                "which never happens here, and would block nvcc from inlining the n-bit unpack helpers "
                "into the GEMV inner loop -- the one place inlining decides memory-bound versus "
                "instruction-bound.",
            ),
            (
                "sub",
                "Fast math implies -ftz=true and approximate division, and a dequantized weight near the "
                "bottom of a 2-bit group's range is exactly the denormal case. Accuracy measured with it "
                "on is not reproducible, so the flag exists and warns loudly when set.",
            ),
            ("rule", None),
            (
                "code",
                "target_link_libraries(_C PRIVATE\n  torch torch_python\n"
                "  CUDA::cudart CUDA::cublas\n  CUDA::cublasLt)",
            ),
            (
                "sub",
                "cublasLt falls back to find_library when an older FindCUDAToolkit does not define the "
                "target even though the library is sitting next to libcublas.",
            ),
        ],
    ),
]

STAMP_ITEMS: list[Item] = [
    ("key", "The ABI number is parsed out of the header, never retyped."),
    (
        "code",
        'string(REGEX MATCH "#define[ \\t]+DYNQUANT_ABI_VERSION[ \\t]+([0-9]+)" \u2026)'
        "   \u2192   configure_file(cmake/_build_info.py.in)",
    ),
    (
        "txt",
        "Repeating the number in CMake would create a third place for it to drift, and a stale stamp "
        "would make the loader's handshake pass while the binary disagreed. The generated _build_info.py "
        "records ABI_VERSION, TORCH_VERSION, TORCH_CUDA_VERSION, CUDA_VERSION, WITH_CUDA, "
        "CUDA_ARCHITECTURES, CXX_COMPILER and BUILD_TYPE -- which is the only reliable way to explain a "
        "load failure in terms of what to install instead.",
    ),
]

MATRIX_FOOTNOTES = (
    "PyPI rejects local versions outright, so exactly one cell is built plain and published there and every "
    "other variant is attached to the GitHub Release, which doubles as a --find-links index. A user on a "
    "non-default torch either points pip at that index or builds the sdist, which compiles against their own "
    "torch -- the only configuration correct by construction.\n"
    "\n"
    "Which cell is the plain one is a user-facing decision, not a tidy one. Through 0.5.2 it was cu126 / "
    "torch 2.7, and pip satisfied that narrow pin by pulling a working torch 2.13 back to 2.7.1 -- which "
    "supports up to sm_90. Every Blackwell owner got a clean install, a successful dlopen, and a dead first "
    "tensor. The plain cell now follows the torch a bare pip install resolves to, and CI asserts the shipped "
    "cubins cover sm_80 through sm_120 -- checkable with no GPU, since the arch list is baked into the "
    "binary by nvcc.\n"
    "\n"
    "A toolkit drops architectures as well as adding them; CUDA 13 removes Maxwell through Volta. So the "
    "requested list is filtered against nvcc --list-gpu-arch at configure time rather than predicted from a "
    "version ladder, because asking for one nvcc no longer knows is a hard fatal that would fail a release "
    "build for a reason having nothing to do with this code.\n"
    "\n"
    "The manylinux CUDA images are third-party and are pinned by digest rather than tag: a tag can be moved, "
    "and whoever controls it controls the compiler that produces the binaries users run. Building our own "
    "remains the better answer.\n"
    "\n"
    "macOS gets no cell and needs none -- there is no CUDA on it. It installs dynquant-core and runs the "
    "torch backend, which is the entire quantization path; only accelerated inference is missing, and it "
    "was never available there to miss."
)

AUDIT_ITEMS: list[Item] = [
    ("key", "auditwheel must bundle almost nothing"),
    (
        "code",
        "--exclude libtorch.so  libtorch_cpu.so\n--exclude libc10.so    libc10_cuda.so\n"
        "--exclude libcudart.so.12  libcublas*.so.12\n--exclude libcuda.so.1",
    ),
    (
        "txt",
        "Vendoring libtorch would give a ~2GB wheel with a second copy of it loaded into the process "
        "next to the user's: two ATen registries, duplicate dispatch keys, and crashes with no line "
        "pointing here. torch's own libraries are found at runtime because torch is imported first.",
    ),
    ("rule", None),
    ("key", "Verified without importing it"),
    (
        "txt",
        "The build container has no GPU and no driver, so importing the extension would fail on "
        "libcuda.so.1 whether or not the wheel is correct. So the zip is read instead:",
    ),
    (
        "code",
        "dynquant_kernels/_C.*.so       present\n_build_info.py                 present\n"
        "WITH_CUDA: Final[bool] = bool(1)\nno libtorch / libc10 / libcud* inside",
    ),
    (
        "sub",
        "The third line is the one that matters: it is what catches a release build in which CUDA "
        "detection silently failed. Numerics are the GPU job's business; this job's business is that the "
        "wheel contains what it claims.",
    ),
    ("rule", None),
    ("key", "Publication is gated three ways"),
    (
        "txt",
        "A version tag, a PUBLISH_ENABLED repository variable, and a protected environment with required "
        "reviewers. Nothing fires until someone deliberately sets the variable and approves the run.",
    ),
    (
        "sub",
        "Because this is the one irreversible step in the repository -- a filename cannot be reused on "
        "PyPI even after a delete -- and because this source began as a confidential reviewer copy under "
        "double-blind review.",
    ),
    ("rule", None),
    (
        "sub",
        "Windows wheels are continue-on-error by design: MSVC + nvcc on a hosted runner has several ways "
        "to fail that have nothing to do with this code, and Windows users are served by the sdist and "
        "the pure-Python core. A failure there must not fail a release.",
    ),
]

GATES: list[tuple[str, str, str, str, str]] = [
    (
        "torch first",
        "import torch  # before _C",
        "The extension links libtorch, and on Windows torch's __init__ is what puts the CUDA DLL directory on "
        "the search path.",
        "Done inside the loader rather than trusted to the caller, so the order cannot be got wrong by an "
        "import that looks innocuous.",
        CUDA,
    ),
    (
        "the shared object",
        'importlib.import_module("dynquant_kernels._C")',
        "An ImportError here is a linker message. It is translated, never re-raised raw.",
        "torch minor differs \u2192 names the exact wheel to install. CUDA major differs \u2192 names the "
        "mismatch. Windows \u2192 points at the DLL search path. The build stamp is what makes each of these "
        "sayable at all.",
        TORCH,
    ),
    (
        "the handshake",
        "int(torch.ops.dynquant.abi_version()) == 1",
        "The one failure mode that yields plausible wrong numbers instead of an exception.",
        "An old kernel reading a new packed layout still returns a tensor of the right shape and dtype, "
        "decoded at the wrong shift. So it fails closed and says to upgrade both together.",
        TEAL,
    ),
    (
        "ops actually registered",
        "except Exception \u2192 unavailable",
        "The .so loaded but its static initializers did not register the schemas.",
        "Reported as a corrupt build. Registration is the entire mechanism -- importing the module is what "
        "makes torch.ops.dynquant.* exist -- so an extension that imported is not yet an extension that "
        "works.",
        FAIL,
    ),
    (
        "device code present",
        "torch.ops.dynquant.built_with_cuda()",
        "A working extension with no device code in it. Whether that is a fault depends on the machine, so "
        "the message does too.",
        "GPU present \u2192 a fault: you installed the sdist somewhere no toolkit was visible at build time. "
        "No GPU \u2192 expected=True, the designed steady state of the CPU-only CI job, and the one outcome "
        "STRICT mode must not turn into an error.",
        WARN,
    ),
    (
        "a device to run on",
        "torch.cuda.is_available()",
        "Kernels installed; the driver or CUDA_VISIBLE_DEVICES says otherwise.",
        "Quantizing works without a GPU and only accelerated inference needs one, so the message says that "
        "rather than implying the install is broken.",
        WARN,
    ),
    (
        "fake impls",
        "import dynquant_kernels.ops\n  \u2192 torch.library.register_fake",
        "Last, because register_fake needs schemas that exist only once _C is in.",
        "A meta kernel computes shapes and dtypes, allocating nothing. Without one, torch.compile cannot "
        "infer a quantized Linear's output shape and breaks the graph at every layer -- 40 breaks on a "
        "40-layer model, which costs more than the kernel saves.",
        PY,
    ),
]

BACKENDS: list[tuple[str, str, str, list[Item]]] = [
    (
        "cuda",
        "first choice",
        CUDA,
        [
            ("code", "dynquant_kernels._C\nkernel ABI 1  ·  sm_75 … sm_120"),
            ("key", "Weights stay packed in VRAM."),
            (
                "txt",
                "Unpacked in registers on the way into the multiply, never materialised as fp16. This is "
                "the only path on which peak memory equals the manifest size rather than the fp16 size.",
            ),
            (
                "sub",
                "Chosen when, and only when, all seven gates above passed. P6 is the phase that makes it "
                "worth choosing -- today the probes load and run, and the compute kernels they de-risk "
                "are not written yet.",
            ),
        ],
    ),
    (
        "triton",
        "portability",
        PY,
        [
            ("code", "pip install dynquant[triton]\nno prebuilt binary, no matrix cell"),
            ("key", "A fallback, never the answer."),
            (
                "txt",
                "Same semantics, JIT compiled at first call. Covers ROCm, and NVIDIA parts newer than any "
                "wheel that was shipped -- the two cases a binary matrix cannot reach by construction.",
            ),
            (
                "sub",
                "Costs a compile on first use per (bits, group_size, shape) and gives up the PTX-level "
                "tricks -- lop3, prmt, cp.async -- that the CUDA path exists to use. Correct, portable, "
                "and slower.",
            ),
        ],
    ),
    (
        "torch",
        "oracle · today",
        TORCH,
        [
            ("code", "always importable\nno compiler, no driver, no GPU"),
            ("key", "The reference, and the honest floor."),
            (
                "txt",
                "Dequantizes to fp16 to compute, so it saves nothing at run time and is not fast. It is "
                "what every other backend is checked against, and it runs on a laptop.",
            ),
            (
                "sub",
                "Which is why quantizing is already useful while inference is not: the checkpoint this "
                "path writes is correct, portable, and smaller on disk. Only the speed is pending, and "
                "the format was designed so that landing it breaks no artifact.",
            ),
        ],
    ),
]

OVERRIDE_ITEMS: list[Item] = [
    ("code", "DYNQUANT_BACKEND=cuda|triton|torch"),
    ("key", "An override that cannot be honoured raises."),
    (
        "txt",
        "Honouring the request or failing are the only two safe outcomes. Silently substituting torch for "
        "a requested cuda turns a benchmark into a number that merely looks like a result, and a "
        "memory-savings claim into a false one.",
    ),
    (
        "sub",
        "Detection itself never raises, though -- a broken install is exactly when doctor needs to run. "
        "And every rejected backend keeps the sentence saying why, so nobody has to guess whether they "
        "are missing a wheel, a driver, or a GPU.",
    ),
    ("rule", None),
    ("code", "dynquant doctor"),
    (
        "txt",
        "Environment report, backend selection with a reason per rejection, then the probes run as a "
        "numerical self-check: pack/unpack bijection at every width, and measured quantization error "
        "against theory.",
    ),
    (
        "sub",
        "Because a wrong-ABI kernel returns tensors of the right shape and dtype full of wrong numbers -- "
        "indistinguishable from quantization simply being lossy, and so never reported as a bug.",
    ),
]

ABI_ITEMS: list[Item] = [
    ("key", "One number. Three declarations. One lint."),
    ("code", "csrc/include/dynquant/abi.h\n  #define DYNQUANT_ABI_VERSION 1"),
    ("code", "dynquant_kernels/__init__.py\n  ABI_VERSION: int = 1"),
    ("code", "dynquant/_version.py\n  KERNEL_ABI_VERSION = 1"),
    (
        "txt",
        "Declared three times deliberately. The kernels wheel must be installable and diagnosable "
        "without dynquant, and core must detect an ABI mismatch without paying to load the shared "
        "object. Importing across the two would recreate exactly the coupling the split exists to "
        "remove -- and would mean a core patch release could not be installed without a matching "
        "kernels rebuild, for a change that touched no binary at all.",
    ),
    ("rule", None),
    ("code", "tests/test_abi.py  \u2014 parses the header"),
    (
        "sub",
        "A source-text lint, so it runs on a machine where no extension is installed and asserts all "
        "three declarations are the same number. Two of the 497 tests skip without dynquant_kernels; a "
        "separate CPU-only CI job imports it for real.",
    ),
]

DERISK_ITEMS: list[Item] = [
    ("key", "What five probes bought, phase by phase"),
    (
        "txt",
        "None of this is throwaway. Each probe is the load-bearing dependency of a later phase, moved to "
        "where a failure costs an afternoon instead of a redesign with three phases stacked on top of "
        "it.",
    ),
    ("gap", 4),
    ("code", "probe_axpy    \u2192  P6  decode GEMV <BITS, GROUP_SIZE>"),
    (
        "sub",
        "The grid-stride geometry, fp32 accumulation, and the launch-config helper. Decode is "
        "memory-bound and dominates real serving, so this is where the time actually goes.",
    ),
    ("code", "probe_reduce  \u2192  P5  fused quantizer  \u00b7  P8  MoE permute"),
    (
        "sub",
        "Thrust and CUB linkage on torch's stream. P5's reduce_by_key per-group min/max plus a "
        "register-resident clipping search replaces eight full-tensor torch reconstructions; P8's "
        "stable_sort_by_key groups tokens by expert so 128 experts is one launch, not 128.",
    ),
    ("code", "probe_gemm    \u2192  P7  prefill, phase A"),
    (
        "sub",
        "The descriptor setup, the row-major mapping and the heuristic query are exactly what "
        "dequant_tile \u2192 cublasLtMatmul needs -- the correctness-guaranteed prefill path that ships "
        "before the fused CUTLASS kernel exists.",
    ),
    ("rule", None),
    ("key", "And what is honestly not there yet."),
    (
        "txt",
        "dynquant doctor reports backend=torch today, and it means it: quantization is complete and "
        "correct on that path, accelerated inference is not. The claim this pipeline exists to make -- "
        "peak VRAM equal to the manifest size rather than the fp16 size -- lands with P6.",
    ),
    (
        "sub",
        "And the pipeline on this page has not itself been compiled: the machine it was written on has no "
        "nvcc and no GPU. The CPU-only extension build, the wheel matrix and every launch above are "
        "asserted by CI jobs that have not run yet. The P0 gate -- pip install dynquant on a clean Linux "
        "GPU box, importing the extension and passing the self-check -- is open. Which is the point of "
        "writing five probes first: when it is run, a failure will name one dependency.",
    ),
]


# ==========================================================================
# Layout
# ==========================================================================


def build() -> Sheet:
    s = Sheet(WIDTH, 5400)
    bx = MARGIN + RAIL
    bw = CONTENT - RAIL

    # ---------------------------------------------------------------- header
    s.rect(0, 0, WIDTH, 6, fill=CUDA)
    s.text(
        MARGIN,
        48,
        "How the DynQuant CUDA kernel gets built, shipped, loaded and chosen",
        F_TITLE,
        FG,
    )
    subtitle = (
        "The P0 binary pipeline exactly as it stands: five source files under csrc/, one shared object, and "
        "seven gates between nvcc and a kernel that is allowed to run. Every box below is a file you can "
        "open. The compute kernels of P5-P8 appear only in the last band -- as the thing this pipeline "
        "exists to de-risk, not as work that is done."
    )
    for i, ln in enumerate(wrap(subtitle, F_SUB, CONTENT - 520)):
        s.text(MARGIN, 112 + i * 27, ln, F_SUB, DIM)

    chips = (
        ("P0 \u00b7 GPU gate not yet run", WARN),
        ("kernel ABI 1", TEAL),
        ("497 tests, 0 GPUs needed", PY),
    )
    x = WIDTH - MARGIN - sum(chip_width(t) + 10 for t, _ in chips) + 10
    for text, color in chips:
        x += chip(s, x, 116, text, color) + 10

    # ------------------------------------------------------------- 1: source
    y = 236
    pw = (bw - GAP * 4) / 5
    h = max(panel_height(items, pw) for _, _, _, items in SOURCE_PANELS)
    band(
        s,
        "1",
        y,
        "Source",
        "packages/dynquant-kernels/csrc/ \u2014 two headers, three translation units",
        CUDA,
        BAND_HEAD + h + BAND_GAP,
    )
    for i, (title, tag, accent, items) in enumerate(SOURCE_PANELS):
        panel(s, bx + i * (pw + GAP), y + BAND_HEAD, pw, items, title, tag, accent, height=h)

    # -------------------------------------------------------------- 2: build
    y += BAND_HEAD + h + BAND_GAP
    pw2 = (bw - GAP * 2) / 3
    h2 = max(panel_height(items, pw2) for _, _, _, items in BUILD_PANELS)
    hs = panel_height(STAMP_ITEMS, bw)
    band(
        s,
        "2",
        y,
        "Build",
        "CMake \u2265 3.26 driven by scikit-build-core \u2014 one project, two modes",
        BUILD,
        BAND_HEAD + h2 + 20 + hs + BAND_GAP,
    )
    for i, (title, tag, accent, items) in enumerate(BUILD_PANELS):
        panel(s, bx + i * (pw2 + GAP), y + BAND_HEAD, pw2, items, title, tag, accent, height=h2)
        if i:
            ax = bx + i * (pw2 + GAP)
            s.arrow(
                ax - GAP + 2, y + BAND_HEAD + h2 / 2, ax - 2, y + BAND_HEAD + h2 / 2, MUTED, 2, 8
            )
    panel(
        s,
        bx,
        y + BAND_HEAD + h2 + 20,
        bw,
        STAMP_ITEMS,
        "Build stamp",
        "configure_file \u2192 dynquant_kernels/_build_info.py",
        TEAL,
    )

    # --------------------------------------------------------------- 3: ship
    y += BAND_HEAD + h2 + 20 + hs + BAND_GAP
    tw = bw * 0.52
    rw = bw - tw - GAP
    # Newest first, and that ordering is the point rather than a preference: the plain
    # cell has to pair with the torch a bare `pip install torch` resolves to, because
    # that is the torch that will be sitting next to the kernels in a fresh environment.
    rows = (
        (("cu130 / torch 2.13", "13.0", "2.13.0", "3.10 \u2013 3.13", "(plain) \u2192 PyPI"), CUDA),
        (("cu128 / torch 2.8", "12.8", "2.8.0", "3.10 \u2013 3.13", "+cu128torch28"), DIM),
        (("cu126 / torch 2.7", "12.6", "2.7.1", "3.10 \u2013 3.13", "+cu126torch27"), DIM),
        (("cu126 / torch 2.6", "12.6", "2.6.0", "3.10 \u2013 3.13", "+cu126torch26"), DIM),
    )
    rh = 42
    foot = wrap(MATRIX_FOOTNOTES, F_SMALL, tw - 44)
    th = 62 + 34 + rh * len(rows) + 14 + len(foot) * LEAD_SMALL + 24
    h3 = max(th, panel_height(AUDIT_ITEMS, rw))
    band(
        s,
        "3",
        y,
        "Ship",
        "cibuildwheel \u2014 a compiled extension is valid only beside the torch minor it was linked against",
        BUILD,
        BAND_HEAD + h3 + BAND_GAP,
    )
    ty = y + BAND_HEAD
    panel(
        s,
        bx + tw + GAP,
        ty,
        rw,
        AUDIT_ITEMS,
        "Wheel hygiene",
        "manylinux_2_34_x86_64",
        TORCH,
        height=h3,
    )

    s.rect(bx, ty, tw, h3, fill=PANEL, outline=BORDER, radius=10, width=1)
    s.rect(bx, ty, 4, h3, fill=BUILD, radius=2)
    s.text(bx + 22, ty + 20, "The matrix", F_PTITLE, FG)
    n_wheels = sum(
        int(hi.rsplit(".", 1)[1]) - int(lo.rsplit(".", 1)[1]) + 1
        for lo, hi in ([p.strip() for p in cells[3].split("\u2013")] for cells, _ in rows)
    )
    s.text(bx + tw - 22, ty + 25, f"{n_wheels} WHEELS PER RELEASE", F_TAG, BUILD, anchor="ra")
    hy = ty + 62
    cxs: list[float] = []
    acc = bx + 22
    for frac in (0.26, 0.15, 0.16, 0.20, 0.23):
        cxs.append(acc)
        acc += (tw - 44) * frac
    for c, head in zip(cxs, ("cell", "CUDA", "torch", "CPython", "local version"), strict=True):
        s.text(c, hy, head.upper(), F_TAG, MUTED)
    s.line(bx + 22, hy + 24, bx + tw - 22, hy + 24, BORDER, 1)
    for r, (cells, tint) in enumerate(rows):
        ry = hy + 34 + r * rh
        if r % 2 == 0:
            s.rect(bx + 14, ry - 6, tw - 28, rh - 4, fill=PANEL_ALT, radius=5)
        for c, cell in zip(cxs, cells, strict=True):
            literal = cell[0].isdigit() or cell[0] in "+("
            s.text(c, ry, cell, F_CODE if literal else F_TXT, tint)
    fy = hy + 34 + rh * len(rows) + 14
    for i, ln in enumerate(foot):
        s.text(bx + 22, fy + i * LEAD_SMALL, ln, F_SMALL, MUTED)

    # --------------------------------------------------------------- 4: load
    y += BAND_HEAD + h3 + BAND_GAP
    frac = (0.30, 0.26, 0.44)
    inner = bw - 44
    heights = [
        max(
            LEAD_CODE * len(wrap(code, F_CODE, inner * frac[0] - 40)) + 26,
            LEAD_TXT * len(wrap(guard, F_TXT, inner * frac[1] - 30)) + 8,
            LEAD_SMALL * len(wrap(msg, F_SMALL, inner * frac[2] - 30)) + 8,
        )
        + 22
        for _, code, guard, msg, _ in GATES
    ]
    h4 = 62 + 34 + sum(heights) + 12
    band(
        s,
        "4",
        y,
        "Load",
        "dynquant_kernels/_loader.py \u2014 importing it never raises; every branch names a remedy",
        TORCH,
        BAND_HEAD + h4 + BAND_GAP,
    )
    ly = y + BAND_HEAD
    s.rect(bx, ly, bw, h4, fill=PANEL, outline=BORDER, radius=10, width=1)
    s.rect(bx, ly, 4, h4, fill=TORCH, radius=2)
    s.text(
        bx + 22,
        ly + 20,
        "Seven gates, in order. Any one failing degrades to the torch backend with a diagnostic \u2014 never "
        "a traceback out of import dynquant.",
        F_PTITLE,
        FG,
    )
    s.text(bx + bw - 22, ly + 25, "FAILS CLOSED", F_TAG, TORCH, anchor="ra")
    hy = ly + 62
    for label, off in (
        ("CHECK", 0.0),
        ("WHAT IT GUARDS", frac[0]),
        ("WHAT THE USER IS TOLD", frac[0] + frac[1]),
    ):
        s.text(bx + 22 + inner * off, hy, label, F_TAG, MUTED)
    s.line(bx + 22, hy + 24, bx + bw - 22, hy + 24, BORDER, 1)
    gy = hy + 34
    for i, ((label, code, guard, msg, accent), gh) in enumerate(zip(GATES, heights, strict=True)):
        if i % 2 == 0:
            s.rect(bx + 14, gy, bw - 28, gh - 4, fill=PANEL_ALT, radius=5)
        s.rect(bx + 14, gy + 6, 3, gh - 16, fill=accent, radius=2)
        c0 = bx + 30
        s.text(c0, gy + 8, f"{i + 1}. {label}", F_KEY, FG)
        code_lines = wrap(code, F_CODE, inner * frac[0] - 40)
        cy = gy + 8 + LEAD_TXT
        s.rect(
            c0 - 6,
            cy - 2,
            inner * frac[0] - 34,
            LEAD_CODE * len(code_lines) + 8,
            fill=CODE_BG,
            radius=4,
        )
        for ln in code_lines:
            s.text(c0, cy + 1, ln, F_CODE, accent)
            cy += LEAD_CODE
        cy = gy + 8
        for ln in wrap(guard, F_TXT, inner * frac[1] - 30):
            s.text(bx + 22 + inner * frac[0], cy, ln, F_TXT, DIM)
            cy += LEAD_TXT
        cy = gy + 8
        for ln in wrap(msg, F_SMALL, inner * (frac[0] + frac[1]) * 0 + inner * frac[2] - 30):
            s.text(bx + 22 + inner * (frac[0] + frac[1]), cy, ln, F_SMALL, MUTED)
            cy += LEAD_SMALL
        gy += gh
        if i < len(GATES) - 1:
            s.dashed(bx + 22, gy - 2, bx + bw - 22, gy - 2, BORDER_SOFT, 1, 6, 6)

    # ----------------------------------------------------------- 5: dispatch
    y += BAND_HEAD + h4 + BAND_GAP
    dw = (bw * 0.66 - GAP * 2) / 3
    ow = bw - bw * 0.66 - 8
    h5 = max(
        max(panel_height(items, dw) for *_, items in BACKENDS),
        panel_height(OVERRIDE_ITEMS, ow),
    )
    band(
        s,
        "5",
        y,
        "Dispatch",
        "dynquant/runtime/backend.py \u2014 preference order, and an override that refuses rather than "
        "substitutes",
        PY,
        BAND_HEAD + h5 + BAND_GAP,
    )
    dy = y + BAND_HEAD
    for i, (name, tag, accent, items) in enumerate(BACKENDS):
        panel(s, bx + i * (dw + GAP), dy, dw, items, name, tag, accent, height=h5)
        if i:
            ax = bx + i * (dw + GAP)
            s.arrow(ax - GAP + 2, dy + h5 / 2, ax - 2, dy + h5 / 2, MUTED, 2, 8)
    panel(
        s,
        bx + bw * 0.66 + 8,
        dy,
        ow,
        OVERRIDE_ITEMS,
        "Never silently",
        "explicit \u00b7 reported",
        WARN,
        height=h5,
    )

    # ------------------------------------------------- closing: ABI + phases
    y += BAND_HEAD + h5 + BAND_GAP
    lw = bw * 0.44
    rw2 = bw - lw - GAP
    h6 = max(panel_height(ABI_ITEMS, lw), panel_height(DERISK_ITEMS, rw2))
    band(
        s,
        "",
        y,
        "The invariant, and what comes next",
        "why the number is declared three times, and why five probes were written before one real kernel",
        TEAL,
        BAND_HEAD + h6,
    )
    panel(
        s,
        bx,
        y + BAND_HEAD,
        lw,
        ABI_ITEMS,
        "The ABI thread",
        "no runtime coupling",
        TEAL,
        height=h6,
    )
    panel(
        s,
        bx + lw + GAP,
        y + BAND_HEAD,
        rw2,
        DERISK_ITEMS,
        "De-risked, not deferred",
        "P5 \u00b7 P6 \u00b7 P7 \u00b7 P8",
        CUDA,
        height=h6,
    )

    # ---------------------------------------------------------------- footer
    y += BAND_HEAD + h6 + 40
    s.line(MARGIN, y, WIDTH - MARGIN, y, BORDER_SOFT, 1)
    s.text(
        MARGIN,
        y + 16,
        "Generated by docs/diagrams/kernel_architecture.py \u2014 every box is a file under "
        "packages/dynquant-kernels/. Regenerate rather than edit.",
        F_TINY,
        MUTED,
    )
    s.text(
        WIDTH - MARGIN,
        y + 16,
        "DynQuant \u00b7 P0 binary pipeline \u00b7 kernel ABI 1",
        F_TINY,
        MUTED,
        anchor="ra",
    )
    s.used = y + 48
    return s


def main() -> int:
    sheet = build()
    used = int(sheet.used)
    if used > sheet.h:
        raise SystemExit(f"content is {used}px but the canvas is {sheet.h}px; raise Sheet height")
    img = sheet.img.crop((0, 0, WIDTH * SS, used * SS))
    out = Path(__file__).resolve().parents[1] / "images" / "cuda-kernel-architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize((WIDTH, used), Image.LANCZOS).save(out, "PNG", optimize=True)
    print(f"{out}  {WIDTH}x{used}  {out.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
