#!/usr/bin/env python3
"""Report each built shader variant's workgroup shape and static LDS.

Vulkan guarantees only 128 invocations per compute workgroup, 128 per dimension
and 16 KiB of workgroup memory, so `gpu_create_pipeline` checks both against the
device before creating a pipeline -- which means the numbers it checks have to
match the SPIR-V. This prints them straight out of `build/vk_spv/*.spv`, which is
the source of truth once the shaders are compiled:

    uv run python tools/shader_limits.py                 # every variant
    uv run python tools/shader_limits.py 'build/vk_spv/nnedi3*'
    uv run python tools/shader_limits.py -v              # also list the Workgroup vars
    uv run python tools/shader_limits.py --check         # fail if the host disagrees

Local size comes from OpExecutionMode LocalSize; a dimension the host supplies at
creation (OpExecutionModeId, e.g. bilateral's block or eedi3's row width) prints
as `spec`. Shared memory is the sum of every Workgroup-storage variable's type; a
variable sized by a spec constant prints a trailing `+?`, and the caller's own
expression decides that number (the host computes those from the values it feeds
the shader), so read it next to the `GpuWorkgroup` literal at the call site.

`--check` does that reading mechanically, so the table cannot rot away from the
host unobserved. It resolves each `*_spv` symbol to the `GpuWorkgroup` the filter
builds it with -- through the local `create_pipeline`/`add` helper every filter
funnels `gpu_create_pipeline` through, through named `constexpr GpuWorkgroup`s,
and through the `ident = <stem>_spv` assignments the variant switches use -- and
then compares:

  * every local-size dimension the shader fixes at compile time must equal the
    host's `x`/`y`/`z`;
  * the host may never declare less static LDS than the module uses, and where
    both sides are compile-time constants they must be equal.

A field the host computes from runtime values (bilateral's block, eedi3's row
width, nlmeans' patch radius) is reported `dependent` and not compared: those are
the caller's numbers, checked against the device at creation instead. Anything the
resolver could not attribute is reported `unchecked` with the reason, so a new
call shape shows up here rather than passing silently.
"""

import argparse
import glob
import re
import struct
import sys

# Only the opcodes and storage classes this needs; SPIR-V 1.6 unified spec.
OP_CAPABILITY = 17
OP_EXTENSION = 10
OP_TYPE_VOID, OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_FLOAT = 19, 20, 21, 22
OP_TYPE_VECTOR, OP_TYPE_MATRIX, OP_TYPE_ARRAY = 23, 24, 28
OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER = 29, 30, 32
OP_CONSTANT = 43
OP_CONSTANT_COMPOSITE = 44
OP_SPEC_CONSTANT = 50
OP_SPEC_CONSTANT_COMPOSITE = 51
OP_VARIABLE = 59
OP_DECORATE = 71
OP_EXECUTION_MODE = 16
OP_EXECUTION_MODE_ID = 331
LOCAL_SIZE_MODE = 17
LOCAL_SIZE_ID_MODE = 38
STORAGE_WORKGROUP = 4
DECORATION_BUILTIN = 11
BUILTIN_WORKGROUP_SIZE = 25


SPIRV_MAGIC = 0x07230203


class Module:
    def __init__(self, path):
        data = open(path, "rb").read()
        self.path = path
        if len(data) < 20 or len(data) % 4 != 0:
            raise ValueError("not a SPIR-V module")
        self.words = struct.unpack("<%dI" % (len(data) // 4), data)
        if self.words[0] != SPIRV_MAGIC:
            raise ValueError("not a SPIR-V module")
        self.types = {}  # id -> (opcode, operands)
        self.constants = {}  # id -> int
        self.spec_ids = set()
        self.variables = []  # (type id, storage class)
        self.composites = {}  # id -> constituent ids
        self.local_size_ids = None
        self.local_size = None
        self.workgroup_size_id = None
        self._parse()
        self.local_size = self._resolve_local_size()

    def _parse(self):
        w = self.words
        i = 5
        while i < len(w):
            count, op = w[i] >> 16, w[i] & 0xFFFF
            if count == 0:
                break
            a = list(w[i + 1 : i + count])
            if op in range(OP_TYPE_VOID, OP_TYPE_POINTER + 1):
                self.types[a[0]] = (op, a)
            elif op == OP_CONSTANT:
                self.constants[a[1]] = a[2]
            elif op in (OP_CONSTANT_COMPOSITE, OP_SPEC_CONSTANT_COMPOSITE):
                self.composites[a[1]] = a[2:]
            elif (
                op == OP_DECORATE
                and len(a) >= 3
                and a[1] == DECORATION_BUILTIN
                and a[2] == BUILTIN_WORKGROUP_SIZE
            ):
                self.workgroup_size_id = a[0]
            elif op == OP_SPEC_CONSTANT:
                self.spec_ids.add(a[1])
                if len(a) > 2:
                    self.constants[a[1]] = a[2]
            elif op == OP_VARIABLE:
                # OpVariable: result type, result id, storage class.
                self.variables.append((a[0], a[2]))
            elif op == OP_EXECUTION_MODE and a[1] == LOCAL_SIZE_MODE:
                # Plain LocalSize operands are literal sizes, not ids.
                self.local_size = tuple(a[2:5])
            elif op == OP_EXECUTION_MODE_ID and a[1] == LOCAL_SIZE_ID_MODE:
                self.local_size_ids = tuple(a[2:5])
            i += count

    def _resolve(self, operands):
        out = []
        for operand in operands:
            if operand in self.spec_ids:
                out.append("spec")
            else:
                out.append(self.constants.get(operand, operand))
        return tuple(out)

    def _resolve_local_size(self):
        """Ids resolve after the whole module is read: a spec constant may be
        declared after the execution mode that names it.

        When glslc targets pre-1.6 SPIR-V it cannot emit LocalSizeId, so a
        `local_size_x_id` becomes the WorkgroupSize builtin as a spec-constant
        composite and the execution mode is a 1x1x1 placeholder -- the builtin
        is what the pipeline runs at, so it wins here."""
        if self.workgroup_size_id is not None:
            composite = self.composites.get(self.workgroup_size_id)
            if composite is not None:
                return self._resolve(composite)
        if self.local_size is not None:
            return self.local_size
        if self.local_size_ids is None:
            return (1, 1, 1)
        return self._resolve(self.local_size_ids)

    def type_size(self, type_id, depth=0):
        """Bytes of a type, or None when a spec constant sizes it."""
        if depth > 16 or type_id not in self.types:
            return 0
        op, a = self.types[type_id]
        if op in (OP_TYPE_INT, OP_TYPE_FLOAT):
            return (a[1] + 7) // 8
        if op in (OP_TYPE_VECTOR, OP_TYPE_MATRIX):
            element = self.type_size(a[1], depth + 1)
            return None if element is None else a[2] * element
        if op == OP_TYPE_ARRAY:
            count = self.constants.get(a[2])
            element = self.type_size(a[1], depth + 1)
            return None if count is None or element is None else count * element
        if op == OP_TYPE_POINTER:
            return self.type_size(a[2], depth + 1)
        if op == OP_TYPE_STRUCT:
            total = 0
            for member in a[1:]:
                size = self.type_size(member, depth + 1)
                if size is None:
                    return None
                total += size
            return total
        if op == OP_TYPE_RUNTIME_ARRAY:
            return None
        return 0

    def shared_bytes(self):
        total, dynamic = 0, False
        for type_id, storage in self.variables:
            if storage != STORAGE_WORKGROUP:
                continue
            size = self.type_size(type_id)
            if size is None:
                dynamic = True
            else:
                total += size
        return total, dynamic

    def workgroup_vars(self):
        out = []
        for type_id, storage in self.variables:
            if storage == STORAGE_WORKGROUP:
                out.append((type_id, self.type_size(type_id)))
        return out


def embedded_variants(header):
    """Stems `build/spirv_binaries.h` embeds, i.e. what the plugin can select.

    A variant dropped from the CMake tables leaves its .spv behind in the build
    directory (nothing collects it), and those files would otherwise show up
    here as if they were live. The generated header is the list that matters.
    """
    try:
        text = open(header).read()
    except OSError:
        return None
    return set(re.findall(r"static const uint32_t ([a-z0-9_]+)_spv\[\]", text))


def path_basename(path):
    """The file name of a glob result, whichever separator the platform used.

    `glob` echoes the separators of the pattern and joins the matched tail with
    the platform's own, so a Windows run hands back `build/vk_spv\\eedi3_row.spv`
    for a forward-slash pattern. Splitting on `/` alone therefore returns the
    whole path there, which then matches no host declaration and leaves every
    module reported unchecked instead of compared.
    """
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def variant_stem(path):
    """The variant name a `.spv` path carries, i.e. its name without the suffix."""
    name = path_basename(path)
    return name[: -len(".spv")] if name.endswith(".spv") else name


# ---------------------------------------------------------------------------
# Host-side expectations (--check)
# ---------------------------------------------------------------------------

SPV_REF = re.compile(r"\b([a-z0-9_]+)_spv\b(?!_size)")
# The identifier an `=` assigns to, read backwards from the `=`: the name may be
# a member (`d->row_code`), an array element (`fused_code[0]`) or a local behind
# a declaration (`const uint32_t * vc_code`), so the tail is what matters.
ASSIGN_LHS = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*(?:->|\.)\s*([A-Za-z_][A-Za-z0-9_]*))?"
    r"(?:\s*\[[^\]]*\])?\s*$"
)
# Braces are allowed: the variant tables are braced initializer lists
# (`d->vcheck_para_code = { eedi3_16_vcheck_para_spv, ... };`).
ASSIGN_RHS = re.compile(r"\s*([^;]{0,900});")
# An identifier holds a module only when it is assigned one (or a list/ternary
# of them). `result = create_agg_pipeline(..., bm3d_agg_spv, ...)` must not make
# `result` -- and so every call that reads it -- a holder of bm3d_agg.
MODULE_RHS = re.compile(r"^[^()]*$", re.S)
WORKGROUP_LITERAL = re.compile(r"GpuWorkgroup\s*\{([^{}]*)\}", re.S)
CASTS = re.compile(r"static_cast<[^>]*>|\([a-z0-9_]+\)|(?<=[0-9])[uUlL]+\b")
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
MEMBER = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)")


def strip_comments(text):
    """Drop comments and string bodies so they cannot look like code."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)


def matching(text, start, open_ch, close_ch):
    """Index just past the `close_ch` matching the `open_ch` at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def split_args(text):
    """Split an argument list on top-level commas."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def constants(text):
    """Compile-time names the workgroup literals are written in terms of.

    `constexpr` values are matched to their `;` rather than to the end of the
    line: a `constexpr GpuWorkgroup kFusedWorkgroup {` opens a braced initializer
    that runs on for several lines.
    """
    out = {}
    for match in re.finditer(
        r"constexpr\s+[A-Za-z_][A-Za-z0-9_:<>]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*(.*?);", text, re.S
    ):
        out.setdefault(match.group(1), match.group(2).strip())
    for match in re.finditer(r"#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+([^\n]*)", text):
        out.setdefault(match.group(1), match.group(2).strip())
    return out


def eval_int(expr, consts, depth=0):
    """Integer value of a literal expression, or None if it is not constant."""
    if depth > 8:
        return None
    # A `constexpr` captures its initializer as `= 128;` while a workgroup field
    # is already bare, so normalise both before evaluating.
    expr = expr.strip().lstrip("=").strip().rstrip(";")
    expr = CASTS.sub(" ", expr).strip()
    expr = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", lambda m: _const(m, consts, depth), expr)
    if not re.fullmatch(r"[\d\s()+\-*/]+", expr):
        return None
    try:
        # Safe by construction: the fullmatch above admits digits and the four
        # arithmetic operators, and the name substitution leaves no identifier.
        value = eval(expr, {"__builtins__": {}})
    except (SyntaxError, TypeError, ZeroDivisionError):
        return None
    return int(value) if isinstance(value, (int, float)) and value == int(value) else None


def _const(match, consts, depth):
    rhs = consts.get(match.group(1))
    if rhs is None:
        return match.group(0)
    value = eval_int(rhs, consts, depth + 1)
    return match.group(0) if value is None else str(value)


def workgroup_fields(expr, consts):
    """`(x, y, z, lds)` from a `GpuWorkgroup { ... }` body; None per runtime field."""
    fields = {"x": 1, "y": 1, "z": 1, "shared_bytes": 0}
    for name, value in re.findall(r"\.([A-Za-z_]+)\s*=\s*([^,}]+)", expr):
        if name in fields:
            fields[name] = eval_int(value, consts)
    return fields["x"], fields["y"], fields["z"], fields["shared_bytes"]


def workgroup_from(text, consts):
    """Fields from `GpuWorkgroup { ... }`, a bare `{ ... }` body, or a bare list.

    The last form is what a named `constexpr GpuWorkgroup kPadWorkgroup { ... }`
    leaves once its name has been split off.
    """
    lit = WORKGROUP_LITERAL.search(text)
    if lit is None:
        lit = re.search(r"\{([^{}]*)\}", text, re.S)
    if lit is not None:
        return workgroup_fields(lit.group(1), consts)
    if re.search(r"\.[A-Za-z_]+\s*=", text):
        return workgroup_fields(text, consts)
    return None


def assignments(text):
    """`(ident, rhs)` for every `ident = ...;`, scanned from the `=` outwards.

    A single regex over the left-hand side backtracks catastrophically on real
    sources (`const uint32_t * vc_code = a ? b : c;`), so the two halves are
    matched separately: the name is the tail of what precedes the `=`, and the
    value is everything up to the statement's `;`.
    """
    for eq in re.finditer(r"(?<![=!<>+\-*/%&|^])=(?!=)", text):
        lhs = ASSIGN_LHS.search(text[max(0, eq.start() - 120) : eq.start()])
        rhs = ASSIGN_RHS.match(text, eq.end())
        if lhs is None or rhs is None:
            continue
        name = f"{lhs.group(1)}->{lhs.group(2)}" if lhs.group(2) else lhs.group(1)
        yield name, rhs.group(1)


def ident_stems(text):
    """`ident -> {stem}` for every `ident = <stem>_spv` (and through other idents).

    Covers the variant switches (`d->row_code = eedi3_16_row_spv;`,
    `fused_code[0] = dfttest_16_fused_r0_spv;`) and the one level of locals the
    call sites use (`mod`, `vc_code`, `pre_code`).
    """
    direct, via = {}, []
    for name, rhs in assignments(text):
        stems = set(SPV_REF.findall(rhs)) if MODULE_RHS.match(rhs) else set()
        if stems:
            direct.setdefault(name, set()).update(stems)
        via.append((name, rhs))
    # Fixed point: `mod = pred_n4_code;` reaches the symbols one hop away.
    for _ in range(4):
        grew = False
        for name, rhs in via:
            found = set()
            for ident in IDENT.findall(rhs):
                found |= direct.get(ident, set())
            for obj, member in MEMBER.findall(rhs):
                found |= direct.get(f"{obj}->{member}", set())
            if found - direct.get(name, set()):
                direct.setdefault(name, set()).update(found)
                grew = True
        if not grew:
            break
    return direct


def definition_sites(text):
    """`(name, params, brace)` for every function and lambda definition.

    A `(` opens a definition when the matching `)` is followed by `{` (possibly
    behind a lambda's `-> type`); anywhere else it is a call. That is what tells
    `create_pipeline(...) { ... }` apart from `create_pipeline(...);`, without
    having to know the helpers' names in advance -- and lambdas (`auto add =
    [&](...) { ... }`, eedi3's creator) are named by the variable they bind.
    """
    out = []
    for match in re.finditer(
        r"(?:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[[^\]]*\]|([A-Za-z_][A-Za-z0-9_]*))\s*\(", text
    ):
        open_i = match.end() - 1
        close_i = matching(text, open_i, "(", ")") - 1
        after = re.sub(r"^->[^{;]*", "", text[close_i + 1 :].lstrip()).lstrip()
        if not after.startswith("{"):
            continue
        out.append(
            (match.group(1) or match.group(2), text[open_i + 1 : close_i], text.index("{", close_i))
        )
    return out


def parameters(params_text):
    """`[(name, type_text)]` for a definition's parameter list, defaults stripped."""
    out = []
    for chunk in split_args(params_text):
        chunk = chunk.split("=")[0].strip()
        names = IDENT.findall(chunk)
        if not names:
            continue
        out.append((names[-1], chunk[: chunk.rfind(names[-1])]))
    return out


def pipelined_call(body):
    """The argument list the helper passes to `gpu_create_pipeline`, or None."""
    at = body.find("gpu_create_pipeline(")
    if at < 0:
        return None
    open_i = body.index("(", at)
    return split_args(body[open_i + 1 : matching(body, open_i, "(", ")") - 1])


def workgroup_declarations(text):
    """`name -> (x, y, z, lds)` for every named `GpuWorkgroup` in the file.

    Covers both the caller's `constexpr GpuWorkgroup kFusedWorkgroup { ... }` and
    the helper-local `const GpuWorkgroup workgroup { ... }` that bilateral,
    gaussblur, nlmeans and bm3d build before forwarding it.
    """
    out = {}
    for match in re.finditer(r"GpuWorkgroup\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text):
        open_i = match.end() - 1
        body = text[open_i + 1 : matching(text, open_i, "{", "}") - 1]
        out[match.group(1)] = body
    return out


def source_expectations(src_dir):
    """`stem -> [(file, (x, y, z, lds) or None)]` from the host's call sites.

    Each filter funnels creation through one local helper, so the module and the
    workgroup meet at that helper's *call sites*: the helper forwards a `code`
    parameter and a `GpuWorkgroup` one, and the symbol assigned to the module is
    what the caller passes. Resolving through the helper is what makes this
    precise -- matching identifiers loosely attributes every module to every call
    in the file, which reports the aggregation workgroup against the estimate
    kernel.
    """
    import os

    found = {}
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".cpp"):
            continue
        text = strip_comments(open(os.path.join(src_dir, name)).read())
        consts = constants(text)
        idents = ident_stems(text)
        decls = workgroup_declarations(text)
        defs = definition_sites(text)
        creators = []
        for helper, params, brace in defs:
            body = text[brace : matching(text, brace, "{", "}")]
            args = pipelined_call(body)
            if args is None:
                continue
            plist = parameters(params)
            code_i = next((i for i, (_n, t) in enumerate(plist) if "uint32_t *" in t), None)
            wg_i = next((i for i, (_n, t) in enumerate(plist) if "GpuWorkgroup" in t), None)
            # The helper's own call: a literal workgroup, the named parameter, or
            # a local `GpuWorkgroup` variable it built from the config.
            literal, forwarded = None, None
            for arg in args:
                if WORKGROUP_LITERAL.search(arg):
                    literal = workgroup_from(arg, consts)
                elif wg_i is not None and arg.strip() == plist[wg_i][0]:
                    forwarded = wg_i
                elif arg.strip() in decls:
                    literal = workgroup_from(decls[arg.strip()], consts)
            creators.append((helper, code_i, forwarded, literal))
        for helper, code_i, wg_i, literal in creators:
            if code_i is None:
                continue
            for open_i, params in call_sites(text, helper):
                args = split_args(params)
                if code_i >= len(args):
                    continue
                stems = SPV_REF.findall(args[code_i])
                if not stems:
                    ident = MEMBER.search(args[code_i].strip())
                    key = (
                        f"{ident.group(1)}->{ident.group(2)}"
                        if ident
                        else IDENT.match(args[code_i].strip()).group(0)
                        if IDENT.match(args[code_i].strip())
                        else None
                    )
                    stems = sorted(idents.get(key, set())) if key else []
                wg = literal
                if wg_i is not None and wg_i < len(args):
                    # A literal at the call site, a named `constexpr
                    # GpuWorkgroup`, or something this resolver cannot follow.
                    wg = workgroup_from(args[wg_i], consts)
                    if wg is None:
                        wg = workgroup_from(consts.get(args[wg_i].strip(), ""), consts)
                for stem in stems:
                    found.setdefault(stem, []).append((name, wg))
    return found


def call_sites(text, name):
    """`(open_paren, argument_text)` for every call to `name` in `text`."""
    out = []
    for match in re.finditer(r"\b%s\s*\(" % re.escape(name), text):
        open_i = match.end() - 1
        close_i = matching(text, open_i, "(", ")") - 1
        after = re.sub(r"^->[^{;]*", "", text[close_i + 1 :].lstrip()).lstrip()
        if after.startswith("{"):
            continue  # a definition, not a call
        out.append((open_i, text[open_i + 1 : close_i]))
    return out


def check(src_dir, modules, embedded, verbose=False):
    """Compare every module against the workgroup its host call site declares."""
    expectations = source_expectations(src_dir)
    bad, unchecked, dependent, checked = [], [], set(), []
    for stem, module in sorted(modules.items()):
        local, (shared, dynamic) = module.local_size, module.shared_bytes()
        sites = expectations.get(stem)
        if not sites:
            unchecked.append((stem, "not referenced from any src/*.cpp call site"))
            continue
        host = None
        for _file, wg in sites:
            if wg is not None:
                host = wg
                break
        if host is None:
            dependent.add(stem)
            continue
        hx, hy, hz, hlds = host
        reasons = []
        for axis, hv, sv in zip("xyz", (hx, hy, hz), local):
            if sv == "spec":
                if hv is None:
                    dependent.add(stem)
                continue
            if hv is None:
                dependent.add(stem)
            elif hv != sv:
                reasons.append(f"{axis}: host {hv}, shader {sv}")
        if hlds is None:
            dependent.add(stem)
        elif hlds < shared:
            reasons.append(f"LDS: host declares {hlds} B, shader uses {shared} B")
        elif not dynamic and hlds != shared:
            reasons.append(f"LDS: host declares {hlds} B, shader uses {shared} B")
        if reasons:
            bad.append((stem, sites[0][0], reasons))
        elif stem not in dependent:
            checked.append(stem)

    if embedded is not None:
        for stem in sorted(embedded - set(modules)):
            unchecked.append((stem, "embedded, but no .spv in the build directory"))
        for stem in sorted(set(modules) - embedded):
            unchecked.append((stem, "built, but not embedded in the header"))

    for stem, where, reasons in bad:
        print("%s: %s" % (stem, "; ".join(reasons)), file=sys.stderr)
        print("  first declared in %s" % where, file=sys.stderr)
    if unchecked:
        print("\nunchecked (%d):" % len(unchecked), file=sys.stderr)
        for stem, why in unchecked:
            print("  %-34s %s" % (stem, why), file=sys.stderr)
    if verbose:
        print("\nfully checked (%d):\n  %s" % (len(checked), " ".join(checked)))
        print("\npartly checked (%d):\n  %s" % (len(dependent), " ".join(sorted(dependent))))
    print(
        "\n%d checked, %d partly checked (host computes a field), %d unchecked, %d MISMATCHED"
        % (len(checked), len(dependent), len(unchecked), len(bad))
    )
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "pattern",
        nargs="?",
        default="build/vk_spv/*.spv",
        help="glob of SPIR-V files (default: build/vk_spv/*.spv)",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="list each Workgroup variable and its size"
    )
    ap.add_argument(
        "--header",
        default="build/spirv_binaries.h",
        help="generated header naming the embedded variants; a "
        ".spv missing from it is marked stale (default: "
        "build/spirv_binaries.h)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare every module against the GpuWorkgroup its host call "
        "site declares, and exit non-zero on a mismatch",
    )
    ap.add_argument(
        "--src",
        default="src",
        help="directory holding the host sources --check reads (default: src)",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print(
            "no SPIR-V matched %r; build first (tools/install.sh)" % args.pattern, file=sys.stderr
        )
        return 1
    embedded = embedded_variants(args.header)

    modules, printed = {}, False
    if not args.check:
        print("%-34s %-14s %-6s %s" % ("variant", "local size", "inv", "LDS bytes"))
        printed = True
    for path in paths:
        try:
            module = Module(path)
        except ValueError as exc:
            print("skipping %s: %s" % (path, exc), file=sys.stderr)
            continue
        local = module.local_size
        invocations = 0
        if all(isinstance(d, int) for d in local):
            invocations = 1
            for d in local:
                invocations *= d
        shared, dynamic = module.shared_bytes()
        name = path_basename(path)
        stem = variant_stem(path)
        modules[stem] = module
        if args.check:
            continue
        stale = (
            "  (stale: not in %s)" % args.header
            if (embedded is not None and stem not in embedded)
            else ""
        )
        print(
            "%-34s %-14s %-6s %s%s%s"
            % (
                name,
                "x".join(str(d) for d in local),
                invocations if invocations else "-",
                shared,
                " +?" if dynamic else "",
                stale,
            )
        )
        if args.verbose:
            for type_id, size in module.workgroup_vars():
                print(
                    "%-34s   Workgroup var %%%d: %s"
                    % ("", type_id, "dynamic" if size is None else "%d B" % size)
                )
    if args.check:
        if not printed:
            print("checking %d modules against %s/*.cpp" % (len(modules), args.src))
        return check(args.src, modules, embedded, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
