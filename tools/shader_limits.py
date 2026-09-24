#!/usr/bin/env python3
"""Report each built shader variant's workgroup shape and static LDS.

Vulkan guarantees only 128 invocations per compute workgroup, 128 per dimension
and 16 KiB of workgroup memory, so `gpu_create_pipeline` checks both against the
device before creating a pipeline -- which means the numbers it checks have to
match the SPIR-V. This prints them straight out of `build/vk_spv/*.spv`, which is
the source of truth once the shaders are compiled:

    python3 tools/shader_limits.py                 # every variant
    python3 tools/shader_limits.py 'build/vk_spv/nnedi3*'
    python3 tools/shader_limits.py -v              # also list the Workgroup vars

Local size comes from OpExecutionMode LocalSize; a dimension the host supplies at
creation (OpExecutionModeId, e.g. bilateral's block or eedi3's row width) prints
as `spec`. Shared memory is the sum of every Workgroup-storage variable's type; a
variable sized by a spec constant prints a trailing `+?`, and the caller's own
expression decides that number (the host computes those from the values it feeds
the shader), so read it next to the `GpuWorkgroup` literal at the call site.

Run it after changing a shader or a `-D` and compare against the GpuWorkgroup
literals at each `gpu_create_pipeline` call site.
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
        self.types = {}        # id -> (opcode, operands)
        self.constants = {}    # id -> int
        self.spec_ids = set()
        self.variables = []    # (type id, storage class)
        self.composites = {}   # id -> constituent ids
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
            a = list(w[i + 1:i + count])
            if op in range(OP_TYPE_VOID, OP_TYPE_POINTER + 1):
                self.types[a[0]] = (op, a)
            elif op == OP_CONSTANT:
                self.constants[a[1]] = a[2]
            elif op in (OP_CONSTANT_COMPOSITE, OP_SPEC_CONSTANT_COMPOSITE):
                self.composites[a[1]] = a[2:]
            elif (op == OP_DECORATE and len(a) >= 3 and
                  a[1] == DECORATION_BUILTIN and
                  a[2] == BUILTIN_WORKGROUP_SIZE):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pattern", nargs="?", default="build/vk_spv/*.spv",
                    help="glob of SPIR-V files (default: build/vk_spv/*.spv)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list each Workgroup variable and its size")
    ap.add_argument("--header", default="build/spirv_binaries.h",
                    help="generated header naming the embedded variants; a "
                         ".spv missing from it is marked stale (default: "
                         "build/spirv_binaries.h)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print("no SPIR-V matched %r; build first (tools/install.sh)" % args.pattern,
              file=sys.stderr)
        return 1
    embedded = embedded_variants(args.header)

    print("%-34s %-14s %-6s %s" % ("variant", "local size", "inv", "LDS bytes"))
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
        name = path.rsplit("/", 1)[-1]
        stem = name[:-len(".spv")] if name.endswith(".spv") else name
        stale = "  (stale: not in %s)" % args.header if (
            embedded is not None and stem not in embedded) else ""
        print("%-34s %-14s %-6s %s%s%s" % (
            name, "x".join(str(d) for d in local),
            invocations if invocations else "-",
            shared, " +?" if dynamic else "", stale))
        if args.verbose:
            for type_id, size in module.workgroup_vars():
                print("%-34s   Workgroup var %%%d: %s" % (
                    "", type_id, "dynamic" if size is None else "%d B" % size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
