"""Where things sit in a Stata ``.dta`` file, and how to change them there.

pandas and pyreadstat read the rows of a .dta file and most of its labels,
and pandas writes one.  Neither keeps everything Stata stores, so a file that
is read and written back loses:

* the *names* of value-label sets (pandas names each set after its variable,
  so a set shared by twenty yes/no items becomes twenty sets);
* which extended missing value (``.a`` ... ``.z``) a row held;
* notes and other characteristics (``xtset`` / ``tsset`` declarations live
  there);
* display formats other than the unit of a date.

This module reads those from the file and puts them back into a file pandas
has written.  It also turns a format 120 / 121 file (Stata 18, alias
variables), which neither reader opens, into the 118 / 119 file it is once
the alias variables are left out.

Written from StataCorp's published format documentation (``help dta``), and
checked against files written by Stata itself.  Formats 113-115 (Stata 8-12)
and 117-121 (Stata 13-18) are covered.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility

Code = Union[int, str]

#: In a value-label table ``.`` is this integer and ``.a`` ... ``.z`` follow.
_MISSING_BASE = 2147483621
_NOTE_RE = re.compile(r"note(\d+)\Z")
#: Type code of an alias variable (format 120 / 121); it holds no data.
_ALIAS = 65525
_STRL = 32768

# name of the storage type and its width in a row, by type code
_MODERN_NUMERIC = {
    65526: ("double", 8),
    65527: ("float", 4),
    65528: ("long", 4),
    65529: ("int", 2),
    65530: ("byte", 1),
}
_LEGACY_NUMERIC = {
    251: ("byte", 1),
    252: ("int", 2),
    253: ("long", 4),
    254: ("float", 4),
    255: ("double", 8),
}
#: The code of ``.`` for each numeric type (``help dta``); ``.a`` ... ``.z``
#: are the next 26 integers, or the next 26 steps of the mantissa for floats.
_INT_MISSING = {"byte": 101, "int": 32741, "long": 2147483621}
_FLOAT_MISSING = 0x7F000000
_FLOAT_STEP = 0x800
_DOUBLE_MISSING = 0x7FE0000000000000
_DOUBLE_STEP = 0x0000010000000000

# Display formats Stata accepts (``help format``).
_NUMERIC_FORMAT_RE = re.compile(
    r"^%-?0?\d+[.,]\d+[efg]c?$|^%21x$|^%(8|16)[HL]$|^%-?[tdTD][^\0]*$"
)
_STRING_FORMAT_RE = re.compile(r"^%[-~]?\d+s$")


@dataclass(frozen=True)
class _Record:
    """One value-label set as it sits in the file."""

    name: str
    table: Dict[Code, str]


@dataclass
class DtaLayout:
    """Offsets and metadata of one .dta file."""

    release: int
    endian: str
    size: int
    n_vars: int
    n_obs: int
    names: List[str]
    #: storage type per variable: ``byte`` ... ``double``, ``str12``, ``strL``,
    #: ``alias``
    types: List[str]
    #: bytes each variable takes in one row
    widths: List[int]
    formats: List[str]
    format_offset: int
    format_width: int
    set_names: List[str]
    set_offset: int
    name_width: int
    #: (variable or ``_dta``, name, text) for every characteristic
    chars: List[Tuple[str, str, str]]
    #: [start, end) of the characteristic entries (tags / terminator excluded)
    chars_at: Tuple[int, int]
    data_offset: int
    #: [start, end) of the value-label records
    vl_at: Tuple[int, int]
    records: List[_Record]
    #: file offset of the 14 map entries and their values (format 117+)
    map_at: Optional[int] = None
    map: Optional[Tuple[int, ...]] = None
    raw_types: bytes = b""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def tagged(self) -> bool:
        return self.map is not None

    @property
    def row_width(self) -> int:
        return sum(self.widths)

    @property
    def text_encoding(self) -> str:
        return "utf-8" if self.release >= 118 else "latin-1"


# ------------------------------------------------------------------- parsing
def _malformed(what: str) -> DataInsufficient:
    return DataInsufficient(f"malformed or truncated .dta file: {what}")


def _read(handle: BinaryIO, offset: int, length: int) -> bytes:
    handle.seek(offset)
    data = handle.read(length)
    if len(data) != length:
        raise _malformed("the file ends early")
    return data


def _expect(data: bytes, pos: int, tag: bytes) -> int:
    if data[pos : pos + len(tag)] != tag:
        raise _malformed(f"expected {tag.decode('ascii')} at byte {pos}")
    return pos + len(tag)


def _decode(raw: bytes, release: int) -> str:
    end = raw.find(b"\0")
    if end >= 0:
        raw = raw[:end]
    if release >= 118:
        return raw.decode("utf-8", errors="replace")
    return raw.decode("latin-1")


def _fields(raw: bytes, n: int, width: int, release: int) -> List[str]:
    return [_decode(raw[i * width : (i + 1) * width], release) for i in range(n)]


def _code_of(value: int) -> Code:
    if value >= _MISSING_BASE:
        return "." + ("" if value == _MISSING_BASE else chr(96 + value - _MISSING_BASE))
    return value


def _value_of(code: Code) -> int:
    if isinstance(code, str):
        return _MISSING_BASE + (ord(code[1]) - 96 if len(code) > 1 else 0)
    return int(code)


def _parse_table(raw: bytes, endian: str, release: int) -> Dict[Code, str]:
    if len(raw) < 8:
        raise _malformed("a value-label table is cut off")
    n, text_length = struct.unpack_from(endian + "ii", raw, 0)
    if n < 0 or text_length < 0 or 8 + 8 * n + text_length > len(raw):
        raise _malformed("a value-label table is inconsistent")
    offsets = struct.unpack_from(f"{endian}{n}i", raw, 8)
    values = struct.unpack_from(f"{endian}{n}i", raw, 8 + 4 * n)
    text = raw[8 + 8 * n : 8 + 8 * n + text_length]
    table: Dict[Code, str] = {}
    for off, value in zip(offsets, values):
        if 0 <= off < text_length:
            table[_code_of(value)] = _decode(text[off:], release)
    return table


def _modern_type(code: int) -> Tuple[str, int]:
    if 1 <= code <= 2045:
        return f"str{code}", code
    if code == _STRL:
        return "strL", 8
    if code == _ALIAS:
        return "alias", 0
    if code in _MODERN_NUMERIC:
        return _MODERN_NUMERIC[code]
    raise _malformed(f"unknown variable type code {code}")


def _locate_tagged(handle: BinaryIO, head: bytes, size: int) -> DtaLayout:
    pos = _expect(head, 0, b"<stata_dta><header><release>")
    try:
        release = int(head[pos : pos + 3].decode("ascii"))
    except ValueError:
        raise _malformed("unreadable release number") from None
    if release not in (117, 118, 119, 120, 121):
        raise MethodIncompatibility(f".dta format {release} is not supported")
    pos = _expect(head, pos + 3, b"</release><byteorder>")
    order = head[pos : pos + 3]
    if order not in (b"LSF", b"MSF"):
        raise _malformed("unknown byte order")
    endian = "<" if order == b"LSF" else ">"
    pos = _expect(head, pos + 3, b"</byteorder><K>")
    wide = release in (119, 121)
    k_at = pos
    (n_vars,) = struct.unpack_from(endian + ("I" if wide else "H"), head, pos)
    pos = _expect(head, pos + (4 if wide else 2), b"</K><N>")
    (n_obs,) = struct.unpack_from(endian + ("I" if release == 117 else "Q"), head, pos)
    pos = _expect(head, pos + (4 if release == 117 else 8), b"</N><label>")
    prefix = 1 if release == 117 else 2
    (label_length,) = struct.unpack_from(
        endian + ("B" if prefix == 1 else "H"), head, pos
    )
    pos = _expect(head, pos + prefix + label_length, b"</label><timestamp>")
    pos = _expect(head, pos + 1 + head[pos], b"</timestamp></header><map>")
    map_at = pos
    section = struct.unpack_from(endian + "14Q", head, pos)
    for i in range(2, 13):
        if section[i] < section[i - 1] or section[i] > size:
            raise _malformed("the section map is out of range")

    name_width = 33 if release == 117 else 129
    format_width = 49 if release == 117 else 57

    def body(index: int, tag: bytes, length: int) -> Tuple[int, bytes]:
        if _read(handle, section[index], len(tag)) != tag:
            raise _malformed(f"{tag.decode('ascii')} is not where the map says")
        offset = section[index] + len(tag)
        return offset, _read(handle, offset, length)

    _, raw_types = body(2, b"<variable_types>", 2 * n_vars)
    kinds = [_modern_type(c) for c in struct.unpack(f"{endian}{n_vars}H", raw_types)]
    _, raw_names = body(3, b"<varnames>", n_vars * name_width)
    format_offset, raw_formats = body(5, b"<formats>", n_vars * format_width)
    set_offset, raw_sets = body(6, b"<value_label_names>", n_vars * name_width)

    chars_raw = _read(handle, section[8], section[9] - section[8])
    cpos = _expect(chars_raw, 0, b"<characteristics>")
    chars_start = section[8] + cpos
    chars: List[Tuple[str, str, str]] = []
    while chars_raw[cpos : cpos + 4] == b"<ch>":
        (length,) = struct.unpack_from(endian + "I", chars_raw, cpos + 4)
        entry = chars_raw[cpos + 8 : cpos + 8 + length]
        cpos = _expect(chars_raw, cpos + 8 + length, b"</ch>")
        if length >= 2 * name_width:
            chars.append(
                (
                    _decode(entry[:name_width], release),
                    _decode(entry[name_width : 2 * name_width], release),
                    _decode(entry[2 * name_width :], release),
                )
            )
    chars_end = section[8] + cpos
    _expect(chars_raw, cpos, b"</characteristics>")

    data_offset = section[9] + len(b"<data>")
    if _read(handle, section[9], 6) != b"<data>":
        raise _malformed("<data> is not where the map says")

    vl_raw = _read(handle, section[11], section[12] - section[11])
    vpos = _expect(vl_raw, 0, b"<value_labels>")
    vl_start = section[11] + vpos
    records: List[_Record] = []
    while vl_raw[vpos : vpos + 5] == b"<lbl>":
        (length,) = struct.unpack_from(endian + "I", vl_raw, vpos + 5)
        at = vpos + 9
        name = _decode(vl_raw[at : at + name_width], release)
        table_at = at + name_width + 3
        table = _parse_table(vl_raw[table_at : table_at + length], endian, release)
        vpos = _expect(vl_raw, table_at + length, b"</lbl>")
        records.append(_Record(name, table))
    _expect(vl_raw, vpos, b"</value_labels>")

    return DtaLayout(
        release=release,
        endian=endian,
        size=size,
        n_vars=n_vars,
        n_obs=n_obs,
        names=_fields(raw_names, n_vars, name_width, release),
        types=[k[0] for k in kinds],
        widths=[k[1] for k in kinds],
        formats=_fields(raw_formats, n_vars, format_width, release),
        format_offset=format_offset,
        format_width=format_width,
        set_names=_fields(raw_sets, n_vars, name_width, release),
        set_offset=set_offset,
        name_width=name_width,
        chars=chars,
        chars_at=(chars_start, chars_end),
        data_offset=data_offset,
        vl_at=(vl_start, section[11] + vpos),
        records=records,
        map_at=map_at,
        map=section,
        raw_types=raw_types,
        extra={"k_at": k_at, "wide": wide},
    )


def _locate_legacy(handle: BinaryIO, head: bytes, size: int) -> DtaLayout:
    release = head[0]
    if head[1] not in (1, 2):
        raise _malformed("unknown byte order")
    endian = "<" if head[1] == 2 else ">"
    if len(head) < 109:
        raise _malformed("the header is cut off")
    (n_vars,) = struct.unpack_from(endian + "H", head, 4)
    (n_obs,) = struct.unpack_from(endian + "I", head, 6)
    format_width = 12 if release == 113 else 49
    desc_length = n_vars * (1 + 33 + format_width + 33 + 81) + 2 * (n_vars + 1)
    desc = _read(handle, 109, desc_length)
    types: List[str] = []
    widths: List[int] = []
    for code in desc[:n_vars]:
        if 1 <= code <= 244:
            types.append(f"str{code}")
            widths.append(code)
        elif code in _LEGACY_NUMERIC:
            types.append(_LEGACY_NUMERIC[code][0])
            widths.append(_LEGACY_NUMERIC[code][1])
        else:
            raise _malformed(f"unknown variable type code {code}")
    at = n_vars
    raw_names = desc[at : at + n_vars * 33]
    at += n_vars * 33 + 2 * (n_vars + 1)
    format_offset = 109 + at
    raw_formats = desc[at : at + n_vars * format_width]
    at += n_vars * format_width
    set_offset = 109 + at
    raw_sets = desc[at : at + n_vars * 33]

    # expansion fields: (type, length, contents) records ending in 0/0
    chars: List[Tuple[str, str, str]] = []
    pos = 109 + desc_length
    chars_start = pos
    while True:
        kind, length = struct.unpack(endian + "BI", _read(handle, pos, 5))
        if kind == 0 and length == 0:
            break
        pos += 5
        if pos + length > size:
            raise _malformed("an expansion field overruns the end of the file")
        if kind == 1 and length >= 66:
            entry = _read(handle, pos, length)
            chars.append(
                (
                    _decode(entry[:33], release),
                    _decode(entry[33:66], release),
                    _decode(entry[66:], release),
                )
            )
        pos += length
    chars_end = pos
    data_offset = pos + 5

    data_end = data_offset + n_obs * sum(widths)
    if data_end > size:
        raise _malformed("the data section is cut off")
    tail = _read(handle, data_end, size - data_end)
    records: List[_Record] = []
    tpos = 0
    while tpos < len(tail):
        if tpos + 40 > len(tail):
            raise _malformed("the value-label section is cut off")
        (length,) = struct.unpack_from(endian + "i", tail, tpos)
        end = tpos + 40 + length
        if length < 8 or end > len(tail):
            raise _malformed("the value-label section is cut off")
        records.append(
            _Record(
                _decode(tail[tpos + 4 : tpos + 37], release),
                _parse_table(tail[tpos + 40 : end], endian, release),
            )
        )
        tpos = end

    return DtaLayout(
        release=release,
        endian=endian,
        size=size,
        n_vars=n_vars,
        n_obs=n_obs,
        names=_fields(raw_names, n_vars, 33, release),
        types=types,
        widths=widths,
        formats=_fields(raw_formats, n_vars, format_width, release),
        format_offset=format_offset,
        format_width=format_width,
        set_names=_fields(raw_sets, n_vars, 33, release),
        set_offset=set_offset,
        name_width=33,
        chars=chars,
        chars_at=(chars_start, chars_end),
        data_offset=data_offset,
        vl_at=(data_end, size),
        records=records,
    )


def dta_release(path: Any) -> Optional[int]:
    """The format number of the .dta file at ``path``; ``None`` if unreadable."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(40)
    except (OSError, TypeError):
        return None
    if head.startswith(b"<stata_dta><header><release>"):
        try:
            return int(head[28:31].decode("ascii"))
        except ValueError:
            return None
    if head and 102 <= head[0] <= 115:
        return int(head[0])
    return None


def read_layout(path: Any) -> DtaLayout:
    """Parse the metadata of the .dta file at ``path``."""
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        head = handle.read(4096)
        if len(head) < 4:
            raise _malformed("the file is too short")
        try:
            if head[:1] == b"<":
                return _locate_tagged(handle, head, size)
            if 113 <= head[0] <= 115:
                return _locate_legacy(handle, head, size)
        except (struct.error, IndexError):
            raise _malformed("a section is cut off") from None
    if 102 <= head[0] <= 112:
        raise MethodIncompatibility(
            f".dta format {head[0]} (Stata 7 or older) is not supported here; "
            f"open it in Stata and save it again to convert it"
        )
    raise DataInsufficient("not a Stata .dta file (unrecognized header)")


# ------------------------------------------------------------ reading extras
def split_characteristics(
    chars: List[Tuple[str, str, str]],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """Notes and the other characteristics, each keyed by variable or ``_dta``.

    Stata keeps a note as the characteristic ``note<k>`` and the number of
    notes as ``note0``.
    """
    numbered: Dict[str, List[Tuple[int, str]]] = {}
    other: Dict[str, Dict[str, str]] = {}
    for owner, name, text in chars:
        m = _NOTE_RE.match(name)
        if m is None:
            other.setdefault(owner, {})[name] = text
        elif m.group(1) != "0":
            numbered.setdefault(owner, []).append((int(m.group(1)), text))
    notes = {
        owner: [text for _, text in sorted(items)] for owner, items in numbered.items()
    }
    return notes, other


def extra_attrs(path: Any, columns: Optional[List[Any]] = None) -> Dict[str, Any]:
    """The ``attrs`` entries neither pandas nor pyreadstat provide.

    ``_value_label_names`` (``{variable: name of its value-label set}``),
    ``_notes`` and ``_characteristics`` (each keyed by variable, with the
    dataset's own under ``'_dta'``), each only when present.  Restricted to
    ``columns`` when given.
    """
    found = read_layout(path)
    keep = None if columns is None else set(columns)

    def wanted(owner: str) -> bool:
        return owner == "_dta" or keep is None or owner in keep

    attrs: Dict[str, Any] = {}
    set_names = {
        name: set_name
        for name, set_name in zip(found.names, found.set_names)
        if set_name and wanted(name)
    }
    if set_names:
        attrs["_value_label_names"] = set_names
    notes, other = split_characteristics(found.chars)
    notes = {o: v for o, v in notes.items() if wanted(o)}
    other = {o: v for o, v in other.items() if wanted(o)}
    if notes:
        attrs["_notes"] = notes
    if other:
        attrs["_characteristics"] = other
    return attrs


# ----------------------------------------------------- alias variables (120+)
def without_alias_variables(path: Any) -> Tuple[Optional[str], List[str]]:
    """Copy a format 120 / 121 file as the 118 / 119 file it is without aliases.

    An alias variable (Stata 18, ``fralias add``) is a view of a variable in
    another frame; the file names it and stores none of its values.  Returns
    the path of a temporary file (the caller removes it) and the names of
    the alias variables left out.  ``(None, [])`` when ``path`` is not such
    a file.
    """
    release = dta_release(path)
    if release not in (120, 121):
        return None, []
    found = read_layout(path)
    assert found.map is not None and found.map_at is not None
    keep = [i for i, kind in enumerate(found.types) if kind != "alias"]
    dropped = [found.names[i] for i, kind in enumerate(found.types) if kind == "alias"]
    gone = set(dropped)
    new_index = {old: new for new, old in enumerate(keep)}
    endian, section = found.endian, found.map
    wide = bool(found.extra["wide"])
    index_format = "I" if wide else "H"
    index_width = 4 if wide else 2

    def pick(raw: bytes, width: int) -> bytes:
        return b"".join(raw[i * width : (i + 1) * width] for i in keep)

    with open(path, "rb") as source:

        def between(index: int, tag: bytes, length: int) -> bytes:
            return _read(source, section[index] + len(tag), length)

        n = found.n_vars
        sort_raw = between(4, b"<sortlist>", (n + 1) * index_width)
        order = struct.unpack(f"{endian}{n + 1}{index_format}", sort_raw)
        new_order = []
        for position in order:
            # 1-based positions, ended by 0; a sort on an alias ends there
            if position == 0 or (position - 1) not in new_index:
                break
            new_order.append(new_index[position - 1] + 1)
        new_order += [0] * (len(keep) + 1 - len(new_order))

        label_width = 321
        chars = b"".join(
            _encode_char(owner, name, text, found)
            for owner, name, text in found.chars
            if owner not in gone
        )
        head = bytearray(_read(source, 0, section[2]))
        head[28:31] = str(release - 2).encode("ascii")
        k_at = int(found.extra["k_at"])
        head[k_at : k_at + index_width] = struct.pack(endian + index_format, len(keep))
        pieces = [
            bytes(head),
            b"<variable_types>" + pick(found.raw_types, 2) + b"</variable_types>",
            b"<varnames>"
            + pick(between(3, b"<varnames>", n * found.name_width), found.name_width)
            + b"</varnames>",
            b"<sortlist>"
            + struct.pack(f"{endian}{len(new_order)}{index_format}", *new_order)
            + b"</sortlist>",
            b"<formats>"
            + pick(between(5, b"<formats>", n * found.format_width), found.format_width)
            + b"</formats>",
            b"<value_label_names>"
            + pick(
                between(6, b"<value_label_names>", n * found.name_width),
                found.name_width,
            )
            + b"</value_label_names>",
            b"<variable_labels>"
            + pick(between(7, b"<variable_labels>", n * label_width), label_width)
            + b"</variable_labels>",
            b"<characteristics>" + chars + b"</characteristics>",
        ]
        offsets = [0, section[1]]
        at = 0
        for piece in pieces:
            at += len(piece)
            offsets.append(at)
        # offsets now holds entries 0..9; the rest keep their distance to <data>
        shift = offsets[9] - section[9]
        offsets += [entry + shift for entry in section[10:]]
        packed = struct.pack(endian + "14Q", *offsets)
        first = bytearray(pieces[0])
        first[found.map_at : found.map_at + len(packed)] = packed
        pieces[0] = bytes(first)

        fd, temp_name = tempfile.mkstemp(prefix="statspai_", suffix=".dta")
        try:
            with os.fdopen(fd, "wb") as out:
                for piece in pieces:
                    out.write(piece)
                _copy(source, out, section[9], found.size - section[9])
        except BaseException:
            _unlink(temp_name)
            raise
    return temp_name, dropped


# ------------------------------------------------------------------ encoding
def _encode_text(text: str, found: DtaLayout, what: str) -> bytes:
    if "\0" in text:
        raise MethodIncompatibility(f"{what} contains a NUL character")
    try:
        return text.encode(found.text_encoding)
    except UnicodeEncodeError:
        raise MethodIncompatibility(
            f"{what} has characters a format-{found.release} file cannot "
            f"hold (it stores Latin-1); write with version=118 or later"
        ) from None


def _encode_name(name: str, found: DtaLayout, what: str) -> bytes:
    encoded = _encode_text(name, found, what)
    if len(encoded) > found.name_width - 1:
        raise MethodIncompatibility(
            f"{what} '{name}' takes {len(encoded)} bytes; a format-"
            f"{found.release} file has room for {found.name_width - 1}"
        )
    return encoded.ljust(found.name_width, b"\0")


def _encode_record(name: str, table: Dict[Code, str], found: DtaLayout) -> bytes:
    """One value-label set as the bytes of its record, tags included."""
    offsets: List[int] = []
    text = bytearray()
    ordered = sorted(table.items(), key=lambda item: _value_of(item[0]))
    for code, label in ordered:
        offsets.append(len(text))
        text += _encode_text(label, found, f"the label of {code} in '{name}'")
        text += b"\0"
    n = len(ordered)
    body = (
        struct.pack(found.endian + "ii", n, len(text))
        + struct.pack(f"{found.endian}{n}i", *offsets)
        + struct.pack(f"{found.endian}{n}i", *(_value_of(code) for code, _ in ordered))
        + bytes(text)
    )
    name_field = _encode_name(name, found, "value-label name") + b"\0\0\0"
    if found.tagged:
        return (
            b"<lbl>"
            + struct.pack(found.endian + "I", len(body))
            + name_field
            + body
            + b"</lbl>"
        )
    return struct.pack(found.endian + "i", len(body)) + name_field + body


def _encode_char(owner: str, name: str, text: str, found: DtaLayout) -> bytes:
    """One characteristic as the bytes of its entry."""
    body = (
        _encode_name(owner, found, "characteristic owner")
        + _encode_name(name, found, "characteristic name")
        + _encode_text(text, found, f"{owner}[{name}]")
        + b"\0"
    )
    if found.tagged:
        return b"<ch>" + struct.pack(found.endian + "I", len(body)) + body + b"</ch>"
    return struct.pack(found.endian + "BI", 1, len(body)) + body


def valid_format(fmt: str, kind: str) -> bool:
    """Whether Stata would take display format ``fmt`` for a ``kind`` variable."""
    if not isinstance(fmt, str) or not fmt.isascii() or len(fmt) > 48:
        return False
    if kind.startswith("str"):
        return bool(_STRING_FORMAT_RE.match(fmt))
    return bool(_NUMERIC_FORMAT_RE.match(fmt))


# ------------------------------------------------------------------- writing
def _unlink(name: str) -> None:
    try:
        os.unlink(name)
    except OSError:
        # the temporary file is already gone; nothing is left to clean up
        return


def _copy(source: BinaryIO, out: BinaryIO, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, 1 << 20))
        if not chunk:
            raise _malformed("the file ends early")
        out.write(chunk)
        remaining -= len(chunk)


def _insert_characteristics(path: Path, found: DtaLayout, entries: bytes) -> None:
    """Replace the characteristics of ``path``; everything after them moves."""
    start, end = found.chars_at
    grow = len(entries) - (end - start)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with open(path, "rb") as source, os.fdopen(fd, "wb") as out:
            _copy(source, out, 0, start)
            out.write(entries)
            _copy(source, out, end, found.size - end)
            if found.map is not None and found.map_at is not None:
                moved = [e + grow if e > start else e for e in found.map]
                out.seek(found.map_at)
                out.write(struct.pack(found.endian + "14Q", *moved))
        shutil.copymode(path, temp_name)
        os.replace(temp_name, path)
    except BaseException:
        _unlink(temp_name)
        raise


def _missing_bytes(kind: str, letters: np.ndarray, endian: str) -> np.ndarray:
    """Rows of bytes: the code of ``.a`` (1) ... ``.z`` (26) for a ``kind``."""
    if kind in _INT_MISSING:
        width = {"byte": 1, "int": 2, "long": 4}[kind]
        values = (_INT_MISSING[kind] + letters).astype(f"{endian}i{width}")
    elif kind == "float":
        values = (_FLOAT_MISSING + letters.astype(np.uint64) * _FLOAT_STEP).astype(
            f"{endian}u4"
        )
    elif kind == "double":
        values = (
            np.uint64(_DOUBLE_MISSING)
            + letters.astype(np.uint64) * np.uint64(_DOUBLE_STEP)
        ).astype(f"{endian}u8")
    else:
        raise MethodIncompatibility(
            f"extended missing values need a numeric variable, not {kind}"
        )
    return values.view(np.uint8).reshape(len(letters), -1)


def patch_dta(
    path: Union[str, Path],
    *,
    set_names: Optional[Dict[str, str]] = None,
    tables: Optional[Dict[str, Dict[Code, str]]] = None,
    formats: Optional[Dict[str, str]] = None,
    characteristics: Optional[List[Tuple[str, str, str]]] = None,
    missing_codes: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> None:
    """Put into a .dta file what ``DataFrame.to_stata`` leaves out.

    Parameters
    ----------
    path : str or Path
        A file pandas has just written.
    set_names : dict, optional
        ``{variable: value-label name}``; replaces every attachment.
    tables : dict, optional
        ``{value-label name: {code: text}}``; replaces every set in the file.
    formats : dict, optional
        ``{variable: display format}``; each must be valid for the variable.
    characteristics : list, optional
        ``(variable or '_dta', name, text)``; replaces those in the file.
    missing_codes : dict, optional
        ``{variable: (rows, letters)}`` with ``letters`` 1 for ``.a`` ...
        26 for ``.z``; the cells are overwritten with that missing value.
    """
    target = Path(path)
    found = read_layout(target)
    if characteristics:
        entries = b"".join(
            _encode_char(owner, name, text, found)
            for owner, name, text in characteristics
        )
        _insert_characteristics(target, found, entries)
        found = read_layout(target)
    index = {name: i for i, name in enumerate(found.names)}

    with open(target, "r+b") as handle:
        for var, fmt in (formats or {}).items():
            handle.seek(found.format_offset + index[var] * found.format_width)
            handle.write(fmt.encode("ascii").ljust(found.format_width, b"\0"))
        if set_names is not None:
            for var, i in index.items():
                handle.seek(found.set_offset + i * found.name_width)
                handle.write(
                    _encode_name(set_names.get(var, ""), found, "value-label name")
                )
        if tables is not None:
            start = found.vl_at[0]
            body = b"".join(
                _encode_record(name, table, found) for name, table in tables.items()
            )
            handle.seek(start)
            handle.truncate()
            handle.write(body)
            if found.map is not None and found.map_at is not None:
                closing = handle.tell()
                handle.write(b"</value_labels></stata_dta>")
                end = handle.tell()
                moved = list(found.map)
                moved[12] = closing + len(b"</value_labels>")
                moved[13] = end
                handle.seek(found.map_at)
                handle.write(struct.pack(found.endian + "14Q", *moved))
        handle.flush()

    if missing_codes:
        offsets = np.concatenate(([0], np.cumsum(found.widths)))
        cells = np.memmap(target, dtype=np.uint8, mode="r+")
        try:
            for var, (rows, letters) in missing_codes.items():
                i = index[var]
                raw = _missing_bytes(found.types[i], np.asarray(letters), found.endian)
                at = (
                    found.data_offset
                    + np.asarray(rows, dtype=np.int64) * found.row_width
                    + int(offsets[i])
                )
                cells[at[:, None] + np.arange(raw.shape[1])] = raw
            cells.flush()
        finally:
            del cells
