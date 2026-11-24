import functools
import itertools
import string
from collections.abc import Sequence
from typing import TypeAlias, Union, Literal
import torch
from math import pi, log
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, is_tensor, tensor, Tensor


Shape: TypeAlias = Sequence[int]

_ellipsis = "..."
_ellipsis_token = "\u2026"  # внутренний маркер ellipsis


# ---------------------------------------------------------------------------
# Ошибка
# ---------------------------------------------------------------------------
class EinopsError(RuntimeError):
    """Ошибка парсинга/применения паттерна (аналог einops.EinopsError)."""


# ---------------------------------------------------------------------------
# Разбор паттернов (упрощённый аналог einops.parsing.ParsedExpression)
# ---------------------------------------------------------------------------
class AnonymousAxis:
    """Анонимная ось, заданная числом в паттерне, например '2'."""

    def __init__(self, value: int):
        self.value = value

    def __repr__(self) -> str:
        return f"AnonymousAxis({self.value})"

    def __eq__(self, other) -> bool:
        return isinstance(other, AnonymousAxis) and self.value == other.value

    def __hash__(self) -> int:
        return hash((AnonymousAxis, self.value))


def _check_axis_name_return_reason(name: str) -> tuple[bool, str]:
    if not name:
        return False, "empty axis name"
    if not (name[0].isalpha() or name[0] == "_"):
        return False, "axis name must start with a letter or underscore"
    for ch in name:
        if not (ch.isalnum() or ch == "_"):
            return False, f"invalid character {ch!r} in axis name"
    return True, ""


class ParsedExpression:
    """Разбирает одну сторону паттерна (например 'b (h h2) w ...')."""

    def __init__(self, expression: str, allow_underscore: bool = False, allow_duplicates: bool = False):
        self.expression = expression
        self.identifiers: set[Union[str, AnonymousAxis]] = set()
        self.composition: list[Union[list[Union[str, AnonymousAxis]], str]] = []
        self.has_ellipsis = False
        self.has_ellipsis_parenthesized = False
        self.has_non_unitary_anonymous_axes = False
        self._allow_underscore = allow_underscore
        self._allow_duplicates = allow_duplicates
        self._parse(expression)

    def _parse(self, expression: str) -> None:
        expr = expression.strip()
        if not expr:
            return
        # Токенизация: скобки, многоточие, имена
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch == "(":
                # найти закрывающую скобку
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if expr[j] == "(":
                        depth += 1
                    elif expr[j] == ")":
                        depth -= 1
                    j += 1
                if depth != 0:
                    raise EinopsError(f"Unbalanced parenthesis in {expression!r}")
                inner = expr[i + 1 : j - 1].strip()
                if _ellipsis in inner:
                    self.has_ellipsis_parenthesized = True
                    raise EinopsError(
                        f"Ellipsis inside parenthesis is not allowed: {expression!r}"
                    )
                group = self._parse_group(inner)
                self.composition.append(group)
                i = j
            elif expr.startswith("...", i):
                self.has_ellipsis = True
                self.composition.append(_ellipsis)
                i += 3
            else:
                # имя оси (или число)
                j = i
                while j < n and not expr[j].isspace() and expr[j] not in "()":
                    j += 1
                token = expr[i:j]
                axis = self._parse_token(token)
                self.composition.append([axis])
                i = j

    def _parse_group(self, inner: str) -> list[Union[str, AnonymousAxis]]:
        parts = inner.split()
        group: list[Union[str, AnonymousAxis]] = []
        for token in parts:
            if token == _ellipsis:
                raise EinopsError("Ellipsis inside parenthesis is not allowed")
            group.append(self._parse_token(token))
        return group

    def _parse_token(self, token: str) -> Union[str, AnonymousAxis]:
        # Число -> анонимная ось
        if token.isdigit():
            value = int(token)
            if value <= 0:
                raise EinopsError(f"Anonymous axis must be positive, got {token!r}")
            axis = AnonymousAxis(value)
            if value != 1:
                self.has_non_unitary_anonymous_axes = True
            return axis
        # Имя оси
        if token == "_" and self._allow_underscore:
            return "_"
        ok, reason = _check_axis_name_return_reason(token)
        if not ok:
            raise EinopsError(f"Invalid axis name {token!r}: {reason}")
        if token in self.identifiers and not self._allow_duplicates:
            raise EinopsError(f"Duplicate axis name {token!r}")
        self.identifiers.add(token)
        return token

    def has_composed_axes(self) -> bool:
        return any(isinstance(c, list) and len(c) > 1 for c in self.composition)


def _validate_axis_name(name: str) -> None:
    ok, reason = _check_axis_name_return_reason(name)
    if not ok:
        raise EinopsError(f"Invalid axis name {name!r}: {reason}")


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=128)
def _analyze_pattern(pattern: str, opname: str) -> tuple[int, int, int]:
    axes = pattern.split()
    if len(axes) != len(set(axes)):
        raise EinopsError(f'Duplicates in axes names in {opname}(..., "{pattern}")')
    if "*" not in axes:
        raise EinopsError(f'No *-axis in {opname}(..., "{pattern}")')
    for axis in axes:
        if axis != "*":
            _validate_axis_name(axis)
    n_before = axes.index("*")
    n_after = len(axes) - n_before - 1
    return n_before, n_after, n_before + n_after


def pack(tensors: Sequence[torch.Tensor], pattern: str) -> tuple[torch.Tensor, list[Shape]]:
    """
    Упаковывает несколько тензоров в один, конкатенируя оси, попавшие на '*'.

    Возвращает (packed_tensor, packed_shapes), где packed_shapes — список
    форм, которые заняли '*', для каждого входа.
    """
    if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
        raise EinopsError("pack() requires a non-empty sequence of tensors")
    n_before, n_after, min_axes = _analyze_pattern(pattern, "pack")

    reshaped: list[torch.Tensor] = []
    packed_shapes: list[Shape] = []
    for i, tensor in enumerate(tensors):
        shape = tuple(tensor.shape)
        if len(shape) < min_axes:
            raise EinopsError(
                f"packed tensor #{i} has shape {shape}, while pattern {pattern} "
                f"assumes at least {min_axes} axes"
            )
        axis_after = len(shape) - n_after
        packed_shapes.append(shape[n_before:axis_after])
        new_shape = (*shape[:n_before], -1, *shape[axis_after:])
        reshaped.append(tensor.reshape(new_shape))

    return torch.cat(reshaped, dim=n_before), packed_shapes


def _prod(x: Shape) -> int:
    result = 1
    for v in x:
        result *= v
    return result


def unpack(tensor: torch.Tensor, packed_shapes: Sequence[Shape], pattern: str) -> list[torch.Tensor]:
    """
    Обратная к pack: разбивает тензор на несколько согласно packed_shapes.
    """
    n_before, n_after, _ = _analyze_pattern(pattern, "unpack")
    input_shape = tuple(tensor.shape)
    if len(input_shape) != n_before + 1 + n_after:
        raise EinopsError(
            f"unpack(..., {pattern}) received input of wrong dim with shape {input_shape}"
        )

    unpacked_axis = n_before
    lengths = [-1 if -1 in p_shape else _prod(p_shape) for p_shape in packed_shapes]

    n_unknown = sum(1 for x in lengths if x == -1)
    if n_unknown > 1:
        raise EinopsError(
            f"unpack(..., {pattern}) received more than one -1 in {packed_shapes}"
        )

    total = input_shape[unpacked_axis]
    split_positions = [0] * len(packed_shapes) + [total]
    if n_unknown == 0:
        for i, x in enumerate(lengths[:-1]):
            split_positions[i + 1] = split_positions[i] + x
        if split_positions[-1] != total:
            raise EinopsError(
                f"unpack(..., {pattern}): sum of packed shapes {sum(lengths)} "
                f"!= axis length {total}"
            )
    else:
        unknown_idx = lengths.index(-1)
        for i in range(unknown_idx):
            split_positions[i + 1] = split_positions[i] + lengths[i]
        for j in range(unknown_idx + 1, len(lengths))[::-1]:
            split_positions[j] = split_positions[j + 1] - lengths[j]
        # проверка неотрицательности
        if any(split_positions[i] > split_positions[i + 1] for i in range(len(split_positions) - 1)):
            raise EinopsError(
                f"unpack(..., {pattern}): cannot infer -1 dimension for {packed_shapes}"
            )

    shape_start = input_shape[:unpacked_axis]
    shape_end = input_shape[unpacked_axis + 1 :]

    results = []
    for i, element_shape in enumerate(packed_shapes):
        sliced = tensor.narrow(unpacked_axis, split_positions[i], split_positions[i + 1] - split_positions[i])
        new_shape = (*shape_start, *element_shape, *shape_end)
        results.append(sliced.reshape(new_shape))
    return results


# ---------------------------------------------------------------------------
# rearrange (упрощённый reduce с reduction="rearrange")
# ---------------------------------------------------------------------------
# Внутреннее представление рецепта для rearrange
class _RearrangeRecipe:
    __slots__ = (
        "elementary_axes_lengths",      # list[int | None]
        "axis_name2elementary_axis",    # dict[str, int]
        "input_composition_known_unknown",  # list[(list[int], list[int])]
        "axes_permutation",             # list[int]
        "added_axes",                   # dict[int, int]
        "output_composite_axes",        # list[list[int]]
        "elementary_axes_order",        # list[str] — порядок элементарных осей
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs[k])


def _prepare_rearrange_recipe(pattern: str, axes_names: tuple[str, ...], ndim: int) -> _RearrangeRecipe:
    if "->" not in pattern:
        raise EinopsError(f"Pattern must contain '->': {pattern!r}")
    left_str, right_str = pattern.split("->", 1)
    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)

    if left.has_non_unitary_anonymous_axes or right.has_non_unitary_anonymous_axes:
        raise EinopsError("Non-unitary anonymous axes are not supported in rearrange")
    diff = left.identifiers.symmetric_difference(right.identifiers)
    if diff:
        raise EinopsError(f"Identifiers only on one side of expression: {diff}")

    if left.has_ellipsis:
        n_other = len(left.composition) - 1
        if ndim < n_other:
            raise EinopsError(f"Wrong shape: expected >={n_other} dims, received {ndim}")
        ell_ndim = ndim - n_other
        ell_axes = [f"{_ellipsis_token}{i}" for i in range(ell_ndim)]
        left_composition: list[list[Union[str, AnonymousAxis]]] = []
        for comp in left.composition:
            if comp == _ellipsis:
                for ax in ell_axes:
                    left_composition.append([ax])
            else:
                left_composition.append(comp)
        right_composition: list[list[Union[str, AnonymousAxis]]] = []
        for comp in right.composition:
            if comp == _ellipsis:
                for ax in ell_axes:
                    right_composition.append([ax])
            else:
                group: list[Union[str, AnonymousAxis]] = []
                for ax in comp:
                    if ax == _ellipsis:
                        group.extend(ell_axes)
                    else:
                        group.append(ax)
                right_composition.append(group)
        left.identifiers.update(ell_axes)
        left.identifiers.discard(_ellipsis)
        if right.has_ellipsis:
            right.identifiers.update(ell_axes)
            right.identifiers.discard(_ellipsis)
    else:
        if ndim != len(left.composition):
            raise EinopsError(f"Wrong shape: expected {len(left.composition)} dims, received {ndim}")
        left_composition = left.composition  # type: ignore[assignment]
        right_composition = right.composition  # type: ignore[assignment]

    # Построим элементарные оси в порядке появления слева, затем добавленные справа
    elem_axes: list[Union[str, AnonymousAxis]] = []
    axis_name2pos: dict[Union[str, AnonymousAxis], int] = {}
    lengths: list[int | None] = []

    def _add_elem(axis_name, length):
        if axis_name not in axis_name2pos:
            axis_name2pos[axis_name] = len(elem_axes)
            elem_axes.append(axis_name)
            lengths.append(length)

    for comp in left_composition:
        for ax in comp:
            if isinstance(ax, AnonymousAxis):
                _add_elem(ax, ax.value)
            else:
                _add_elem(ax, None)
    for comp in right_composition:
        for ax in comp:
            if ax not in axis_name2pos:
                if isinstance(ax, AnonymousAxis):
                    _add_elem(ax, ax.value)
                else:
                    _add_elem(ax, None)

    # Квarg'и (axes_lengths) должны быть среди осей и получают известную длину
    for name in axes_names:
        if name not in axis_name2pos:
            raise EinopsError(f"Axis {name} is not used in transform")
        lengths[axis_name2pos[name]] = -1  # маркер "задан пользователем"

    # input_composition_known_unknown: для каждой входной оси — (known, unknown) позиции
    input_comp_known_unknown: list[tuple[list[int], list[int]]] = []
    for comp in left_composition:
        known: list[int] = []
        unknown: list[int] = []
        for ax in comp:
            pos = axis_name2pos[ax]
            if lengths[pos] is not None:
                known.append(pos)
            else:
                unknown.append(pos)
        if len(unknown) > 1:
            raise EinopsError(f"Could not infer sizes for {[elem_axes[p] for p in unknown]}")
        input_comp_known_unknown.append((known, unknown))

    # Порядок осей после перестановки: сначала те, что есть в правой части (в порядке right),
    # затем "лишние" (reduced) — но в rearrange reduced быть не должно.
    ordered_left = list(itertools.chain(*left_composition))
    ordered_right = list(itertools.chain(*right_composition))
    reduced = [ax for ax in ordered_left if ax not in right.identifiers]
    if reduced:
        raise EinopsError(f"Unexpected axes removed in rearrange: {reduced}")

    order_after_transposition = [ax for ax in ordered_right if ax in left.identifiers]
    axes_permutation = [ordered_left.index(ax) for ax in order_after_transposition]

    added_axes = {
        i: axis_name2pos[ax]
        for i, ax in enumerate(ordered_right)
        if ax not in left.identifiers
    }

    # Группировка для выходной формы
    output_composite_axes: list[list[int]] = [
        [axis_name2pos[ax] for ax in comp] for comp in right_composition
    ]

    return _RearrangeRecipe(
        elementary_axes_lengths=lengths,
        axis_name2elementary_axis={name: axis_name2pos[name] for name in axes_names},
        input_composition_known_unknown=input_comp_known_unknown,
        axes_permutation=axes_permutation,
        added_axes=added_axes,
        output_composite_axes=output_composite_axes,
        elementary_axes_order=[str(a) for a in elem_axes],
    )


def _apply_rearrange_recipe(
    recipe: _RearrangeRecipe,
    tensor: torch.Tensor,
    axes_lengths: dict[str, int],
) -> torch.Tensor:
    shape = list(tensor.shape)

    # 1) Восстановить длины элементарных осей
    lengths: list[int] = []
    for v in recipe.elementary_axes_lengths:
        lengths.append(v if v is not None and v != -1 else 0)

    # Пользовательские значения
    for name, value in axes_lengths.items():
        pos = recipe.axis_name2elementary_axis[name]
        if lengths[pos] != 0:
            raise EinopsError(f"Axis {name} length specified twice")
        lengths[pos] = value

    # Вывести неизвестные из формы входа
    need_init_reshape = False
    for input_axis, (known, unknown) in enumerate(recipe.input_composition_known_unknown):
        dim = shape[input_axis]
        if not known and len(unknown) == 1:
            lengths[unknown[0]] = dim
            continue
        known_product = 1
        for p in known:
            if lengths[p] == 0:
                raise EinopsError("Cannot infer axis length: some axis unspecified")
            known_product *= lengths[p]
        if not unknown:
            if dim != known_product:
                raise EinopsError(f"Shape mismatch, {dim} != {known_product}")
        else:
            if dim % known_product != 0:
                raise EinopsError(
                    f"Shape mismatch, can't divide axis of length {dim} in chunks of {known_product}"
                )
            lengths[unknown[0]] = dim // known_product
        if len(known) + len(unknown) != 1:
            need_init_reshape = True

    if any(v == 0 for v in lengths):
        raise EinopsError("Could not infer all axis lengths")

    # 2) Начальный reshape (если нужно "разбить" составные оси)
    if need_init_reshape:
        init_shape = lengths[: len(recipe.axes_permutation)]
        tensor = tensor.reshape(init_shape)

    # 3) Перестановка
    if recipe.axes_permutation != list(range(len(recipe.axes_permutation))):
        tensor = tensor.permute(recipe.axes_permutation)

    # 4) Добавленные оси (repeat-семантика) — broadcasting
    if recipe.added_axes:
        n_axes = len(recipe.axes_permutation) + len(recipe.added_axes)
        # добавляем единичные оси на нужные позиции
        for pos, elem_pos in sorted(recipe.added_axes.items()):
            tensor = tensor.unsqueeze(pos)
        final_shape = list(tensor.shape)
        for pos, elem_pos in recipe.added_axes.items():
            final_shape[pos] = lengths[elem_pos]
        tensor = tensor.expand(*final_shape)

    # 5) Финальный reshape (если нужно "склеить" оси)
    final_shape = []
    need_final_reshape = False
    for group in recipe.output_composite_axes:
        size = 1
        for p in group:
            size *= lengths[p]
        final_shape.append(size)
        if len(group) != 1:
            need_final_reshape = True
    if need_final_reshape:
        tensor = tensor.reshape(final_shape)

    return tensor


@functools.lru_cache(maxsize=256)
def _prepare_rearrange_recipe_cached(pattern: str, axes_names: tuple[str, ...], ndim: int) -> _RearrangeRecipe:
    return _prepare_rearrange_recipe(pattern, axes_names, ndim)


def rearrange(tensor: Union[torch.Tensor, list[torch.Tensor]], pattern: str, **axes_lengths: int) -> torch.Tensor:
    """
    Reader-friendly перестановка/reshape/permute для PyTorch.
    Поддерживает многоточие, составные оси и анонимные оси длины 1.
    """
    if isinstance(tensor, list):
        if len(tensor) == 0:
            raise EinopsError("rearrange can't be applied to an empty list")
        tensor = torch.stack(tensor, dim=0)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"rearrange expects torch.Tensor, got {type(tensor)}")

    for name in axes_lengths:
        _validate_axis_name(name)

    recipe = _prepare_rearrange_recipe_cached(pattern, tuple(axes_lengths), tensor.dim())
    return _apply_rearrange_recipe(recipe, tensor, axes_lengths)


# ---------------------------------------------------------------------------
# nn.Module-обёртка
# ---------------------------------------------------------------------------
class Rearrange(torch.nn.Module):
    """torch.nn.Module-обёртка над rearrange (аналог einops.layers.torch.Rearrange)."""

    def __init__(self, pattern: str, **axes_lengths: int):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = dict(axes_lengths)
        # Валидация паттерна на этапе создания
        # (проверим на нескольких размерностях, чтобы поймать ошибки рано)
        _prepare_rearrange_recipe(pattern, tuple(axes_lengths), ndim=max(1, len(axes_lengths)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rearrange(input, self.pattern, **self.axes_lengths)

    def extra_repr(self) -> str:
        return f"pattern={self.pattern!r}, axes_lengths={self.axes_lengths}"
        
        
# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------
class _RepeatRecipe:
    __slots__ = (
        "elementary_axes_lengths",          # list[int | None]  (-1 = задаётся пользователем)
        "axis_name2elementary_axis",        # dict[str, int]
        "input_composition_known_unknown",  # list[(list[int], list[int])]
        "axes_permutation",                 # list[int]  (перестановка для входа)
        "added_axes",                       # dict[int, int]  позиция в выходе -> элементарная ось
        "output_composite_axes",            # list[list[int]]
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs[k])


def _prepare_repeat_recipe(pattern: str, axes_names: tuple[str, ...], ndim: int) -> _RepeatRecipe:
    if "->" not in pattern:
        raise EinopsError(f"Pattern must contain '->': {pattern!r}")
    left_str, right_str = pattern.split("->", 1)
    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)

    # Новые оси могут появляться только справа
    diff = left.identifiers.difference(right.identifiers)
    if diff:
        raise EinopsError(f"Unexpected identifiers on the left side of repeat: {diff}")

    # Оси справа, которых нет слева, должны иметь размер (из kwargs или быть анонимными)
    axes_without_size = set()
    for ax in right.identifiers:
        if isinstance(ax, AnonymousAxis):
            continue
        if ax not in left.identifiers and ax not in axes_names:
            axes_without_size.add(ax)
    if axes_without_size:
        raise EinopsError(f"Specify sizes for new axes in repeat: {axes_without_size}")

    # Разворачиваем ellipsis (аналогично rearrange)
    if left.has_ellipsis:
        n_other = len(left.composition) - 1
        if ndim < n_other:
            raise EinopsError(f"Wrong shape: expected >={n_other} dims, received {ndim}")
        ell_ndim = ndim - n_other
        ell_axes = [f"{_ellipsis_token}{i}" for i in range(ell_ndim)]
        left_composition: list[list[Union[str, AnonymousAxis]]] = []
        for comp in left.composition:
            if comp == _ellipsis:
                for ax in ell_axes:
                    left_composition.append([ax])
            else:
                left_composition.append(comp)
        right_composition: list[list[Union[str, AnonymousAxis]]] = []
        for comp in right.composition:
            if comp == _ellipsis:
                for ax in ell_axes:
                    right_composition.append([ax])
            else:
                group: list[Union[str, AnonymousAxis]] = []
                for ax in comp:
                    if ax == _ellipsis:
                        group.extend(ell_axes)
                    else:
                        group.append(ax)
                right_composition.append(group)
        left.identifiers.update(ell_axes)
        left.identifiers.discard(_ellipsis)
        if right.has_ellipsis:
            right.identifiers.update(ell_axes)
            right.identifiers.discard(_ellipsis)
    else:
        if ndim != len(left.composition):
            raise EinopsError(f"Wrong shape: expected {len(left.composition)} dims, received {ndim}")
        left_composition = left.composition  # type: ignore[assignment]
        right_composition = right.composition  # type: ignore[assignment]

    # Элементарные оси: сначала все из левой части (в порядке появления),
    # затем новые из правой.
    elem_axes: list[Union[str, AnonymousAxis]] = []
    axis_name2pos: dict[Union[str, AnonymousAxis], int] = {}
    lengths: list[int | None] = []

    def _add_elem(axis_name, length):
        if axis_name not in axis_name2pos:
            axis_name2pos[axis_name] = len(elem_axes)
            elem_axes.append(axis_name)
            lengths.append(length)

    for comp in left_composition:
        for ax in comp:
            if isinstance(ax, AnonymousAxis):
                _add_elem(ax, ax.value)
            else:
                _add_elem(ax, None)

    for comp in right_composition:
        for ax in comp:
            if ax not in axis_name2pos:
                if isinstance(ax, AnonymousAxis):
                    _add_elem(ax, ax.value)
                else:
                    _add_elem(ax, None)

    # Пользовательские размеры
    for name in axes_names:
        if name not in axis_name2pos:
            raise EinopsError(f"Axis {name} is not used in transform")
        lengths[axis_name2pos[name]] = -1  # маркер "задано пользователем"

    # Входные оси: известные/неизвестные
    input_comp_known_unknown: list[tuple[list[int], list[int]]] = []
    for comp in left_composition:
        known: list[int] = []
        unknown: list[int] = []
        for ax in comp:
            pos = axis_name2pos[ax]
            if lengths[pos] is not None:
                known.append(pos)
            else:
                unknown.append(pos)
        if len(unknown) > 1:
            raise EinopsError(f"Could not infer sizes for {[elem_axes[p] for p in unknown]}")
        input_comp_known_unknown.append((known, unknown))

    # Порядок для входа (учитывая возможную перестановку осей из left в right)
    ordered_left = list(itertools.chain(*left_composition))
    ordered_right = list(itertools.chain(*right_composition))

    # Оси, которые есть и слева, и справа — переставляем по порядку правой части
    axes_permutation = [ordered_left.index(ax) for ax in ordered_right if ax in left.identifiers]

    # Новые оси справа — added_axes: позиция в выходе -> элементарная ось
    added_axes = {
        i: axis_name2pos[ax]
        for i, ax in enumerate(ordered_right)
        if ax not in left.identifiers
    }

    # Группировка выходных осей
    output_composite_axes: list[list[int]] = [
        [axis_name2pos[ax] for ax in comp] for comp in right_composition
    ]

    return _RepeatRecipe(
        elementary_axes_lengths=lengths,
        axis_name2elementary_axis={name: axis_name2pos[name] for name in axes_names},
        input_composition_known_unknown=input_comp_known_unknown,
        axes_permutation=axes_permutation,
        added_axes=added_axes,
        output_composite_axes=output_composite_axes,
    )


def _apply_repeat_recipe(
    recipe: _RepeatRecipe,
    tensor: torch.Tensor,
    axes_lengths: dict[str, int],
) -> torch.Tensor:
    shape = list(tensor.shape)

    # 1) Восстановить длины элементарных осей
    lengths: list[int] = []
    for v in recipe.elementary_axes_lengths:
        lengths.append(v if v is not None and v != -1 else 0)

    for name, value in axes_lengths.items():
        pos = recipe.axis_name2elementary_axis[name]
        if lengths[pos] != 0:
            raise EinopsError(f"Axis {name} length specified twice")
        lengths[pos] = value

    # 2) Вывести неизвестные из формы входа
    need_init_reshape = False
    for input_axis, (known, unknown) in enumerate(recipe.input_composition_known_unknown):
        dim = shape[input_axis]
        if not known and len(unknown) == 1:
            lengths[unknown[0]] = dim
            continue
        known_product = 1
        for p in known:
            if lengths[p] == 0:
                raise EinopsError("Cannot infer axis length: some axis unspecified")
            known_product *= lengths[p]
        if not unknown:
            if dim != known_product:
                raise EinopsError(f"Shape mismatch, {dim} != {known_product}")
        else:
            if dim % known_product != 0:
                raise EinopsError(
                    f"Shape mismatch, can't divide axis of length {dim} in chunks of {known_product}"
                )
            lengths[unknown[0]] = dim // known_product
        if len(known) + len(unknown) != 1:
            need_init_reshape = True

    if any(v == 0 for v in lengths):
        raise EinopsError("Could not infer all axis lengths")

    # 3) Начальный reshape (разбиение составных осей)
    if need_init_reshape:
        init_shape = lengths[: len(recipe.axes_permutation)]
        tensor = tensor.reshape(init_shape)

    # 4) Перестановка осей
    if recipe.axes_permutation != list(range(len(recipe.axes_permutation))):
        tensor = tensor.permute(recipe.axes_permutation)

    # 5) Добавление новых осей через broadcasting
    if recipe.added_axes:
        # Добавляем единичные оси на нужные позиции
        for pos in sorted(recipe.added_axes):
            tensor = tensor.unsqueeze(pos)
        # Расширяем до нужных длин
        expand_shape = list(tensor.shape)
        for pos, elem_pos in recipe.added_axes.items():
            expand_shape[pos] = lengths[elem_pos]
        tensor = tensor.expand(*expand_shape)

    # 6) Финальный reshape (склейка составных осей)
    final_shape = []
    need_final_reshape = False
    for group in recipe.output_composite_axes:
        size = 1
        for p in group:
            size *= lengths[p]
        final_shape.append(size)
        if len(group) != 1:
            need_final_reshape = True
    if need_final_reshape:
        tensor = tensor.reshape(final_shape)

    return tensor


@functools.lru_cache(maxsize=256)
def _prepare_repeat_recipe_cached(pattern: str, axes_names: tuple[str, ...], ndim: int) -> _RepeatRecipe:
    return _prepare_repeat_recipe(pattern, axes_names, ndim)


def repeat(tensor: Union[torch.Tensor, list[torch.Tensor]], pattern: str, **axes_lengths: int) -> torch.Tensor:
    """
    Повторяет/расширяет тензор согласно паттерну (аналог einops.repeat для PyTorch).

    Оси, которых нет в левой части, но есть в правой, создаются заново и
    размножают данные (broadcast, а не копирование памяти).

    Примеры:
        >>> image = torch.randn(30, 40)
        >>> repeat(image, 'h w -> h w c', c=3).shape
        torch.Size([30, 40, 3])

        >>> repeat(image, 'h w -> (repeat h) w', repeat=2).shape
        torch.Size([60, 40])

        >>> repeat(image, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
        torch.Size([60, 80])
    """
    if isinstance(tensor, list):
        if len(tensor) == 0:
            raise EinopsError("repeat can't be applied to an empty list")
        tensor = torch.stack(tensor, dim=0)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"repeat expects torch.Tensor, got {type(tensor)}")

    for name in axes_lengths:
        _validate_axis_name(name)

    recipe = _prepare_repeat_recipe_cached(pattern, tuple(axes_lengths), tensor.dim())
    return _apply_repeat_recipe(recipe, tensor, axes_lengths)



# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as tortoise-tts was using it

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    dtype = t.dtype

    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place    
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len = 0

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_scales_seq_len = 0

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device = None, dtype = None, offset = 0):
        device = default(device, self.device)
        dtype = default(dtype, self.cached_freqs.dtype)

        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype = dtype, device = device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        freqs = self.forward(seq, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            exists(seq_len) and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales_seq_len
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len = seq_len

        return scale

    def get_axial_freqs(
        self,
        *dims,
        offsets: (
            tuple[int | float, ...] |
            Tensor |
            None
        ) = None
    ):
        Colon = slice(None)
        all_freqs = []

        # handle offset

        if exists(offsets):
            if not is_tensor(offsets):
                offsets = tensor(offsets)

            assert len(offsets) == len(dims)

        # get frequencies for each axis

        for ind, dim in enumerate(dims):

            offset = 0
            if exists(offsets):
                offset = offsets[ind]

            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            pos = pos + offset

            freqs = self.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        # concat all freqs

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs
