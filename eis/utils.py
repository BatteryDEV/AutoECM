# author: Raymond Gasper
from typing import List, Tuple, Set, Any


def flatten2list(
    object, unpack_types: Tuple[type, ...] = (list, tuple, set)
) -> List[Any]:
    """Takes nested python iterables and turns them into a one-dimensional list, with items expanded in their position.
    Idempotent.

    Args:
        object (List[Any] | Tuple[Any, ...] | Set[Any]): Iterable with Iterables inside
        unpack_types (Tuple[type, ...], optional): What kind of nested iterable you'd like to unpack during flattening.
            Defaults to (list, tuple, set).

    Returns:
        List[Any]: A list with the unpacked iterables expanded in place.

    Example:
        flatten2list([(1,2),(3,4)])
        > [1,2,3,4]
        flatten2list([(1,(2,3)),(4,(5,6))])
        > [1,2,3,4,5,6]
        flatten2list([(1,{2}),(3,{4})], unpack_types=(tuple,))
        > [1,{2},3,{4}]

    """
    gather = []
    for item in object:
        if isinstance(item, unpack_types):
            gather.extend(flatten2list(item, unpack_types))
        else:
            gather.append(item)
    return gather
