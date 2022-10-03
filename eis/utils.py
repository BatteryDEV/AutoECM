from typing import List, Tuple, Set, Any


def flatten2list(
    object, unpack_types: Tuple[type, ...] = (list, tuple, set)
) -> List[Any]:
    gather = []
    for item in object:
        if isinstance(item, unpack_types):
            gather.extend(flatten2list(item, unpack_types))
        else:
            gather.append(item)
    return gather
