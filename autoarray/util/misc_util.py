from typing import List, Optional, Type, ValuesView, Union


def has(values: Union[List, ValuesView], cls: Type) -> bool:
    """
    Does an input instance of a class have an attribute which is of type `cls`?

    Parameters
    ----------
    values
        A list or dictionary of values which instances of the input `cls` are checked to see if it has an instance
        of that class.
    cls
        The type of class that is checked if the object has an instance of.

    Returns
    -------
    A bool accoridng to whether the input clas dictionary has an instance of the input class.
    """
    for value in values:
        if isinstance(value, cls):
            return True
    return False


def cls_list_from(
    values: Union[List, ValuesView], cls: Type, cls_filtered: Optional[Type] = None
) -> List:
    """
    Returns a list of objects in a class which are an instance of the input `cls`.

    The optional `cls_filtered` input removes classes of an input instance type.

    For example:

    - If the input is `cls=aa.mesh.Mesh`, a list containing all pixelizations in the class are returned.

    - If `cls=aa.mesh.Mesh` and `cls_filtered=aa.mesh.RectangularRTUAdaptDensity`, a list of all pixelizations
    excluding those which are `RectangularRTUAdaptDensity` pixelizations will be returned.

    Parameters
    ----------
    values
        A list or dictionary of values which instances of the input `cls` are extracted from.
    cls
        The type of class that a list of instances of this class in the galaxy are returned for.
    cls_filtered
        A class type which is filtered and removed from the class list.

    Returns
    -------
    The list of objects in the galaxy that inherit from input `cls`.
    """
    cls_list = [value for value in values if isinstance(value, cls)]

    if cls_filtered is not None:
        return [value for value in cls_list if not isinstance(value, cls_filtered)]

    return cls_list


def total(values: Union[List, ValuesView], cls: Type) -> int:
    """
    Returns the total number of instances of a class dictionary have an attribute which is of type `cls`?

    Parameters
    ----------
    values
        A list or dictionary of values which instances of the input `cls` are checked to see how many instances
        of that class it contains.
    cls
        The type of class that is checked if the object has an instance of.

    Returns
    -------
    The number of instances of the input class dictionary.
    """
    return len(cls_list_from(values=values, cls=cls))
