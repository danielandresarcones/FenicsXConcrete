from __future__ import annotations

from collections import UserDict

import pint


class Parameters(UserDict):
    """
    A class that contains physical quantities for our model. Each new entry needs to be a pint quantity.
    """

    def __setitem__(self, key: str, value: pint.Quantity):
        assert isinstance(value, (pint.Quantity, bool))
        try:
            self.data[key] = value.to_base_units()
        except AttributeError:
            self.data[key] = value

    def __add__(self, other: Parameters | None) -> Parameters:
        if other is None:
            dic = self
        else:
            dic = Parameters({**self, **other})
        return dic

    def to_magnitude(self) -> dict[str, int | str | float]:
        magnitude_dictionary = {}
        for key in self.keys():
            try:
                magnitude_dictionary[key] = self[key].magnitude
            except AttributeError:
                magnitude_dictionary[key] = self[key]

        return magnitude_dictionary
