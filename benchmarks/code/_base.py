class NeighborList:
    """Base class for neighbor list implementations.

    Subclasses must define the ``name``/``devices`` class attributes, a lazy
    ``version`` class method, and a ``run`` method performing the neighbor list
    computation. The ``full_list`` class attribute lists the supported values of
    ``full_list`` (``True`` and/or ``False``).
    """

    name = None
    devices = ("cpu",)
    full_list = (True, False)

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        if device not in self.devices:
            raise ValueError(
                f"{type(self).__name__} does not support device '{device}', "
                f"supported devices are {self.devices}"
            )

        self.device = device
        self.atoms = atoms
        self.cutoff = cutoff
        self.full_list = full_list

    @classmethod
    def version(cls):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
