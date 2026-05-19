import numpy as np


class KnownFacesStore:

    known_faces = {
        1: np.random.rand(512),
        2: np.random.rand(512),
    }

    @classmethod
    def get_all_faces(cls):
        return cls.known_faces