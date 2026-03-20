import time
from functools import wraps
from typing import Any

from vedo import Cube, Plotter, show


def time_init(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        print(f"{self.__class__.__name__}.__init__ took {end - start:.6f} seconds")
        return result

    return wrapper


class MultiFigViewer(Plotter):
    @time_init
    def __init__(self):
        kwargs: dict[str, Any] = {"size": (1200, 800), "pos": (1920, 0)}
        kwargs.update({"bg": "black", "bg2": "black"})

        super().__init__(shape=(2, 3), title="CT Viewer", sharecam=True, **kwargs)
        self.add_cubes()

        ## Seems important to reset camera position and make sure that sharecam is applied.
        # for idx in range(6):
        #     self.at(idx).show()

    def add_cubes(self):
        """
        Populate the 2x3 grid with cubes.
        """
        for idx in range(6):
            cube = Cube(
                side=1.0,
                c="tomato",
                alpha=0.8,
            )

            cube.lighting("plastic")

            self.at(idx).add(cube)

    def _share_camera_explicitly(self):
        """
        Force all subplots to use the exact same camera object.
        """
        cam = self.at(0).camera
        for i in range(1, 6):
            self.at(i).camera = cam


def main1():
    viewer = MultiFigViewer()

    ## .show() function is important to ensure that all subplots share the same camera object.
    ## Alternative .show() can be called from every subplot individually.
    viewer.show().interactive().close()


def main2():
    plotter = Plotter(
        shape=(2, 3),
        title="CT Viewer",
        sharecam=True,
        resetcam=True,
        size=(1200, 800),  # type: ignore
        pos=(1920, 0),
    )

    for idx in range(6):
        cube = Cube(
            side=1.0,
            c="tomato",
            alpha=0.8,
        )

        cube.lighting("plastic")

        plotter.at(idx).add(cube)

    plotter.show().interactive().close()


def main3():
    cubes = [
        Cube(side=1.0, c="tomato", alpha=0.8).lighting("plastic") for _ in range(6)
    ]

    show(
        [cubes[0], cubes[1], cubes[2], cubes[3], cubes[4], cubes[5]],
        sharecam=True,
        shape=(2, 3),
        title="CT Viewer",
        size=(1200, 800),  # type: ignore
        pos=(1920, 0),
    )


if __name__ == "__main__":
    main1()
    main2()
    main3()
