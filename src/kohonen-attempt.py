#!/usr/bin/env python3

from __future__ import annotations


import itertools
import math

from pathlib import Path as _Path

import numpy as np

import moderngl
import moderngl_window as mglw

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self


def interleave_buffer(ctx: moderngl.Context, *arrays):

    count = len(arrays[0])

    if any(len(arr) != count for arr in arrays[1:]):
        raise TypeError("arrays must have the same length")

    nbytes = sum(arr.nbytes for arr in arrays)

    buf = ctx.buffer(reserve=nbytes)

    offset = 0
    for i in range(count):
        for arr in arrays:
            data = arr[i]
            buf.write(data.tobytes(), offset=offset)
            offset += data.nbytes

    return buf


class MyWindow(mglw.WindowConfig):  # type: ignore

    gl_version = (3, 3)
    title = "Kohonen Attempt"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4

    resource_dir = (_Path(__file__).parent / '..' / 'data').resolve()

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

    @classmethod
    def run(cls: type[Self]) -> None:

        mglw.run_window_config(cls)


class AnimationWindow(MyWindow):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        self.__program = self.load_program(
            vertex_shader='shaders/vertex.glsl',
            geometry_shader='shaders/geometry.glsl',
            fragment_shader='shaders/fragment.glsl',
        )

        self.__program['projection'].write(np.array([  # type: ignore
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32))

        self.__program['view'].write(np.array([  # type: ignore
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32))

        self.__program['model'].write(np.array([  # type: ignore
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32))

        # self.__scope = self.ctx.scope(uniform_buffers=[])

        self.__rng = np.random.Generator(np.random.PCG64(seed=1))

        self.__coords = self.__rng.random(size=(15, 3), dtype=np.float32) * 2 - 1

        self.__orient = np.empty((len(self.__coords), 4, 4), dtype=np.float32)
        self.__orient[...] = np.identity(4, dtype=np.float32)

        self.__colors = np.fromiter(itertools.cycle([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
        ]), dtype=np.float32, count=len(self.__coords) * 3)
        self.__colors.shape = (-1, 3)

        self.__vbo_coords = self.ctx.buffer(self.__coords.data, dynamic=True)
        self.__vbo_orient = self.ctx.buffer(self.__orient.data, dynamic=True)
        self.__vbo_colors = self.ctx.buffer(self.__colors.data, dynamic=False)

        self.__vao = self.ctx.vertex_array(self.__program, [
            (self.__vbo_orient, '16f', 'in_orient'),
            (self.__vbo_coords, '3f', 'in_vert'),
            (self.__vbo_colors, '3f', 'in_color'),
        ])

        self.__force = np.zeros((len(self.__coords), 3), dtype=np.float64)

    def __update(self) -> None:

        self.__force[...] = 0.0

        for i, (coords, force) in enumerate(zip(self.__coords, self.__force)):

            pos2d = coords[:2]
            radius = math.sqrt(pos2d.dot(pos2d))

            if radius > 0.0:

                attractor = np.empty_like(coords)
                attractor[:2] = pos2d / radius
                attractor[2:] = 0.0

                force += (attractor - coords) * 0.25

            for j, other in enumerate(self.__coords):

                if i == j:
                    continue

                direction = (coords - other).astype(force.dtype, copy=False)
                sqdistance = direction.dot(direction)

                if sqdistance <= 0.0:
                    continue

                distance = math.sqrt(sqdistance)
                force += direction * (0.05 / distance / sqdistance)

        for coords, force in zip(self.__coords, self.__force):

            coords += force * 0.05

        for coords in self.__coords:

            pos2d = coords[:2]
            radius = math.sqrt(pos2d.dot(pos2d))

            if radius <= 1.0:
                continue

            pos2d /= radius
            coords[2] = 0.0

        self.__vbo_coords.write(self.__coords.data)

    def on_render(self, time: float, frame_time: float) -> None:

        self.__update()

        self.ctx.clear()
        self.ctx.point_size = 2.0

        self.__vao.render(moderngl.POINTS)


def main() -> None:

    AnimationWindow.run()


if __name__ == '__main__':
    main()
