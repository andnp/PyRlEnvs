import unittest
import numpy as np
from PyRlEnvs.mazes import Wilson

np.random.seed(0)

# Wilson.sample((5, 5), seed=0)
"""
+---+---+---+---+---+
|       |           |
+   +   +   +   +---+
|   |       |       |
+   +   +---+   +---+
|   |       |       |
+---+   +---+---+   +
|       |   |   |   |
+   +   +   +   +   +
|   |           |   |
+---+---+---+---+---+
"""

class WilsonTest(unittest.TestCase):
    def test_consistency(self):
        Maze = Wilson.sample((5, 5))

        maze = Maze(0)

        s = maze.start()
        self.assertEqual(Maze.getCoords(s), (0, 0))

        r, sp, t = maze.step(2)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (0, 0))
        self.assertFalse(t)

        r, sp, t = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (0, 0))
        self.assertFalse(t)

        r, sp, t = maze.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (0, 1))
        self.assertFalse(t)

        r, sp, t = maze.step(3)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (0, 1))
        self.assertFalse(t)

        r, sp, t = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (1, 1))
        self.assertFalse(t)

        r, sp, t = maze.step(3)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (0, 1))
        self.assertFalse(t)

        r, sp, t = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (1, 1))
        self.assertFalse(t)

        r, sp, t = maze.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (1, 2))
        self.assertFalse(t)

        r, sp, t = maze.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (1, 3))
        self.assertFalse(t)

        r, sp, t = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (2, 3))
        self.assertFalse(t)

        r, sp, t = maze.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (2, 4))
        self.assertFalse(t)

        r, sp, t = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (3, 4))
        self.assertFalse(t)

        r, sp, t = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (4, 4))
        self.assertTrue(t)
