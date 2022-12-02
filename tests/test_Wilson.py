import unittest
import numpy as np
from PyRlEnvs.domains.mazes import Wilson

np.random.seed(0)

# Wilson.sample((5, 5), seed=0)
"""
+---+---+---+---+---+
|                   |
+---+   +---+---+   +
|       |       |   |
+   +---+   +---+   +
|   |   |   |       |
+---+   +   +---+   +
|       |   |   |   |
+---+   +   +   +   +
|                   |
+---+---+---+---+---+
"""

class WilsonTest(unittest.TestCase):
    def test_consistency(self):
        Maze = Wilson.sample((5, 5))

        maze = Maze(0)

        s = maze.start()
        self.assertEqual(Maze.getCoords(s), (0, 0))

        r, sp, t, _ = maze.step(2)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (0, 0))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(3)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (0, 0))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (1, 0))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (2, 0))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (3, 0))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(1)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (4, 0))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (4, 1))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (4, 2))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (4, 3))
        self.assertFalse(t)

        r, sp, t, _ = maze.step(0)
        self.assertEqual(r, -1)
        self.assertEqual(Maze.getCoords(sp), (4, 4))
        self.assertTrue(t)
