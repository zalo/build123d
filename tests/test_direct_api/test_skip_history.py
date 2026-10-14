"""
build123d tests

name: test_skip_history.py
by:   Gumyr
date: October 13, 2026

desc:
    The SkipHistory context: booleans made inside it keep no record of what
    they did to their inputs, so Select.LAST and Select.NEW cannot be asked
    of them, and the results are otherwise the same.

license:

    Copyright 2026 Gumyr

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

import unittest

from build123d.build_enums import Mode, Select
from build123d.build_part import BuildPart
from build123d.geometry import Plane
from build123d.objects_part import Box, Sphere
from build123d.topology import SkipHistory, Solid


class TestSkipHistory(unittest.TestCase):
    def setUp(self):
        SkipHistory.record = True

    def test_context_manager_sets_record_false(self):
        self.assertTrue(SkipHistory.record)
        with SkipHistory():
            self.assertFalse(SkipHistory.record)
        self.assertTrue(SkipHistory.record)

    def test_exception_restores_record(self):
        try:
            with SkipHistory():
                raise ValueError("Test exception")
        except ValueError:
            pass
        self.assertTrue(SkipHistory.record)

    def test_boolean_without_record(self):
        """The result is the same shape, but does not know how it was made."""
        box = Solid.make_box(10, 20, 30, Plane((-5, -10, -15)))
        sphere = Solid.make_sphere(8)

        recorded = box.cut(sphere)
        with SkipHistory():
            unrecorded = box.cut(sphere)

        self.assertAlmostEqual(unrecorded.volume, recorded.volume, 6)
        self.assertEqual(len(unrecorded.faces()), len(recorded.faces()))
        self.assertIsNotNone(recorded._history)
        self.assertIsNone(unrecorded._history)
        self.assertEqual(len(recorded.faces(Select.LAST)), 1)
        self.assertEqual(len(unrecorded.faces(Select.ALL)), len(recorded.faces()))
        with self.assertRaises(ValueError):
            unrecorded.faces(Select.LAST)
        with self.assertRaises(ValueError):
            unrecorded.edges(Select.NEW)

    def test_builder_without_record(self):
        """A builder keeps answering, by identity: whatever is not a
        sub-shape it had before counts as last and new."""
        with BuildPart() as recorded:
            Box(10, 20, 30)
            Sphere(8, mode=Mode.SUBTRACT)
        with BuildPart() as unrecorded:
            Box(10, 20, 30)
            with SkipHistory():
                Sphere(8, mode=Mode.SUBTRACT)

        self.assertAlmostEqual(unrecorded.part.volume, recorded.part.volume, 6)
        self.assertEqual(len(recorded.faces(Select.LAST)), 1)
        self.assertEqual(len(recorded.faces(Select.NEW)), 0)
        # the spherical face and the two box faces it cut through
        self.assertEqual(len(unrecorded.faces(Select.LAST)), 3)
        self.assertEqual(len(unrecorded.faces(Select.NEW)), 3)


if __name__ == "__main__":
    unittest.main()
