"""The formula plugin's typesetting without an editor: the built-in typesetter (ziamath, third_party/) sets fractions, roots,
Greek letters, units and matrices into the rows draw.fill paints - M, L, C and Z only, centred, in ems; it stands in when
the computer has no LaTeX; a source it cannot read is said in words."""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kinetica-editor", "clients", "python"))
sys.path.insert(0, HERE)
import plugin  # noqa: E402


def rows_of(latex):
    return plugin._outlines(plugin._builtin(latex))


class Formula(unittest.TestCase):
    def test_the_builtin_sets_what_a_callout_needs(self):
        for latex in (r"F = ma", r"\tau = K I", r"T = 2\pi\sqrt{\frac{L}{g}}", r"m = 12\,\mathrm{kg}", r"\sum_{i=1}^n x_i",
                      r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"):
            rows = rows_of(latex)
            self.assertTrue(rows, latex)
            words = " ".join(rows)
            self.assertEqual(set(re.findall(r"[A-Za-z]", words)) - {"M", "L", "C", "Z", "e"}, set(), latex)  # e: an exponent
            numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", words)]
            self.assertLess(max(abs(v) for v in numbers), 20, latex)  # ems about the centre, not points of a page

    def test_it_stands_in_when_latex_is_missing(self):
        real = plugin._svg

        def missing(latex):
            raise plugin.Missing("LaTeX не установлен")
        plugin._svg, plugin._typeset = missing, {}
        try:
            self.assertTrue(plugin.typeset(r"P = U I"))
        finally:
            plugin._svg, plugin._typeset = real, {}

    def test_a_source_it_cannot_read_is_said(self):
        with self.assertRaises(plugin.Problem):
            plugin._builtin(r"\frac{1}{")


if __name__ == "__main__":
    unittest.main()
