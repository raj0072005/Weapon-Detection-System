import tempfile
import unittest
from pathlib import Path

from scripts.fix_labels import find_empty_label_files, remove_empty_label_files


class FixLabelsTests(unittest.TestCase):
    def test_remove_empty_label_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            label_dir = root / "train" / "labels"
            label_dir.mkdir(parents=True)

            empty_path = label_dir / "empty.txt"
            empty_path.write_text("", encoding="utf-8")

            populated_path = label_dir / "populated.txt"
            populated_path.write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")

            empty_files = find_empty_label_files(root)
            self.assertEqual([path.name for path in empty_files], ["empty.txt"])

            removed = remove_empty_label_files(root)
            self.assertEqual([path.name for path in removed], ["empty.txt"])
            self.assertFalse(empty_path.exists())
            self.assertTrue(populated_path.exists())


if __name__ == "__main__":
    unittest.main()
