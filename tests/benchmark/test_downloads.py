"""Linux regression tests for local shard selection, retries and row-limited conversion."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from seco.downloads import iter_local_shards, read_local_records, retry_call, select_source_files


class DownloadTests(unittest.TestCase):
    def test_card_mapping_keeps_config_split_and_lexical_shard_order(self):
        spec = {"config": "default", "split": "train"}
        files = ["dev/validation-0.parquet", "default/train-002.parquet",
                 "default/validation-0.parquet", "default/train-001.parquet"]
        card = {"configs": [{"config_name": "default", "data_files": [
            {"split": "train", "path": "default/train-*"},
            {"split": "validation", "path": "default/validation-*"}]},
            {"config_name": "dev", "data_files": [{"split": "validation", "path": "dev/*"}]}]}
        self.assertEqual(select_source_files(spec, files, card),
                         ["default/train-001.parquet", "default/train-002.parquet"])

    def test_ambiguous_split_is_rejected(self):
        with self.assertRaises(ValueError):
            select_source_files({"config": "default", "split": "test"}, ["data.json"], {})

    def test_single_declared_test_split_and_converted_branch(self):
        card = {"dataset_info": {"splits": [{"name": "test"}]}}
        self.assertEqual(select_source_files({"config": "default", "split": "test"},
                                            ["dataset_infos.json", "NQ.json"], card), ["NQ.json"])
        spec = {"config": "default", "split": "train", "converted_parquet": True}
        self.assertEqual(select_source_files(spec, ["default/train/0000.parquet", "default/test/0000.parquet"], {}),
                         ["default/train/0000.parquet"])

    def test_truncated_transfer_retries_but_403_does_not(self):
        import requests

        calls = []

        def download():
            calls.append(1)
            if len(calls) == 1:
                raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
            return "complete-local-file"

        result = retry_call(download, description="shard", attempts=3, delay=0, log=lambda message: None)
        self.assertEqual(result, "complete-local-file")
        self.assertEqual(len(calls), 2)
        response = requests.Response()
        response.status_code = 403
        with patch("requests.get", side_effect=requests.HTTPError(response=response)) as get:
            with self.assertRaises(requests.HTTPError):
                retry_call(lambda: get("https://example.invalid"), description="metadata", attempts=3,
                           delay=0, log=lambda message: None)
            self.assertEqual(get.call_count, 1)

    def test_disk_error_is_not_retried(self):
        calls = []

        def no_space():
            calls.append(1)
            raise OSError(28, "No space left on device")

        with self.assertRaises(OSError):
            retry_call(no_space, description="disk", attempts=3, delay=0)
        self.assertEqual(len(calls), 1)

    def test_only_needed_shard_is_downloaded_and_nested_values_survive(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"id": str(index), "annotations": [{"answer": "yes", "spans": [1, 3]}]}
                    for index in range(3)]
            pq.write_table(pa.Table.from_pylist(rows), root / "first.parquet")
            spec = {"repo": "example/data", "config": "default", "split": "train",
                    "data_files": {"train": ["train-*.parquet"]}}
            info = SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename=name)
                                   for name in ("train-1.parquet", "train-2.parquet")])
            fetched = []

            def fetch(**kwargs):
                fetched.append(kwargs)
                return str(root / "first.parquet")

            transport = {}
            with patch("huggingface_hub.hf_hub_download", side_effect=fetch):
                actual = list(iter_local_shards(spec, info, cache_dir=root / "cache", limit=2,
                              attempts=2, delay=0, batch_size=1, transport=transport, log=lambda message: None))
            self.assertEqual(actual, rows[:2])
            self.assertEqual([call["filename"] for call in fetched], ["train-1.parquet"])
            self.assertEqual(fetched[0]["revision"], "a" * 40)
            self.assertEqual(len(transport["shards"]), 1)

    def test_local_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, text in (("array.json", '[{"id":1},{"id":2}]'),
                               ("lines.jsonl", '{"id":1}\n{"id":2}\n')):
                path = root / name
                path.write_text(text, encoding="utf-8")
                self.assertEqual(list(read_local_records(path, name, batch_size=1)), [{"id": 1}, {"id": 2}])
